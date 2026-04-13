"""
PPO with discrete grid actions for macro placement.

Follows Google's chip placement methodology (Nature 2021):
- Sequential placement of hard macros (largest first by area)
- Discrete action space: pick a grid cell on the chip canvas
- Action masking: cells causing overlap or out-of-bounds are masked
- PPO (Proximal Policy Optimization) with GAE (Generalized Advantage Estimation)

MDP formulation:
(1) States encode the partial placement via a GNN over the netlist graph.
    Node features per macro: (width, height, area, x, y, is_placed).
    The GNN produces a graph embedding (mean of node embeddings), the
    current macro's embedding, and a netlist metadata vector.
(2) Actions are grid cell indices on the chip canvas (grid_rows x grid_cols
    from the benchmark). The deconv policy outputs a fixed 128x128 spatial
    map; for grids smaller than 128, the L-shaped unused region is masked.
(3) Reward is 0 for all steps except the final step, where it is the
    negative proxy cost: -(1.0 * wirelength + 0.5 * density + 0.5 * congestion).

Network architecture (from Google's chip placement paper):
  Feature embeddings:
    - Graph conv (3 layers message-passing) → reduce mean → graph embedding
    - Current macro id → index into GNN node embeddings → fc → macro embedding
    - Netlist metadata → fc → metadata embedding
  Policy network:
    - Concat [graph_emb, macro_emb, metadata_emb] → fc → 4x4x32
    - Deconv chain: 4x4x32 → 8x8x16 → 16x16x8 → 32x32x4 → 64x64x2 → 128x128x1
    - Slice to grid_rows x grid_cols, apply placement mask → logits
  Value network:
    - Same concatenated embedding → fc → scalar

Usage:
    uv run python submissions/bowrango/train_placer.py -b ibm01 -n 200
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from placer import _load_plc


# ---------------------------------------------------------------------------
# Placement environment — discrete grid actions
# ---------------------------------------------------------------------------

class PlacementEnv:
    """Sequential macro placement environment.

    Places hard macros one at a time in largest-area-first order.
    Action = grid cell index (row-major). Macro placed at cell center.
    Action mask disables cells that would cause overlap or go out of bounds.
    Reward is sparse: 0 for all steps, -proxy_cost on the final step.
    """

    def __init__(self, benchmark_name: str):
        from macro_place.loader import load_benchmark_from_dir

        root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
        if root.exists():
            self.benchmark, self.plc = load_benchmark_from_dir(str(root))
        else:
            raise ValueError(f"Benchmark {benchmark_name} not found")

        self.n_hard = self.benchmark.num_hard_macros
        self.sizes = self.benchmark.macro_sizes[:self.n_hard].numpy().astype(np.float64)
        self.cw = float(self.benchmark.canvas_width)
        self.ch = float(self.benchmark.canvas_height)
        self.movable = self.benchmark.get_movable_mask()[:self.n_hard].numpy()

        # Discrete grid from benchmark
        self.grid_rows = self.benchmark.grid_rows
        self.grid_cols = self.benchmark.grid_cols
        self.n_actions = self.grid_rows * self.grid_cols
        self.cell_w = self.cw / self.grid_cols
        self.cell_h = self.ch / self.grid_rows

        # Placement order: largest first
        self.order = sorted(range(self.n_hard),
                            key=lambda i: -self.sizes[i, 0] * self.sizes[i, 1])
        self.movable_order = [i for i in self.order if self.movable[i]]
        self.n_movable = len(self.movable_order)

        # Build adjacency for GNN
        self._build_adjacency()

        # Per-node features for GNN: (width, height, area, x, y, is_placed) — all normalized
        self.node_feat_dim = 6
        # Global netlist metadata: (macro_count, edge_count, packing_density,
        #                           h_routing_capacity, v_routing_capacity, step_progress)
        self.metadata_dim = 6

        self.reset()

    def _build_adjacency(self):
        """Build bidirectional edge index and weights from the netlist hypergraph."""
        plc = _load_plc(self.benchmark.name)
        # Adjacency as list of (i, j, weight)
        edge_dict = {}
        self.adj = [[] for _ in range(self.n_hard)]
        if plc is None:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)
            self.edge_weight = np.zeros(0, dtype=np.float32)
            return

        name_to_bidx = {}
        for bidx, idx in enumerate(plc.hard_macro_indices):
            name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx

        for driver, sinks in plc.nets.items():
            macros = set()
            for pin in [driver] + sinks:
                parent = pin.split("/")[0]
                if parent in name_to_bidx:
                    macros.add(name_to_bidx[parent])
            if len(macros) >= 2:
                ml = list(macros)
                w = 1.0 / (len(ml) - 1)
                for i in range(len(ml)):
                    for j in range(i + 1, len(ml)):
                        a, b = min(ml[i], ml[j]), max(ml[i], ml[j])
                        edge_dict[(a, b)] = edge_dict.get((a, b), 0.0) + w
                        self.adj[ml[i]].append((ml[j], w))
                        self.adj[ml[j]].append((ml[i], w))

        if edge_dict:
            edges = list(edge_dict.keys())
            # Bidirectional
            src = [e[0] for e in edges] + [e[1] for e in edges]
            dst = [e[1] for e in edges] + [e[0] for e in edges]
            wts = [edge_dict[e] for e in edges] * 2
            self.edge_index = np.array([src, dst], dtype=np.int64)
            self.edge_weight = np.array(wts, dtype=np.float32)
        else:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)
            self.edge_weight = np.zeros(0, dtype=np.float32)

    def reset(self):
        self.step_idx = 0
        self.positions = self.benchmark.macro_positions[:self.n_hard].numpy().copy()
        self.placed = np.zeros(self.n_hard, dtype=bool)
        for i in range(self.n_hard):
            if not self.movable[i]:
                self.placed[i] = True
        return self._get_state()

    def _get_node_features(self):
        """Node features for all macros: (w, h, area, x, y, is_placed) — vectorized."""
        feats = np.zeros((self.n_hard, self.node_feat_dim), dtype=np.float32)
        feats[:, 0] = self.sizes[:, 0] / self.cw
        feats[:, 1] = self.sizes[:, 1] / self.ch
        feats[:, 2] = (self.sizes[:, 0] * self.sizes[:, 1]) / (self.cw * self.ch)
        feats[:, 3] = np.where(self.placed, self.positions[:, 0] / self.cw, 0.5)
        feats[:, 4] = np.where(self.placed, self.positions[:, 1] / self.ch, 0.5)
        feats[:, 5] = self.placed.astype(np.float32)
        return feats

    def _get_metadata(self):
        """Netlist metadata vector."""
        total_area = np.sum(self.sizes[:, 0] * self.sizes[:, 1])
        return np.array([
            self.n_hard / 500.0,  # normalized macro count
            len(self.edge_weight) / 2000.0,  # normalized edge count
            total_area / (self.cw * self.ch),  # packing density
            self.benchmark.hroutes_per_micron / 100.0,
            self.benchmark.vroutes_per_micron / 100.0,
            self.step_idx / max(self.n_movable, 1),
        ], dtype=np.float32)

    def _get_state(self):
        """Return dict of state components for the policy network."""
        current_idx = (self.movable_order[self.step_idx]
                       if self.step_idx < self.n_movable else 0)
        return {
            "node_features": self._get_node_features(),
            "edge_index": self.edge_index,
            "edge_weight": self.edge_weight,
            "current_macro_idx": current_idx,
            "metadata": self._get_metadata(),
        }

    def get_action_mask(self):
        """Return boolean mask: True = valid action. Vectorized."""
        if self.step_idx >= self.n_movable:
            return np.ones(self.n_actions, dtype=bool)

        idx = self.movable_order[self.step_idx]
        mw, mh = self.sizes[idx]
        half_w, half_h = mw / 2, mh / 2

        # Grid cell centers: (n_actions,)
        cols = np.arange(self.grid_cols)
        rows = np.arange(self.grid_rows)
        cx = (cols + 0.5) * self.cell_w  # (grid_cols,)
        cy = (rows + 0.5) * self.cell_h  # (grid_rows,)
        # Broadcast to (grid_rows, grid_cols)
        cx_grid = np.broadcast_to(cx[None, :], (self.grid_rows, self.grid_cols))
        cy_grid = np.broadcast_to(cy[:, None], (self.grid_rows, self.grid_cols))

        # Bounds check
        mask = ((cx_grid - half_w >= -0.01) & (cx_grid + half_w <= self.cw + 0.01) &
                (cy_grid - half_h >= -0.01) & (cy_grid + half_h <= self.ch + 0.01))

        # Overlap check against placed macros (vectorized)
        placed_idx = np.where(self.placed)[0]
        if len(placed_idx) > 0:
            px = self.positions[placed_idx, 0]  # (P,)
            py = self.positions[placed_idx, 1]  # (P,)
            pw = self.sizes[placed_idx, 0] / 2  # (P,)
            ph = self.sizes[placed_idx, 1] / 2  # (P,)
            # For each grid cell, check overlap with all placed macros
            # dx: (grid_rows, grid_cols, P)
            dx = np.abs(cx_grid[:, :, None] - px[None, None, :])
            dy = np.abs(cy_grid[:, :, None] - py[None, None, :])
            sep_x = half_w + pw[None, None, :]  # (1, 1, P)
            sep_y = half_h + ph[None, None, :]
            overlaps = (dx < sep_x - 0.05) & (dy < sep_y - 0.05)  # (R, C, P)
            any_overlap = overlaps.any(axis=2)  # (R, C)
            mask &= ~any_overlap

        mask = mask.ravel()
        if not mask.any():
            mask[:] = True
        return mask

    def step(self, action):
        """action: int grid cell index."""
        if self.step_idx >= self.n_movable:
            return self._get_state(), 0.0, True

        idx = self.movable_order[self.step_idx]

        r = action // self.grid_cols
        c = action % self.grid_cols
        x = (c + 0.5) * self.cell_w
        y = (r + 0.5) * self.cell_h

        half_w, half_h = self.sizes[idx, 0] / 2, self.sizes[idx, 1] / 2
        x = np.clip(x, half_w, self.cw - half_w)
        y = np.clip(y, half_h, self.ch - half_h)

        self.positions[idx] = [x, y]
        self.placed[idx] = True

        self.step_idx += 1
        done = self.step_idx >= self.n_movable

        # Sparse reward: 0 for all steps except the last, where it is
        # the negative weighted sum of wirelength, congestion and density.
        if done:
            full_pos = self.benchmark.macro_positions.clone()
            full_pos[:self.n_hard] = torch.tensor(self.positions, dtype=torch.float32)
            result = compute_proxy_cost(full_pos, self.benchmark, self.plc)
            reward = -result["proxy_cost"]
        else:
            reward = 0.0

        return self._get_state(), reward, done


# ---------------------------------------------------------------------------
# Graph convolution (simple message-passing, no torch_geometric needed)
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """Single layer of message-passing graph convolution."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.msg_fc = nn.Linear(in_dim * 2 + 1, out_dim)  # +1 for edge weight
        self.update_fc = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        """x: (N, in_dim), edge_index: (2, E), edge_weight: (E,)"""
        n = x.size(0)
        if edge_index.size(1) == 0:
            return self.update_fc(torch.cat([x, torch.zeros(n, self.update_fc.in_features - x.size(1), device=x.device)], dim=-1))

        src, dst = edge_index[0], edge_index[1]
        # Compute edge messages
        src_feat = x[src]  # (E, in_dim)
        dst_feat = x[dst]  # (E, in_dim)
        edge_feat = edge_weight.unsqueeze(-1)  # (E, 1)
        msg_input = torch.cat([src_feat, dst_feat, edge_feat], dim=-1)
        messages = F.relu(self.msg_fc(msg_input))  # (E, out_dim)

        # Aggregate messages per node (mean)
        agg = torch.zeros(n, messages.size(1), device=x.device)
        count = torch.zeros(n, 1, device=x.device)
        agg.index_add_(0, dst, messages)
        count.index_add_(0, dst, torch.ones(dst.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count

        # Update node features
        updated = F.relu(self.update_fc(torch.cat([x, agg], dim=-1)))
        return updated


# ---------------------------------------------------------------------------
# Policy and value networks (matching Google's architecture)
# ---------------------------------------------------------------------------

class PlacementNetwork(nn.Module):
    """Actor-critic network for macro placement.

    Feature embeddings:
      - GNN (message-passing) over netlist graph → node embeddings
      - Graph embedding = mean-pool node embeddings
      - Current macro embedding = index into node embeddings → fc
      - Netlist metadata (macro count, edge count, area, routing) → fc

    Policy head (deconvolution):
      - Concat [graph_emb, macro_emb, meta_emb] → fc → 4x4x32
      - 5 deconv layers: 4→8→16→32→64→128 (halving channels each step)
      - Output: 128x128x1 spatial logits
      - Slice to grid_rows x grid_cols, mask invalid cells

    Value head:
      - Same concatenated embedding → fc → scalar
    """

    # Fixed spatial output size — unused cells are masked for smaller grids
    GRID_OUT = 128

    def __init__(self, node_feat_dim, metadata_dim, grid_rows, grid_cols,
                 embed_dim=32, gnn_layers=3):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.embed_dim = embed_dim

        # --- GNN encoder ---
        self.node_proj = nn.Linear(node_feat_dim, embed_dim)
        self.gnn_layers = nn.ModuleList([
            GraphConv(embed_dim, embed_dim) for _ in range(gnn_layers)
        ])

        # --- Current macro embedding ---
        self.macro_fc = nn.Linear(embed_dim, embed_dim)

        # --- Metadata embedding ---
        self.metadata_fc = nn.Linear(metadata_dim, embed_dim // 2)

        # --- Combined embedding dimension ---
        combined_dim = embed_dim + embed_dim + embed_dim // 2  # graph + macro + metadata

        # --- Policy network: fc → 4x4x32 → deconv → 128x128x1 ---
        self.policy_fc = nn.Linear(combined_dim, 4 * 4 * 32)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),   # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),    # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 4, stride=2, padding=1),    # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 4, stride=2, padding=1),    # 64→128
        )

        # Pre-compute the grid mask: True for cells that map to valid grid positions.
        # For grids smaller than 128x128, the L-shaped region beyond
        # (grid_rows, grid_cols) is masked out.
        grid_mask = torch.zeros(self.GRID_OUT, self.GRID_OUT, dtype=torch.bool)
        grid_mask[:grid_rows, :grid_cols] = True
        self.register_buffer("grid_mask", grid_mask.view(1, -1))  # (1, 128*128)

        # --- Value network ---
        self.value_fc = nn.Sequential(
            nn.Linear(combined_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def _encode(self, state):
        """Encode state dict into (graph_emb, macro_emb, metadata_emb)."""
        node_feats = state["node_features"]   # (B, N, node_feat_dim)
        edge_index = state["edge_index"]      # (2, E) long
        edge_weight = state["edge_weight"]    # (E,)
        macro_idx = state["current_macro_idx"]  # (B,) long
        metadata = state["metadata"]          # (B, metadata_dim)

        B = node_feats.shape[0]

        # Process each graph in the batch
        graph_embs = []
        macro_embs = []
        for b in range(B):
            x = self.node_proj(node_feats[b])  # (N, embed_dim)
            for gnn in self.gnn_layers:
                x = gnn(x, edge_index, edge_weight)
            # Graph embedding: mean of all node embeddings
            graph_embs.append(x.mean(dim=0))
            # Current macro embedding
            macro_embs.append(x[macro_idx[b]])

        graph_emb = torch.stack(graph_embs)  # (B, embed_dim)
        macro_emb = self.macro_fc(torch.stack(macro_embs))  # (B, embed_dim)
        meta_emb = F.relu(self.metadata_fc(metadata))  # (B, embed_dim//2)

        return graph_emb, macro_emb, meta_emb

    def forward(self, state, mask):
        """
        state: dict of batched tensors
        mask: (B, grid_rows * grid_cols) bool — placement-level mask

        Returns: logits (B, grid_rows*grid_cols), value (B, 1)
        """
        graph_emb, macro_emb, meta_emb = self._encode(state)
        combined = torch.cat([graph_emb, macro_emb, meta_emb], dim=-1)
        B = combined.size(0)

        # --- Policy: deconv to 128x128 spatial logits ---
        h = F.relu(self.policy_fc(combined))  # (B, 4*4*32)
        h = h.view(-1, 32, 4, 4)
        h = self.deconv(h)  # (B, 1, 128, 128)
        full_logits = h.view(B, self.GRID_OUT * self.GRID_OUT)  # (B, 128*128)

        # Mask the L-shaped unused region (cells beyond grid_rows x grid_cols)
        full_logits[~self.grid_mask.expand(B, -1)] = -1e8

        # Extract only the valid grid_rows x grid_cols cells
        # Map from 128x128 flat index to (r, c), keep only r < grid_rows and c < grid_cols
        logits = full_logits.view(B, self.GRID_OUT, self.GRID_OUT)
        logits = logits[:, :self.grid_rows, :self.grid_cols]
        logits = logits.reshape(B, self.grid_rows * self.grid_cols)

        # Apply placement mask (overlaps, OOB)
        logits[~mask] = -1e8

        # --- Value ---
        value = self.value_fc(combined)  # (B, 1)

        return logits, value

    def get_action(self, state_dict, mask, deterministic=False):
        """Single-step action selection. Returns (action, value, log_prob)."""
        with torch.no_grad():
            state_b = self._to_batch(state_dict)
            mask_b = torch.BoolTensor(mask).unsqueeze(0)
            logits, value = self.forward(state_b, mask_b)
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
            return action, value.item(), log_prob

    def evaluate(self, state_batch, actions, masks):
        """Evaluate actions for PPO update (batched)."""
        logits, values = self.forward(state_batch, masks)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy

    @staticmethod
    def _to_batch(state_dict):
        """Add batch dimension to a single state dict."""
        return {
            "node_features": torch.FloatTensor(state_dict["node_features"]).unsqueeze(0),
            "edge_index": torch.LongTensor(state_dict["edge_index"]),
            "edge_weight": torch.FloatTensor(state_dict["edge_weight"]),
            "current_macro_idx": torch.LongTensor([state_dict["current_macro_idx"]]),
            "metadata": torch.FloatTensor(state_dict["metadata"]).unsqueeze(0),
        }


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def push(self, state, action, reward, done, log_prob, value, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(mask)

    def compute_returns(self, gamma=0.99, lam=0.95):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1 or self.dones[t]:
                next_val = 0.0
            else:
                next_val = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val - self.values[t]
            last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def collate_batch(self, indices):
        """Collate a batch of states into batched tensors."""
        states = [self.states[i] for i in indices]
        return {
            "node_features": torch.FloatTensor(
                np.stack([s["node_features"] for s in states])),  # (B, N, F)
            "edge_index": torch.LongTensor(states[0]["edge_index"]),  # shared
            "edge_weight": torch.FloatTensor(states[0]["edge_weight"]),  # shared
            "current_macro_idx": torch.LongTensor(
                [s["current_macro_idx"] for s in states]),  # (B,)
            "metadata": torch.FloatTensor(
                np.stack([s["metadata"] for s in states])),  # (B, M)
        }

    def get_batches(self, advantages, returns, batch_size=64):
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            state_batch = self.collate_batch(idx)
            yield (
                state_batch,
                torch.LongTensor(np.array([self.actions[i] for i in idx])),
                torch.FloatTensor(np.array([self.log_probs[i] for i in idx])),
                torch.FloatTensor(advantages[idx]),
                torch.FloatTensor(returns[idx]),
                torch.BoolTensor(np.array([self.masks[i] for i in idx])),
            )

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.masks = []


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------

def train_ppo(benchmark_name: str, episodes: int = 200,
              lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
              clip_eps: float = 0.2, epochs_per_update: int = 4,
              batch_size: int = 64, embed_dim: int = 32,
              update_every: int = 5):
    """Train PPO on a single benchmark."""

    env = PlacementEnv(benchmark_name)

    policy = PlacementNetwork(
        node_feat_dim=env.node_feat_dim,
        metadata_dim=env.metadata_dim,
        grid_rows=env.grid_rows,
        grid_cols=env.grid_cols,
        embed_dim=embed_dim,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    buffer = RolloutBuffer()
    best_proxy = float("inf")
    start_ep = 0

    # Resume from checkpoint if it exists
    ckpt_path = f"submissions/bowrango/ppo_ckpt_{benchmark_name}.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_proxy = ckpt["best_proxy"]
        start_ep = ckpt["episode"]
        print(f"Resumed from episode {start_ep}, best proxy: {best_proxy:.4f}")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Training PPO on {benchmark_name}")
    print(f"  Macros: {env.n_hard} ({env.n_movable} movable)")
    print(f"  Grid: {env.grid_cols}x{env.grid_rows} = {env.n_actions} actions")
    print(f"  Network params: {n_params:,}")
    print(f"  Episodes: {start_ep} → {start_ep + episodes}")
    print()

    for ep in range(start_ep, start_ep + episodes):
        state = env.reset()
        episode_reward = 0.0

        while True:
            mask = env.get_action_mask()
            action, value, log_prob = policy.get_action(state, mask)

            next_state, reward, done = env.step(action)

            buffer.push(state, action, reward, done, log_prob, value, mask)
            episode_reward += reward
            state = next_state

            if done:
                break

        # Track best (episode_reward is -proxy_cost on final step, 0 elsewhere)
        final_proxy = -episode_reward
        if final_proxy < best_proxy:
            best_proxy = final_proxy
            torch.save(policy.state_dict(),
                        f"submissions/bowrango/ppo_policy_{benchmark_name}.pt")

        # Save checkpoint for resuming
        torch.save({
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_proxy": best_proxy,
            "episode": ep + 1,
        }, ckpt_path)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1:4d}  proxy={final_proxy:.4f}  "
                  f"best={best_proxy:.4f}  reward={episode_reward:.4f}")

        # PPO update every N episodes
        if (ep + 1) % update_every == 0:
            advantages, returns = buffer.compute_returns(gamma, lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(epochs_per_update):
                for (state_b, act_b, old_lp_b, adv_b,
                     ret_b, mask_b) in buffer.get_batches(advantages, returns, batch_size):

                    log_probs, values, entropy = policy.evaluate(state_b, act_b, mask_b)

                    ratio = (log_probs - old_lp_b).exp()
                    surr1 = ratio * adv_b
                    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values, ret_b)
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

            buffer.clear()

    print(f"\nDone. Best proxy: {best_proxy:.4f}")
    print(f"Model saved to submissions/bowrango/ppo_policy_{benchmark_name}.pt")
    return policy, best_proxy


# ---------------------------------------------------------------------------
# Inference placer
# ---------------------------------------------------------------------------

class RLPlacer:
    """Inference placer using a trained PPO policy.

    Loads saved weights, rolls out the policy deterministically
    (greedy action selection), and returns macro positions.
    """

    def __init__(self, seed: int = 42, model_path: str = None):
        self.seed = seed
        self.model_path = model_path

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        env = PlacementEnv(benchmark.name)
        policy = PlacementNetwork(
            node_feat_dim=env.node_feat_dim,
            metadata_dim=env.metadata_dim,
            grid_rows=env.grid_rows,
            grid_cols=env.grid_cols,
        )

        model_path = (self.model_path or
                      f"submissions/bowrango/ppo_policy_{benchmark.name}.pt")
        if Path(model_path).exists():
            policy.load_state_dict(torch.load(model_path, weights_only=True))
            policy.eval()

        state = env.reset()
        for _ in range(env.n_movable):
            mask = env.get_action_mask()
            action, _, _ = policy.get_action(state, mask, deterministic=True)
            state, _, done = env.step(action)
            if done:
                break

        full_pos = benchmark.macro_positions.clone()
        full_pos[:env.n_hard] = torch.tensor(env.positions, dtype=torch.float32)
        return full_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", default="ibm01")
    parser.add_argument("--episodes", "-n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--update-every", type=int, default=5)
    args = parser.parse_args()

    train_ppo(
        benchmark_name=args.benchmark,
        episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        update_every=args.update_every,
    )
