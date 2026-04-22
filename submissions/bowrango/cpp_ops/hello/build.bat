@echo off
REM Minimal CUDA extension build — sanity-checks the toolchain.
REM No DREAMPlace, no include-order games — just PyTorch + CUDA.

setlocal

set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if not exist %VCVARS% (
    echo ERROR: vcvarsall.bat not found at %VCVARS%
    goto :error
)

echo === Pinning MSVC toolset to v14.40 and Windows SDK to 10.0.22621 ===
call %VCVARS% x64 10.0.22621.0 -vcvars_ver=14.40
if errorlevel 1 goto :error

echo.
echo === Compiler check ===
cl 2>&1 | findstr /C:"Version"

echo.
echo === Building hello_cuda ===
set DISTUTILS_USE_SDK=1
cd /d "%~dp0"
uv run python setup.py build_ext --inplace
if errorlevel 1 goto :error

echo.
echo === Running test ===
uv run python test.py
if errorlevel 1 goto :error

echo.
echo === Success — toolchain is working ===
pause
exit /b 0

:error
echo.
echo === Build or test failed ===
pause
exit /b 1
