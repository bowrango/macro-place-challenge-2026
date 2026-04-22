@echo off
REM One-shot build for DREAMPlace electric_potential extension.
REM Pins MSVC to v14.40 (CUDA 12.6's max supported toolset), sets the
REM PyTorch build env var, and runs setup.py from the script's own dir.
REM
REM Double-click to build, or run from any cmd window.

setlocal

set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if not exist %VCVARS% (
    echo ERROR: vcvarsall.bat not found at %VCVARS%
    echo Edit build.bat to point at your VS install if it's in a different edition
    echo (Professional / Enterprise / BuildTools instead of Community^).
    goto :error
)

echo === Pinning MSVC toolset to v14.40 and Windows SDK to 10.0.22621 ===
call %VCVARS% x64 10.0.22621.0 -vcvars_ver=14.40
if errorlevel 1 (
    echo ERROR: vcvarsall.bat failed. Check:
    echo  - MSVC v14.40 toolset installed ^(VS Installer -^> Individual components
    echo    -^> "MSVC v143 - VS 2022 C++ x64/x86 build tools v14.40-17.10"^)
    echo  - Windows 11 SDK 10.0.22621.0 installed ^(same place, search "22621"^)
    goto :error
)

echo.
echo === Compiler check ===
cl 2>&1 | findstr /C:"19.40" >nul
if errorlevel 1 (
    echo WARNING: cl is not v19.40.x. Build will likely fail with cmath errors.
    echo Full cl banner:
    cl 2>&1 | findstr /C:"Version"
    echo.
    echo Continuing anyway...
) else (
    cl 2>&1 | findstr /C:"Version"
)

echo.
echo === Building extension ===
set DISTUTILS_USE_SDK=1
cd /d "%~dp0"
uv run python setup.py build_ext --inplace
if errorlevel 1 goto :error

echo.
echo === Build succeeded ===
dir /b *.pyd 2>nul
pause
exit /b 0

:error
echo.
echo === Build failed ===
pause
exit /b 1
