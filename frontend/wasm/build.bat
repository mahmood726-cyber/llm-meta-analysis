@echo off
REM Build WASM module for meta-analysis (Windows)

echo Building WASM module...

REM Check if wasm-pack is installed
where wasm-pack >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo wasm-pack not found. Installing...
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
)

REM Build the WASM package
wasm-pack build --target web --out-dir pkg

REM Copy the WASM files to the frontend directory
copy pkg\meta_analysis_wasm_bg.wasm ..\
copy pkg\meta_analysis_wasm.js ..\
copy pkg\meta_analysis_wasm_bg.wasm.d.ts ..\meta_analysis_wasm.d.ts 2>nul

echo WASM build complete!
echo Files generated:
echo   - meta_analysis_wasm_bg.wasm
echo   - meta_analysis_wasm.js
echo.
echo To use in dashboard.html, add:
echo ^<script src="meta_analysis_wasm.js"^>^</script^>
