#!/bin/bash
# Build WASM module for meta-analysis

echo "Building WASM module..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build the WASM package
wasm-pack build --target web --out-dir pkg

# Copy the WASM files to the frontend directory
cp pkg/meta_analysis_wasm_bg.wasm ../
cp pkg/meta_analysis_wasm.js ../
cp pkg/meta_analysis_wasm_bg.wasm.d.ts ../meta_analysis_wasm.d.ts 2>/dev/null || true

echo "WASM build complete!"
echo "Files generated:"
echo "  - meta_analysis_wasm_bg.wasm"
echo "  - meta_analysis_wasm.js"
echo ""
echo "To use in dashboard.html, add:"
echo '<script src="meta_analysis_wasm.js"></script>'
