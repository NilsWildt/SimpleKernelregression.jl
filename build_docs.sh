#!/bin/bash

# Simple wrapper script to build documentation
# Usage: ./build_docs.sh [serve]

echo "🚀 SimpleKernelRegression.jl Documentation Builder"
echo "=================================================="

# Check if Julia is available
if ! command -v julia &> /dev/null; then
    echo "❌ Julia is not installed or not in PATH"
    exit 1
fi

# Run the Julia documentation build script
if [ "$1" = "serve" ]; then
    echo "🌐 Building documentation and starting server..."
    julia build_docs.jl serve
else
    echo "🏗️  Building documentation..."
    julia build_docs.jl
fi