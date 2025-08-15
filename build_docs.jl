#!/usr/bin/env julia

"""
Documentation Build Script for SimpleKernelRegression.jl built with Claude Code.

This script builds the documentation locally and optionally serves it.

Usage:
    julia build_docs.jl [serve]

Arguments:
    serve   - Start a local server to view the documentation (optional)

Examples:
    julia build_docs.jl        # Build docs only
    julia build_docs.jl serve  # Build docs and start server
"""

using Pkg

println("📚 Building SimpleKernelRegression.jl Documentation")
println("=" ^ 50)

# Check if we're in the right directory
if !isfile("Project.toml") || !isdir("docs")
    error("❌ Please run this script from the package root directory (where Project.toml is located)")
end

# Activate the docs environment
println("🔧 Activating docs environment...")
Pkg.activate("docs")

# Install/update dependencies
println("📦 Installing documentation dependencies...")
Pkg.instantiate()

# Add development version of the package
println("🔗 Adding development version of SimpleKernelRegression...")
try
    Pkg.develop(PackageSpec(path="."))
catch e
    println("⚠️  Package already in development mode or error occurred: $e")
end

# Build the documentation
println("🏗️  Building documentation...")
try
    include("docs/make.jl")
    println("✅ Documentation built successfully!")
    println("📁 Documentation available in: docs/build/")
catch e
    println("❌ Error building documentation:")
    println(e)
    exit(1)
end

# Check if user wants to serve the documentation
if length(ARGS) > 0 && ARGS[1] == "serve"
    println("\n🌐 Starting documentation server...")
    
    # Check if LiveServer is available
    try
        using LiveServer
        println("📡 Server starting at: http://localhost:8000")
        println("🔄 The server will auto-reload when files change")
        println("🛑 Press Ctrl+C to stop the server")
        
        # Serve the documentation
        serve(dir="docs/build", port=8000)
    catch e
        println("❌ Error starting server (LiveServer might not be installed):")
        println(e)
        println("\n💡 You can still view the documentation by opening:")
        println("   docs/build/index.html")
    end
else
    println("\n💡 To view the documentation:")
    println("   1. Open docs/build/index.html in your browser")
    println("   2. Or run: julia build_docs.jl serve")
end

println("\n🎉 Documentation build completed!")