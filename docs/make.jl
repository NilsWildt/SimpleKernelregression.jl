using SimpleKernelRegression
using Documenter

DocMeta.setdocmeta!(SimpleKernelRegression, :DocTestSetup, :(using SimpleKernelRegression); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [SimpleKernelRegression],
    authors = "NilsWildt <nils.wildt@iws.uni-stuttgart.de>",
    repo = "https://github.com/NilsWildt/SimpleKernelRegression.jl/blob/{commit}{path}#{line}",
    sitename = "SimpleKernelRegression.jl",
    format = Documenter.HTML(; canonical = "https://NilsWildt.github.io/SimpleKernelRegression.jl"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/NilsWildt/SimpleKernelRegression.jl")
