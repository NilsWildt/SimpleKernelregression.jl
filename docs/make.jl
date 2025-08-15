using SimpleKernelRegression
using Documenter

DocMeta.setdocmeta!(SimpleKernelRegression, :DocTestSetup, :(using SimpleKernelRegression); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs")
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [SimpleKernelRegression],
    authors = "NilsWildt <nils.wildt@iws.uni-stuttgart.de>",
    repo = "https://github.com/NilsWildt/SimpleKernelRegression.jl/blob/{commit}{path}#{line}",
    sitename = "SimpleKernelRegression.jl",
    format = Documenter.HTML(; 
        canonical = "https://NilsWildt.github.io/SimpleKernelRegression.jl",
        prettyurls = true,
        edit_link = "main"
    ),
    pages = ["index.md"; numbered_pages],
)

# Only deploy if we're on the main branch
if get(ENV, "GITHUB_REPOSITORY", "") == "NilsWildt/SimpleKernelRegression.jl"
    deploydocs(; 
        repo = "github.com/NilsWildt/SimpleKernelRegression.jl",
        target = "build",
        push_preview = true
    )
end