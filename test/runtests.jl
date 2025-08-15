using SimpleKernelRegression
using Test

@testset "SimpleKernelRegression.jl" begin
    @test SimpleKernelRegression.hello_world() == "Hello, World!"
end
