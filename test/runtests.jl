using SimpleKernelRegression
using Test
using Aqua
using TestItemRunner


@run_package_tests


@testset "Aqua.jl" begin
    Aqua.test_all(
      SimpleKernelRegression;
      stale_deps=(ignore=[:MKL,:AppleAccelerate],)
    )
  end