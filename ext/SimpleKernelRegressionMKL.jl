module SimpleKernelRegressionMKL

# Loading MKL swaps the active BLAS backend in its own `__init__`; the
# extension only makes that pairing loadable alongside SimpleKernelRegression.
using MKL

end
