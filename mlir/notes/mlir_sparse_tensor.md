## Evaluate sparse runtime support library performance
代码实现在：SparseTensorUtils.cpp
mlir_c_runner_utils
    SparseTensorUtils

cc_library(
    name = "mlir_c_runner_utils",
    srcs = [
        "lib/ExecutionEngine/CRunnerUtils.cpp",
        "lib/ExecutionEngine/SparseTensorUtils.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/CRunnerUtils.h",
        "include/mlir/ExecutionEngine/SparseTensorUtils.h",
    ],
    includes = ["include"],
)