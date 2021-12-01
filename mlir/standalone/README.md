# An out-of-tree MLIR dialect

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a standalone `opt`-like tool to operate on that dialect.

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

说明： 
* `$BUILD_DIR`是llvm项目的编译路径，比如：${YOUR_GIT_CLONE_PATH}/llvm-project/build/
* `$PREFIX`是llvm安装路径，如果在编译安装llvm时通过cmake的`cmake_install_prefix`指定安装路径的话，就是该路径；否则为默认值，一般是`/usr/local/`.