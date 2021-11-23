cmake_minimum_required(VERSION 3.13.4)
project(standalone-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

# MLIR_DIR 由编译参数指定 -DMLIR_DIR=$PREFIX/lib/cmake/mlir
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
# LLVM_DIR 由`find_package`创建，指向保存`LLVMConfig.cmake`文件所在路径
# 也可由-DLLVM_DIR指定
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# CMAKE_BINARY_DIR: 工程编译发生的目录，本工程中为$(PROJECT_DIR)/build
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# MLIR_CMAKE_DIR/LLVM_CMAKE_DIR: LLVM安装目录，由`LLVMConfig.cmake`创建
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
message(STATUS "MLIR/LLVM CMAKEDIR: ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# MLIR_CMAKE_DIR/LLVM_CMAKE_DIR: LLVM安装目录/include
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
message(STATUS, "LLVM/MLIR_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
message(STATUS, "PROJ_SRC/BIN_DIR: ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR}")
# LLVM_BUILD_LIBRARY_DIR指向${LLVM_LIBRARY_DIR}
# LLVM cmake file 提供llvm_map_components_to_libnames来链接library
# 详见llvm-project/llvm/docs/CMake.rst#Embedding LLVM in your project
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
message(STATUS, "LLVM_BUILD_LIB/DEF: ${LLVM_BUILD_LIBRARY_DIR} ${LLVM_DEFINITIONS}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(standalone-opt)
add_subdirectory(standalone-translate)
