# 构建JIT：极度懒惰-使用LazyReexport从AST重新导出到JIT
=

**本教程正在积极开发中。它是不完整的，细节可能会经常变化。** Nonetheless we invite you to try it out as it stands, and we welcome any feedback.

## 第四章绪论
* * *

欢迎学习`在LLVM中构建基于ORC的JIT`教程的第4章。本章介绍自定义MaterializationUnits和Layers，以及Lazy Reexports API。这些将一起用于将[第3章](zh-BuildingAJIT3.html)中的CompileOnDemandLayer替换为直接来自万花筒AST的自定义延迟JITing方案。

**待完成的工作：**

**(1)从IR描述JITING的缺点(必须先编译到IR，减少了懒惰带来的好处)。**

**(2)详细描述编译器回调管理器和间接存根管理器。**

**(3)贯穿addFunctionAST的实现。**

## 完整代码列表
* * *

下面是我们的运行示例的完整代码清单，JIT是从万花筒AST懒惰地编写的。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

```
../../examples/Kaleidoscope/BuildingAJIT/Chapter4/KaleidoscopeJIT.h
```

[下一步：Remote-JITing\--进程隔离和远程懒惰](zh-BuildingAJIT5.html)
