# Toy教程

本教程基于MLIR构建了一中基础的Toy语言实现。本教程的目标是介绍MLIR的概念；特别是[方言(dialects)](../../LangRef.md#dialects)如何帮助轻松支持特定于语言的构造和转换，同时仍然提供一条降低到LLVM或其他代码生成(codegen)基础设施的简单途径。本教程基于[LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)的模型

本教程假定您已经克隆并构建了MLIR；如果您还没有这样做，请参阅[MLIR入门](https://mlir.llvm.org/getting_started/).

本教程分为以下几章：

- [第1章](zh-Ch-1.md)：Toy语言简介及其AST的定义。
- [第2章](zh-Ch-2.md)：遍历AST以发出MLIR中的方言，介绍基本的MLIR概念。这里我们展示了如何开始将语义附加到MLIR中的自定义操作。
- [第3章](zh-Ch-3.md)：使用模式重写系统的高级语言特定优化。
- [第4章](zh-Ch-4.md)：使用接口编写与通用方言无关的转换。在这里，我们将展示如何将特定的方言信息插入到通用转换中，如维度推断和内联。
- [第5章](zh-Ch-5.md)：部分降低到较低级别的方言。为了优化，我们将把一些高级语言特定语义转换为面向仿射的通用方言。
- [第6章](zh-Ch-6.md)：降低到LLVM和代码生成。在这里，我们将把LLVM IR作为代码生成的目标，并详细介绍降低框架的更多内容。
- [第7章](zh-Ch-7.md)：扩展Toy：添加对复合类型的支持。我们将演示如何将自定义类型添加到MLIR，以及它如何适应现有流程。

[第一章](zh-Ch-1.md)将介绍Toy语言和AST。
