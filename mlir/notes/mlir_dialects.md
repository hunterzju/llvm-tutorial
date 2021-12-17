# MLIR中Dialects分类及关联
## 背景
（可以跳过的废话）
Dialect可以算是MLIR设计的灵魂所在，但是在学习MLIR过程中，众多Dialect也会带来很多困惑：某个Dialect具体作用和含义是什么？为什么要lowering到某个Dialect？虽然[官方文档](https://mlir.llvm.org/docs/Dialects/)中有Dialect相关的文档，但是一方面文档给出的信息有限，有的文档并没有对Dialect做整体介绍（比如`SCF Dialect`，甚至都没介绍其全称）；另一方面缺少对各个Dialect之间关系的介绍。这给深入理解`Dialect`带来一些困难。在翻阅MLIR讨论区的时候意外发现了一篇对Dialect的介绍：[codegen-dialect-overview](https://llvm.discourse.group/t/codegen-dialect-overview/2723)，觉得受益匪浅，整理分享给大家。

*注意：下文中涉及Dialect属于MLIR早期版本，和当前代码仓库中Dialect存在一定差异（比如Standard Dialect被拆分了），但是对于从整体上理解Dialect具有指导作用。*

## Dialect分类
MLIR中Dialect分类可以通过两个坐标轴来看：tensor/buffer和payload/structre。

![引用自mlir讨论区](https://aws1.discourse-cdn.com/free1/uploads/llvm/optimized/1X/4e51f8ba1dbf21f4dd171342b420178b00d4dbe8_2_613x500.png)

tensor/buffer维度含义是：Dialect主要数据类型是按照机器学习框架中的Tensor表示的（tensor），还是底层编译器中的Memory Buffer表示的（buffer）。很多方言的操作既有基于Tensor的也有基于Buffer的，比如`Linalg`和`Standard`。结合具体用例会更好理解一些（参考Toy中ch5转换到Linalg部分）。

payload/structure维度含义是：payload表示Dialect中操作描述执行什么计算（What）；structure表示Dialect中操作描如何执行计算（How）。比如`Math Dialect`描述了执行什么计算，属于payload类型，`SCF Dialect`描述了如何执行计算，属于structure类型。

## Dialect抽象层级
![引用自mlir讨论区](https://aws1.discourse-cdn.com/free1/uploads/llvm/optimized/1X/0931943b5f58428e5a47b2bfebf1aaf855ddfe9f_2_613x500.png)

`Linalg Dialect`: 对结构化数据进行结构化处理的通用表示。既可以将`tensor`作为操作数，也可以将`buffer`作为操作数；`Operation`中既有表示执行具体运算的`payload`类型操作，也有表示如何进行运算的`struct`类型操作。实际应用中外部`Dialect`很多情况下会先转换到`Linalg Dialect`再执行后续优化。

`Vector Dialect`：对`SIMD`或者`SIMT`模型的抽象。其作为一种向量的中间表示，可以被转换到不同Target对应的底层表示，从而实现Tensor在不同平台的支持。

`Affine Dialect`：对面体编译（polyhedral compilation）的实现。其主要包含了多维数据结构的控制流操作，比如：多维数据的循环和条件控制，存储映射操作等。其目标是实现多面体变换，比如：自动并行化、用于局部改进的循环融合和平铺，以及 MLIR 中的循环矢量化。

`SCF(Structured Control Flow) Dialect`：比控制流图CFG更高层的抽象，比如并行的`for`和`while`循环以及条件判断。通常`Affine`和`Linalg`会降低到`SCF`，`SCF`也可以降低为`Standard`（貌似被拆分）中的CFG。

`Async Dialect`：通常用来表示异步操作模型，一般为一些操作的集合，在不同的抽象层次含义有所变化。

最终，底层抽象Dialect被转换为特定平台的Dialect执行，比如：`LLVM`, `NVVM`, `AVX`, `Neon`, `SVE`等。

## Dialect转换通路
![引用自mlir讨论区](https://aws1.discourse-cdn.com/free1/uploads/llvm/optimized/1X/bf319d01863dd4174ff41867f231d8aa9d7b10e9_2_613x500.png)
这里参考tensorflow中的Dialect转换来说明MLIR中Dialect的转换：

在Tensorflow层，先从TF Dialet转换到HLO Dialect, 在HLO(High-Level Optimizer) Dialect中完成一些高层优化，之后转换到MHLO(Meta HLO)。

在MLIR层，基本标量运算和Tensor运算被分解到不同的转换流程。标量运算被转换为Standard中的基本数学运算算子，进而下降到LLVM Dialect；标量运算中的控制流图也被转换到对应的Standard CFG中，进而下降到LLVM的CFG。Tensor运算部分被转换到Linalg；然后基本运算转换到Vector，控制流降低到Affine后转换为SCF，SCF根据运行模型转换到相应的Dialect。

## 总结
感觉学习MLIR如果对Dialect缺乏一个抽象层次的认知理解起来会很困难。上述内容仅仅是对Dialect的粗浅认知，并且MLIR中的Dialect还处在不断变化中，对具体某一个Dialect的理解可能并不是关键，更多还是需要理解Dialect所对应的抽象层次，并结合项目需求理解。

以上内容基于[codegen-dialect-overview](https://llvm.discourse.group/t/codegen-dialect-overview/2723)对MLIR中的Dialect简单理解，如有不准确的地方，欢迎指正。