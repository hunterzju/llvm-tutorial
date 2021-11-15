# 第2章：发射基本MLIR

[TOC]

现在我们已经熟悉了我们的语言和AST，让我们看看MLIR如何帮助编译Toy。

## 简介：多级中间表示法

其他编译器，如llvm(参见[Kaleidcope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html))，])提供了一组固定的预定义类型和(通常是*低级*/risc-like)指令。在发出LLVM IR之前，由给定语言的前端执行任何特定于语言的类型检查、分析或转换。例如，Clang将使用其AST不仅执行静电分析，还执行转换，例如通过AST克隆和重写进行C++模板实例化。最后，具有比C/C++更高级别结构的语言可能需要从其AST降低很多，才能生成LLVM IR。

因此，多个前端最终重新实现重要的基础设施，以支持这些分析和转换的需求。MLIR通过为可扩展性而设计，从而解决了这个问题。因此，预定义的指令(MLIR术语中的*运营*)或类型很少。

## 与MLIR接口

[语言参考](../../LangRef.md)

MLIR被设计为一个完全可扩展的基础设施；没有封闭的属性集(想一想：常量元数据)、操作或类型。MLIR通过[方言](../../LangRef.md#方言)的概念支持这种可扩展性。方言提供了一种分组机制，用于在唯一的“命名空间”下进行抽象。

在MLIR中，[`Operations`](../../LangRef.md#Operations)是抽象和计算的核心单元，在很多方面类似于LLVM指令。操作可以具有特定于应用程序的语义，并且可以用来表示LLVM中的所有核心IR结构：指令、全局变量(如函数)、模块等。

以下是玩具`transspose`操作的MLIR程序集：

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

让我们来剖析一下这个MLIR操作：

- “%”
  * 为此操作定义的结果指定的名称(包括[前缀符号以避免collisions](../../LangRef.md#identifiers-and-keywords)).一个操作可以定义零个或多个结果(在Toy的上下文中，我们将自己限制为单结果操作)，它们是SSA值。该名称在解析期间使用，但不是持久的(例如，在SSA值的内存表示中不跟踪该名称)。
- “玩具转置”
  * 操作的名称。它应该是唯一的字符串，在“`.`”之前加上方言的命名空间。这可以理解为`toy`方言中的`transspose`操作。
- “%”
  * 零个或多个输入操作数(或参数)的列表，它们是由其他操作定义或引用挡路参数的SSA值。
- `{inplace=true}`
  * 一种包含零个或多个属性的字典，这些属性是始终恒定的特殊操作数。在这里，我们定义了一个名为‘inplace’的布尔属性，该属性的常量值为true。
- ‘(张量<2x3xf64>)->张量<3x2xf64>’
  * 这指的是函数形式中的操作类型，即拼写括号中的参数类型和随后返回值的类型。
- `loc(“示例/文件/路径”：12：1)`
  * 这是源代码中发起此操作的位置。

这里显示的是操作的一般形式。如上所述，MLIR中的操作集是可扩展的。使用一小组概念对操作进行建模，从而能够对操作进行一般的推理和操作。这些概念是：

- 操作的名称。
- SSA操作数值的列表。
-(.)属性。
- 结果值的[类型](../../LangRef.md#type-system)列表。
- a[用于调试目的的源location](../../Diagnostics.md#source-locations)。
- 后继[块](../../LangRef.md#块)列表(主要用于分支)。
- [Regions](../../LangRef.md#Regions)列表(用于函数等结构化操作)。

在MLIR中，每个操作都有一个与之关联的必需源位置。与LLVM相反，在LLVM中，调试信息位置是元数据，可以删除，而在MLIR中，位置是核心需求，API依赖并操作它。因此，丢弃位置是一种明确的选择，不能因为错误而发生。

举例说明：如果转换将一个操作替换为另一个操作，则该新操作必须仍附加有位置。这使得追踪操作的来源成为可能。

值得注意的是，mlir-opt工具-用于测试编译器通道的工具-默认情况下不包括输出中的位置。`-mlir-print-debuginfo`标志指定包含位置。(更多选项请运行`mlir-opt--help`。)

### 不透明API不透明API

MLIR旨在允许自定义大多数IR元素，如属性、操作和类型。同时，IR元素始终可以归结为上述基本概念。这允许MLIR解析、表示和[round-trip](../getting_started/Glossary.md#round-trip)IR For*任何*操作。例如，我们可以将上面的Toy操作放到一个`.mlir`文件中，并在*Myloropt*中往返，而无需注册任何方言：

```mlir
func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

对于未注册的属性、操作和类型，MLIR将强制执行一些结构约束(ssa、挡路终止等)，但在其他情况下，它们是完全不透明的。例如，MLIR几乎没有关于未注册操作是否可以操作特定数据类型、可以接受多少操作数或产生多少结果的信息。这种灵活性对于引导目的很有用，但在成熟的系统中通常不建议这样做。未注册的操作必须通过转换和分析保守地对待，而且它们更难构造和操作。

通过为Toy制作应该无效的IR并在不触发验证器的情况下在往返过程中查看它，可以观察到此处理：

```mlir
func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

这里有多个问题：`toy.print`操作不是终止符；它应该接受一个操作数；并且它不应该返回任何值。在下一节中，我们将使用MLIR注册我们的方言和操作，插入验证器，并添加更好的API来操作我们的操作。

## 定义玩具方言

为了有效地与MLIR交互，我们将定义一个新的玩具方言。这种方言将模拟玩具语言的结构，并为高级分析和转换提供一条简单的途径。

```c++
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override virtual methods to change some general
/// behavior, which will be demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
 public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities.
  static llvm::StringRef getDialectNamespace() { return "toy"; }
};
```

现在可以在全局注册表中注册该方言：

```c++
  mlir::registerDialect<ToyDialect>();
```

从现在开始创建的任何新的`MLIRContext`都将包含玩具方言的一个实例，并调用特定的钩子来解析属性和类型。

## 定义玩具操作

既然我们有了“东方话”，我们就可以开始注册业务了。这将允许提供系统的睡觉可以挂钩的语义信息。下面我们来介绍一下`toy.constant`操作的创建过程：

```mlir
 %4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

该操作的操作数为零，[密集elements](../../LangRef.md#dense-elements-attribute)属性名为`value`，返回[张量类型](../../LangRef.md#tensor-type)的单个结果。操作继承自[CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)`mlir：：op`类，该类还需要一些可选的[*特性*](../../Traits.md)来自定义其行为。这些特征可以提供额外的访问器、验证等。

```c++
class ConstantOp : public mlir::Op<ConstantOp,
                     /// The ConstantOp takes no inputs.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// The result of getType is `Type`.
                     mlir::OpTraits::OneTypedResult<Type>::Impl> {

 public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations can provide additional verification beyond the traits they
  /// define. Here we will ensure that the specific invariants of the constant
  /// operation are upheld, for example the result type must be of TensorType.
  LogicalResult verify();

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the builder to allow for easily generating
  /// instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

并且我们在`Toyota Dialect`构造函数中注册此操作：

```c++
ToyDialect::ToyDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<ConstantOp>();
}
```

### 操作VS操作：使用MLIR操作

既然我们已经定义了一个操作，我们将需要访问和转换它。在MLIR中，与操作相关的类主要有两个：`Operation`和`Op`。`Operation`类用于对所有操作进行通用建模。它是“不透明的”，因为它没有描述特定操作或操作类型的属性。相反，“Operation”类为操作实例提供了一个通用API。另一方面，每种特定类型的操作都由一个`Op`派生类表示。例如，`ConstantOp`表示零输入、一输出的操作，始终设置为相同的值。`Op`派生类充当`operation*`的智能指针包装器，提供特定于操作的访问器方法，以及操作的类型安全属性。这意味着当我们定义玩具操作时，我们只是定义了一个干净的、语义上有用的接口，用于构建`Operation`类并与其交互。这就是为什么我们的`ConstantOp`没有定义类字段；所有的数据结构都存储在引用的`Operation`中。一个副作用是，我们总是通过值传递`Op`派生类，而不是通过引用或指针(*按值传递*是一种常见的习惯用法，类似于属性、类型等)。给定一个通用的`operation*`实例，我们始终可以使用LLVM的强制转换基础设施获取具体的`Op`实例：

```c++
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);

  // This operation is not an instance of `ConstantOp`.
  if (!op)
    return;

  // Get the internal operation instance wrapped by the smart pointer.
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation &&
         "these operation instances are the same");
}
```

### 使用操作定义规范(ODS)框架

MLIR除了专门化`mlir：：op`C++模板外，还支持声明式定义操作。这是通过[操作定义规范](../../OpDefinitions.md)框架实现的。关于操作的事实被简明地指定到TableGen记录中，该记录将在编译时展开为等效的`mlir：：Op`C++模板专门化。考虑到面对C++API更改时的简洁性、简明性和一般稳定性，使用ODS框架是在MLIR中定义操作的理想方式。

让我们看看如何定义ConstantOp的ODS等效项：

要做的第一件事是定义一个指向我们用C++定义的玩具方言的链接。它用于将我们将定义的所有操作链接到我们的方言：

```tablegen
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "toy";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "toy";
}
```

现在我们已经定义了到玩具方言的链接，我们可以开始定义操作了。ODS中的操作是通过继承`Op`类来定义的。为了简化我们的操作定义，我们将用玩具方言为操作定义一个基类。

```tablegen
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

定义了所有的初始部分后，我们可以开始定义常量操作。

我们通过继承上面的“Toy_Op”基类来定义玩具操作。在这里，我们提供了助记符和操作的特征列表。这里的[mnemonic](../../OpDefinitions.md#operation-name)与`ConstantOp：：getOperationName`中给出的没有方言前缀`toy.`匹配。我们的C++定义中缺少`ZeroOperands`和`OneResult`特性；这些特性将根据我们稍后定义的`arguments`和`Results`字段自动推断出来。

```tablegen
def ConstantOp : Toy_Op<"constant"> {
}
```

此时，您可能想知道TableGen生成的C++代码是什么样子。只需使用`Gen-op-decls`或`Gen-op-defs`操作运行`mlir-tblgen`命令，如下所示：

```shell
${build_root}/bin/mlir-tblgen -gen-op-defs ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

根据选择的操作，这将打印`ConstantOp`类声明或其实现。在开始使用TableGen时，将此输出与手工创建的实现进行比较非常有用。

#### 定义参数和结果

定义了操作的外壳后，我们现在可以为我们的操作提供[inputs](../../OpDefinitions.md#operation-arguments)和[outputs](../../OpDefinitions.md#operation-results)。操作的输入或参数可以是SSA操作数值的属性或类型。结果对应于操作生成的值的一组类型：

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

通过给参数或结果命名，如`$value`，ODS会自动生成匹配的访问器：`DenseElementsAttr ConstantOp：：Value()`。

#### 添加文档

定义操作后的下一步是对其进行文档记录。操作可以提供[``description`](../../OpDefinitions.md#operation-documentation)]和摘要字段来描述操作的语义。此信息对该方言的用户很有用，甚至可以用来自动生成Markdown文档。

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

#### 验证操作语义

至此，我们已经介绍了原始C++操作定义的大部分。下一个要定义的部分是验证器。幸运的是，与命名访问器非常相似，ODS框架将根据我们给出的约束自动生成大量必要的验证逻辑。这意味着我们不需要验证返回类型的结构，甚至不需要验证输入属性`value`。在许多情况下，消耗臭氧层物质业务甚至不需要额外核查。要添加其他验证逻辑，操作可以覆盖[`verifier`](../../OpDefinitions.md#custom-verifier-code)字段。`verifier`字段允许定义一个C++代码blob，它将作为`ConstantOp：：verify`的一部分运行。此BLOB可以假设该操作的所有其他不变量都已经过验证：

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Here we invoke
  // a static `verify` method in a C++ source file. This codeblock is executed
  // inside of ConstantOp::verify, so we can use `this` to refer to the current
  // operation instance.
  let verifier = [{ return ::verify(*this); }];
}
```

#### 附加`build`方法

我们的原始C++示例中缺少的最后一个组件是`build`方法。ODS可以自动生成一些简单的构建方法，在这种情况下，它将为我们生成我们的第一个构建方法。对于睡觉，我们定义了[`builders`](../../OpDefinitions.md#custom-builder-methods)字段。此字段包含一个`OpBuilder`对象列表，这些对象接受与C++参数列表相对应的字符串，以及一个可选代码挡路，该代码可用于指定内联实现。

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  ...

  // Add custom build methods for the constant operation. These methods populate
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilderDAG<(ins "DenseElementsAttr":$value), [{
      // Call into an autogenerated `build` method.
      build(builder, result, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilderDAG<(ins "double":$value)>
  ];
}
```

#### 指定自定义装配格式

在这一点上，我们可以生成我们的“玩具IR”。例如，以下内容：

```toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

结果为以下IR：

```mlir
module {
  func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  func @main() {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    "toy.return"() : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

这里需要注意的一件事是，我们所有的Toy操作都是使用通用装配格式打印的。此格式是本章开头分解`toy.transspose`时显示的格式。MLIR允许操作定义它们自己的自定义程序集格式，可以是[declaratively](../../OpDefinitions.md#declarative-assembly-format)，也可以是通过C++。定义自定义程序集格式允许将生成的IR裁剪成更具可读性的内容，方法是去掉通用格式所需的大量乱七八糟的东西。让我们来演练一个我们想要简化的操作格式的示例。

##### `toy.print`
当前的`toy.print`形式有点冗长。我们想要去掉很多额外的角色。让我们首先考虑一下`toy.print`的格式会有多好，然后看看如何实现它。看一下`toy.print`的基础知识，我们会得到：

```mlir
toy.print %5 : tensor<*xf64> loc(...)
```

在这里，我们剥离了大部分格式，使其只剩下最基本的部分，可读性也大大提高。为了提供自定义的汇编格式，操作可以覆盖C++格式的`parser`和`printer`字段，也可以覆盖声明性格式的`AssemyFormat`字段。让我们首先看一下C++变体，因为这是声明性格式在内部映射到的。

```tablegen
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // Divert the printer and parser to static functions in our .cpp
  // file that correspond to 'print' and 'printPrintOp'. 'printer' and 'parser'
  // here correspond to an instance of a 'OpAsmParser' and 'OpAsmPrinter'. More
  // details on these classes is shown below.
  let printer = [{ return ::print(printer, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}
```

打印机和解析器的C++实现如下所示：

```c++
/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
static void print(mlir::OpAsmPrinter &printer, PrintOp op) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
static mlir::ParseResult parsePrintOp(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::OperandType inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();

  // Resolve the input operand to the type we parsed in.
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();

  return mlir::success();
}
```

定义了C++实现之后，让我们看看如何将其映射到[声明性format](../../OpDefinitions.md#declarative-assembly-format).声明性格式主要由三个不同的组件组成：

* 指令
  - 一种内置函数，具有一组可选的参数。
* 文字
  - 用\`\`括起来的关键字或标点符号。
* 变量
  - 已在操作本身上注册的实体，即
`PrintOp`中的参数(属性或操作数)、结果、后继等
在上面的示例中，变量应该是`$input`。

我们的C++格式的直接映射类似于：

```tablegen
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // In the following format we have two directives, `attr-dict` and `type`.
  // These correspond to the attribute dictionary and the type of a given
  // variable represectively.
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

[声明性format](../../OpDefinitions.md#declarative-assembly-format)有更多有趣的特性，因此在用C++实现自定义格式之前一定要检查它。在美化了几个操作的格式之后，我们现在得到一个可读性更强的：

```mlir
module {
  func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

上面我们介绍了几个在ODS框架中定义操作的概念，但是还有更多我们还没有机会介绍的概念：区域、各种操作数等。有关更多详细信息，请查看[完整规范](../../OpDefinitions.md)。

## 完整的玩具示例

我们现在可以生成我们的“玩具IR”了。您可以构建`toyc-ch2`，然后尝试上面的示例：`toyc-CH2 test/Examples/Toy/CH2/codegen.toy-emit=mlir-mlir-print-debuginfo`。我们还可以检查往返过程：`toyc-CH2 test/examples/Toy/CH2/codegen.toy-emit=mlir-mlir-print-debuginfo 2>codegen.mlir`后跟`toyc-CH2 codegen.mlir-emit=mlir`。您还应该对最终的定义文件使用`mlir-tblgen`，并研究生成的C++代码。

在这一点上，MLIR知道我们的玩具方言和操作。在[下一章](CH-3.md)中，我们将利用我们的新方言实现一些针对玩具语言的高级语言特定分析和转换。
