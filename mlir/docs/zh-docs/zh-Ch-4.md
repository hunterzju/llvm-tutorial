# 第4章：使用接口启用泛型转换

[TOC]

## 背景：努力解决可扩展的IR问题

通过方言，MLIR允许表示许多不同的抽象级别；我们之前定义的toy方言就是这样一个例子。尽管这些不同的方言可能代表不同的抽象，但我们通常想要执行一组共同的转换和分析。出现的问题是，为每种方言幼稚地实现每个转换都会导致大量代码重复，因为内部算法通常非常相似(如果不是相同的话)。我们希望为转换提供不透明地挂钩到像toy这样的方言的能力，以获得他们需要的信息。

MLIR为某些核心转换提供了一组始终可用的钩子，如[上一章](zh-Ch-3.md)所示，我们通过操作上的钩子(`getCanonicalizationPatterns`)注册了一些规范。然而，这些类型的钩子并不能很好地扩展。因此，设计了一个更通用的解决方案，其形式为[interface](../../Interfaces.md)，以使MLIR基础设施与表示一样具有可扩展性。接口为方言和操作提供通用机制，以便为转换或分析提供信息。

## 形状推断：为代码生成做准备

我们的toyIR当前在泛型张量上操作，这意味着除了在常量初始化期间之外，我们不知道张量的形状。这使得优化和代码生成变得复杂。幸运的是，我们可以简单地在计算过程中传播形状，直到它们都知道为止。问题是如何处理对用户定义的泛型函数的调用：每个调用点可以推导出不同的形状。一种可能性是基于参数类型执行符号推理，但是如果我们要在语言中引入更多的控制流，这将很难推广。另一种方法是函数专门化，每个具有新参数形状的调用点都复制被调用的函数并专门化它。我们对Toy采取的方法是内联所有函数调用，然后执行过程内形状传播。

### 内联

在这里，我们可以编写一个专门为toy方言设计的内联算法，但这可能会变得相当复杂，这取决于我们想要的复杂程度。抛开建模成本不谈，从零开始实现纯粹的结构转换已经很复杂了。值得庆幸的是，MLIR提供了方言可以插入的通用内联算法。在Toy中，我们所需要做的就是提供[interface](../../Interfaces.md)供内联程序挂接到其中。

我们需要做的第一件事是定义对toy方言内联操作的约束。此信息通过[方言interface](../../Interfaces.md#dialect-interfaces)提供。这实质上是一个包含一组虚拟钩子的类，方言可以覆盖这些虚拟钩子。在本例中，接口为`DialectInlinerInterface`。

```c++
/// This class defines the interface for handling inlining with Toy operations.
/// We simplify inherit from the base interface class and override
/// the necessary methods.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// This hook checks to see if the given callable operation is legal to inline
  /// into the given call. For Toy this hook can simply return true, as the Toy
  /// Call operation is always inlinable.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// This hook checks to see if the given operation is legal to inline into the
  /// given region. For Toy this hook can simply return true, as all Toy
  /// operations are inlinable.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  /// This hook is called when a terminator operation has been inlined. The only
  /// terminator that we have in the Toy dialect is the return
  /// operation(toy.return). We handle the return by replacing the values
  /// previously returned by the call operation with the operands of the
  /// return.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
```

然后，我们直接在toy方言上注册我们的方言接口，类似于我们对operations所做的操作。

```c++
ToyDialect::ToyDialect(mlir::MLIRContext *ctx) : mlir::Dialect("toy", ctx) {
  addInterfaces<ToyInlinerInterface>();
}
```

接下来，我们需要提供一种方法，让内联程序知道`toy.Generic_call`表示对函数的调用。MLIR提供了可用于将操作标记为“call-like”的[operation interface](../../Interfaces.md#operation-interfaces)。与方言接口不同，操作接口提供了更精细的信息粒度，这些信息是单个操作的特定和核心信息。这里我们要添加的接口是`CallOpInterface`。

要添加此接口，我们只需将定义包含到我们的操作规范文件(`Ops.td`)中：

```tablegen
include "mlir/Interfaces/CallInterfaces.td"
```

并添加到`GenericCallOp`的特征列表中：

```tablegen
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
}
```

在上面，我们还使用`DeclareOpInterfaceMethods`指令自动声明GenericCallOp的类声明中的所有接口方法。这意味着我们只需要提供一个定义：

```c++
/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return inputs(); }
```

既然已经通知了内嵌器有关toy方言的信息，我们可以将内联过程添加到Toy的过程管理器中：

```c++
  pm.addPass(mlir::createInlinerPass());
```

现在，让我们看一个工作示例：

```mlir
func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %5 : tensor<*xf64>
  toy.return
}
```

我们有两个对Multiple_Transpose的调用，我们希望将它们内联到main中，但是如果我们查看输出，什么都没有改变。我们遗漏了最后一个细微的部分：在调用的边缘有一个隐藏的类型转换。如果我们查看上面的内容，则Generic_call的操作数类型为`tensor<2x3xf64>`，而函数的输入应为`tensor<*xf64>`。要解决此差异，内联程序需要插入显式强制转换操作。为此，我们需要向toy方言添加一个新操作`ToyCastOp`(toy.cast)，以表示两个不同形状之间的类型转换。

```tablegen
def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    NoSideEffect,
    SameOperandsAndResultShape]
  > {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked,
    then shape is required to match. The operation is invalid if converting
    to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
}
```

请注意，此强制转换操作的定义在特征列表中添加了一个`CastOpInterface`。此接口为类似强制转换的操作提供了几个实用程序，例如折叠一致性强制转换和验证。我们通过为`areCastCompatible`方法提供定义来挂钩到此接口：

```c++
/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = inputs.front().dyn_cast<TensorType>();
  TensorType output = outputs.front().dyn_cast<TensorType>();
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}

```

通过正确的强制转换操作，我们现在可以覆盖ToyInlinerInterface上的必要挂钩，以便在需要时为我们插入它：

```c++
struct ToyInlinerInterface : public DialectInlinerInterface {
  ...

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```

如果我们再次按照流程运行工作示例，我们会得到预期的结果：

```mlir
func @main() {
  %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %1 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %2 = "toy.cast"(%1) : (tensor<2x3xf64>) -> tensor<*xf64>
  %3 = "toy.cast"(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
  %4 = "toy.transpose"(%2) : (tensor<*xf64>) -> tensor<*xf64>
  %5 = "toy.transpose"(%3) : (tensor<*xf64>) -> tensor<*xf64>
  %6 = "toy.mul"(%4, %5) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  toy.print %6 : tensor<*xf64>
  toy.return
}
```

注意：泛型内联还将执行简化，因此输出可能比预期的要干净一些。

### 程序内形状推断

现在我们已经内联了所有函数，剩下的主函数包含静态和动态成形操作的混合。现在，我们可以编写一个简单的形状推断过程来在程序内(在单个函数内)传播形状。我们可以将其编写为直接编码toy方言中的操作约束的PASS，并且这似乎是一个可以通用编写的转换的很好选择。作为一个好的经验法则，最好尽可能通用地表达转换，以便将来可以扩展到其他方言。不知道还有多少其他方言可能有类似的需求或遇到同样的问题。

对于形状推断，如果我们将问题分解到其核心，我们实际上只希望操作告诉我们给定一组静态已知输入的预期输出。(我们当然可以变得更复杂，但根据我们的需要，我们可以保持简单。)鉴于此属性是特定操作的核心，我们可以定义一个操作接口，该接口可以在需要推断其结果形状的操作上指定。

与操作类似，我们也可以使用操作定义规范(ODS)框架[定义操作interfaces](../../OpDefinitions.md#operation-interfaces)]。

接口是通过继承`OpInterface`定义的，`OpInterface`将生成的C++接口类的名称作为模板参数。出于我们的目的，我们将简单地将生成的类命名为`ShapeInference`。我们还提供了接口的描述。

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];
}
```

接下来，我们定义操作需要提供的接口方法。接口方法包括：描述；字符串形式的C++返回类型；字符串形式的方法名称；以及一些可选组件，具体取决于需要。有关详细信息，请参阅[ODS documentation](../../OpDefinitions.md#operation-interfaces)]。

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  ...

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}
```

现在接口已经定义好了，我们可以将其添加到必要的Toy操作中，方法与将`CallOpInterface`添加到GenericCallOp中的方式类似：

```tablegen
def MulOp : Toy_Op<"mul",
    [..., DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
```

然后，这些操作中的每个操作都需要为`inferShapes()`方法提供定义。例如，对于mul op，结果形状被推断为输入的形状。

```c++
/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
void MulOp::inferShapes() { getResult().setType(getOperand(0).getType()); }
```

此时，每个必要的toy操作都提供了一种机制来推断它们的输出形状。ShapeInferencePass是一个FunctionPass：它将在每个函数上独立运行。MLIR还支持在任何孤立操作(即其他类似函数的操作)上运行的通用[OperationPasses](../../PassManagement.md#operation-pass)，但这里我们的模块只包含函数，因此不需要对所有操作进行泛化。

通过创建一个继承自`mlir：：FunctionPass`的类并覆盖`runOnFunction()`方法来实现这样的传递。

```c++
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, FunctionPass> {
  void runOnFunction() override {
    FuncOp function = getFunction();
    ...
  }
};
```

同时，让我们还创建一个用于实例化该过程的帮助器方法：

```c++
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
```

形状推断算法的操作如下：

1. 构建一个包含返回动态形状张量的所有操作的工作列表：这些操作需要形状推断。
2. 在工作列表上迭代：
- 查找要处理的操作：工作列表中的下一个就绪操作具有其所有非泛型参数，
- 如果找不到任何操作，则中断循环，
- 从工作列表中删除该操作，
- 从参数类型推断其输出的形状。
3. 如果工作列表为空，则算法成功。

在处理上述操作时，我们使用以下代码片段查询它是否注册了`ShapeInference`接口：

```c++
  // Ask the operation to infer its output shapes.
  LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");

  /// We check if an operation has a particular interface by casting.
  if (ShapeInference shapeOp = dyn_cast<ShapeInference>(op)) {
    shapeOp.inferShapes();
  } else {
    op->emitError("unable to infer shape of operation without shape "
                  "inference interface");
    return signalPassFailure();
  }
```

然后，我们可以将通行证添加到pass管理器：

```c++
  pm.addPass(mlir::createShapeInferencePass());
```

如果我们重新运行原始示例，现在会得到以下结果：

```mlir
func @main() {
  %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %1 = "toy.transpose"(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
  %2 = "toy.mul"(%1, %1) : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

您可以构建`toyc-ch4`并亲自试用：`toyc-ch4 test/examples/Toy/ch4/codegen.toy -emit=mlir -opt`。

在[下一章](CH-5.md)中，我们将以较低级别的方言为目标开始代码生成过程，以优化一些计算量较大的toy操作。
