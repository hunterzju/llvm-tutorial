#! https://zhuanlan.zhihu.com/p/441471026
# MLIR-Toy-实践-2-采用Interface做转换
之前采用ODS框架为Toy语言添加了一个[逻辑或的操作OrOp](https://zhuanlan.zhihu.com/p/441237921)，使得`|`操作符能够被解析为MLIR中的一个Op节点。本文主要结合MLIR中的Toy教程ch4为添加的Op增加采用Interface的通用接口操作，通过该Interface可以实现函数调用的内联(inline)和输入变量的形状推断（shapeInference)。这部分主要用到了MLIR的`Interface`和`Pass`。

![Image](https://pic4.zhimg.com/80/v2-7257dee4e740762f2458fb9bb46b53b6.png)

## 函数调用内联(inline)
Inline处理作用是将函数调用嵌入到主函数进程中，有点类似函数链接的感觉。在Toy语言的MLIR中，`toy.generic_call`操作表示函数调用，我们需要在遇到该操作时，将被调用函数语句块关联到调用者的语句块中。MLIR中提供了标准化的[接口（Interface）](https://mlir.llvm.org/docs/Interfaces/)可以用来对MLIR进行分析和转换。可以利用Interface来实现上述操作和转换。

首先需要通过MLIR中的`Dialect Interface`提供一个Toy Dialect的内联接口定义，该内联接口继承自MLIR中内置的`DialectInlinerInterface`接口，定义了执行内联的一些操作接口函数，实现在`Dialect.cpp`文件中：
```c++
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  //...
};
```
此外，内联操作仅针对被调用的私有函数，因此在从AST生成MLIR的时候还需要修改函数的可见性为`private`，实现过程在`mlir/mycode/Ch6/mlir/MLIRGen.cpp`中：
```c++
mlir::FuncOp mlirGen(FunctionAST &funcAST) {
    // ...
    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();
    // ...
}
```
然后在ToyDialect的初始化过程中注册DialectInterface，添加该接口。此时，还需要告诉接口对`toy.generic_call`操作执行内联过程。MLIR中提供了`OperationInterface`可以对Operation定义针对操作的更加细粒度的接口，其中MLIR内置的`CallOpInterface`接口表示对应的操作是一个函数调用。这里需要给`GenericCallOp`添加`CallOpInterface`特性：
```mlir
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
```
另外`CallOpInterface`还需要知道被调函数和函数参数，需要在`Dialect.cpp`中实现：
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
最终Inline过程作为一个pass被添加到处理过程中。

另外，在内联过程中，由于函数调用输入参数是一个实参，函数定义中参数为形参，二者的tensor维度需要进行类型转换，才能符合InlineInterface的执行条件。因此，另外定义了一个`CastOp`用来执行显示类型转换。该Op定义方式与之前的Op相同，这里不再赘述。需要注意的是`CastOp`也采用OpInterface定义的操作接口，实现了`areCastCompatible`函数，用于判断类型转换是否合法。


## 变量维度推断(shapeInference)
执行完内联后，还有一个问题：需要根据输入参数的维度推断出输出参数的维度，产生符合预期的输出。比如，我们内联结束后得到的MLIR为：
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
两个2x3的矩阵转置后对应元素相乘，输出应该是一个3x2的矩阵，但是此时的输出维度是未指定的。这里也可以通过Tablegen的方式来定义一个OperationInterfapce，其中包含一个接口方法`inferShapes`：
```mlir
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}
```
接下来需要在`Ops.td`定义中给每个Op添加`ShapeInferenceOpInterface`特性，并在`Dialect.cpp`中为对应Op实现一个`inferShapes()`方法，指定输出的维度。
```c++
// ops.td
def OrOp : Toy_Op<"or",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    // ...
}
// dialect.cpp
void OrOp::inferShapes() { getResult().setType(getOperand(0).getType()); }
```

最后需要实现一个pass来执行维度推断，shapeInference是针对function执行的，因此需要继承一个`function pass`(MLIR中也提供了`operation pass`来对operation进行处理)。该pass的执行逻辑通过重写`runOnFunction()`方法实现。

## 验证：
此过程中主要需要对添加的操作增加Interface特性，并提供一个用于shapeinference的接口函数，最终验证可以通过：
```
./bin/toyc-ch6 ../../testcode/Ch2/codegen.toy --emit mlir -opt
```
此过程主要实现了函数Inline和shapeInference，项目工程在[github](https://github.com/hunterzju/llvm-tutorial)上。