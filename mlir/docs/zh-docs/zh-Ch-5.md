# 第5章：部分降级到更低级别的方言以进行优化

现在，我们渴望生成实际的代码，并看到我们的Toy语言诞生。我们将使用LLVM生成代码，但是在这里仅仅显示LLVM构建器接口不会非常令人兴奋。取而代之的是，我们将展示如何通过在同一函数中共存的混合方言来执行渐进式降级。

为了更有趣，在本章中，我们将考虑重用现有的优化，该优化是一种用方言实现的仿射变换：`Affine`。这种方言是为程序的计算量大的部分量身定做的，而且是有限的：例如，它不支持表示我们的`toy.print`内置函数，也不应该支持！相反，我们可以将`Affine`作为Toy的计算量较大的部分，并在[下一章](zh-Ch-6.md)中直接将`LLVM IR`方言作为`print`的降低目标。作为降低的一部分，我们将从`Toy`操作的[TensorType](../../LangRef.md#tensor-type)降低到通过仿射循环嵌套索引的[MemRefType](../../LangRef.md#memref-type)。张量表示抽象值类型的数据序列，这意味着它们不存在于任何内存中。另一方面，MemRef表示较低级别的缓冲区访问，因为它们是对内存区域的具体引用。

# 方言转换

MLIR有许多不同的方言，因此在它们之间有一个统一的[converting](../getting_started/Glossary.md#conversion)框架是很重要的。这就是`DialectConversion`框架发挥作用的地方。此框架允许将一组*非法*操作转换为一组*合法*操作。要使用此框架，我们需要提供两个条件(以及可选的第三个条件)：

* 一个[转换目标](../../DialectConversion.md#CONVERSION-TARGET)
  - 这是一个正式规范，规定了哪些操作或方言对于转换是合法的。不合法的操作将需要重写模式来执行[legalization](../getting_started/Glossary.md#legalization).
* 一组[Rewrite Patterns](../../DialectConversion.md#rewrite-pattern-specification)
  - 这是一组[Patterns](../QuickstartRewrites.md)，用于将*非法*操作转换为一组零个或多个*合法*操作。
* (可选)一个[Type Converter](../../DialectConversion.md#type-conversion).
  - 如果提供，它用于转换模块参数的类型。我们的转换不需要这个。

## 转换目标

出于我们的目的，我们希望将计算密集型的`toy`运算转换为来自`Affine`中`Standard`方言的运算组合，以便进一步优化。要开始降低，我们首先定义转换目标：

```c++
void ToyToAffineLoweringPass::runOnFunction() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<mlir::AffineDialect, mlir::StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as *legal*.
  target.addIllegalDialect<ToyDialect>();
  target.addLegalOp<PrintOp>();
  ...
}
```

在上面，我们首先将Toy方言设置为非法，然后将打印操作设置为合法。我们可以反过来做这件事。单个操作始终优先于(更通用的)方言定义，因此顺序无关紧要。详见`ConversionTarget::getOpInfo`。

## 转换模式

定义转换目标后，我们可以定义如何将*非法*操作转换为*合法*操作。与[第3章](zh-Ch-3.md)中介绍的规范化框架类似，[`DialectConversion`框架](../../DialectConversion.md)也使用[RewritePatterns](../QuickstartRewrites.md)来执行转换逻辑。这些模式可以是以前看到的`RewritePatterns`，也可以是转换框架`ConversionPattern`特有的新类型的模式。`ConversionPatterns`与传统的`RewritePatterns`不同之处在于，它们接受包含已重新映射/替换的操作数的额外的`operands`参数。这是在处理类型转换时使用的，因为模式希望对新类型的值进行操作，但与旧类型的值匹配。对于我们的降级，此不变量将非常有用，因为它将当前正在操作的[TensorType](../../LangRef.md#tensor-type)转换为[MemRefType](../../LangRef.md#memref-type)。让我们来看一段降低`toy.transspose`操作的代码片段：

```c++
/// Lower the `toy.transpose` operation to an affine loop nest.
struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, ctx) {}

  /// Match and rewrite the given `toy.transpose` operation, with the given
  /// operands that have been remapped from `tensor<...>` to `memref<...>`.
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Call to a helper function that will lower the current operation to a set
    // of affine loops. We provide a functor that operates on the remapped
    // operands, as well as the loop induction variables for the inner most
    // loop body.
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::PatternRewriter &rewriter,
              ArrayRef<mlir::Value> memRefOperands,
              ArrayRef<mlir::Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the TransposeOp.
          // This allows for using the nice named accessors that are generated
          // by the ODS. This adaptor is automatically provided by the ODS
          // framework.
          TransposeOpAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.input();

          // Transpose the elements by generating a load from the reverse
          // indices.
          SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
        });
    return success();
  }
};
```

现在我们可以准备在下降过程中使用的pattern列表：

```c++
void ToyToAffineLoweringPass::runOnFunction() {
  ...

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::OwningRewritePatternList patterns;
  patterns.insert<..., TransposeOpLowering>(&getContext());

  ...
```

## 局部下降

一旦定义了模式，我们就可以执行实际的下降。`DialectConversion`框架提供了几种不同的下降模式，但考虑到我们的目的，我们将执行部分下降，因为我们此时不会转换`toy.print`。

```c++
void ToyToAffineLoweringPass::runOnFunction() {
  ...

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our *illegal*
  // operations were not converted successfully.
  auto function = getFunction();
  if (mlir::failed(mlir::applyPartialConversion(function, target, patterns)))
    signalPassFailure();
}
```

### 部分降低时的设计注意事项

在深入研究我们降低的结果之前，现在是讨论部分降低时潜在的设计注意事项的好时机。在我们的降级过程中，我们从原始类型TensorType转换为分配的(类似缓冲区的)类型MemRefType。但是，如果我们不降低`toy.print`操作，我们需要临时桥接这两个世界。有很多方法可以做到这一点，每种方法都有自己的tradeoff：

* 从缓冲区生成`load`操作
  
  一种选择是从缓冲区类型生成`load`操作，以实体化值类型的实例。这允许`toy.print`操作的定义保持不变。这种方法的缺点是，对`affine`方言的优化是有限的，因为`load`实际上会涉及到一个仅可见的完整副本*之后*我们已经执行了优化。
* 生成新版本的`toy.print`，它在降低的类型上操作
  
  另一种选择是让`toy.print`的另一个降低变种在降低的类型上操作。此选项的好处是没有到优化器的隐藏的、不必要的副本。缺点是需要另一个操作定义，它可能会重复第一个操作的许多方面。在[ODS](../../OpDefinitions.md)中定义基类可能会简化这一过程，但您仍然需要单独处理这些操作。
* 更新`toy.print`以允许在降低的类型上操作
  
  第三个选项是更新`toy.print`的当前定义，以允许在降低的类型上操作。这种方法的好处是它很简单，不会引入额外的隐藏副本，也不需要另一个操作定义。此选项的缺点是，它需要混合`Toy`方言的抽象层。

为简单起见，我们将使用第三个选项来降低。这涉及更新操作定义文件中PrintOp的类型约束：

```tablegen
def PrintOp : Toy_Op<"print"> {
  ...

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}
```

## 完整的Toy示例

让我们举一个具体的例子：

```mlir
func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

通过将仿射降低添加到我们的管道中，我们现在可以生成：

```mlir
func @main() {
  %cst = constant 1.000000e+00 : f64
  %cst_0 = constant 2.000000e+00 : f64
  %cst_1 = constant 3.000000e+00 : f64
  %cst_2 = constant 4.000000e+00 : f64
  %cst_3 = constant 5.000000e+00 : f64
  %cst_4 = constant 6.000000e+00 : f64

  // Allocating buffers for the inputs and outputs.
  %0 = alloc() : memref<3x2xf64>
  %1 = alloc() : memref<3x2xf64>
  %2 = alloc() : memref<2x3xf64>

  // Initialize the input buffer with the constant values.
  affine.store %cst, %2[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %2[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %2[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %2[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %2[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %2[1, 2] : memref<2x3xf64>

  // Load the transpose value from the input buffer and store it into the
  // next input buffer.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
      affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Multiply and store into the output buffer.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %5 = mulf %3, %4 : f64
      affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Print the value held by the buffer.
  toy.print %0 : memref<3x2xf64>
  dealloc %2 : memref<2x3xf64>
  dealloc %1 : memref<3x2xf64>
  dealloc %0 : memref<3x2xf64>
  return
}
```

## 利用仿射优化的优势

我们原生的降低是对的，但在效率上还有很多不尽如人意的地方。例如，`toy.mul`的降低产生了一些冗余负载。让我们看看向流程中添加一些现有的优化如何帮助清理这一问题。将`LoopFusion`和`MemRefDataFlowOpt`pass添加到流程中会得到以下结果：

```mlir
func @main() {
  %cst = constant 1.000000e+00 : f64
  %cst_0 = constant 2.000000e+00 : f64
  %cst_1 = constant 3.000000e+00 : f64
  %cst_2 = constant 4.000000e+00 : f64
  %cst_3 = constant 5.000000e+00 : f64
  %cst_4 = constant 6.000000e+00 : f64

  // Allocating buffers for the inputs and outputs.
  %0 = alloc() : memref<3x2xf64>
  %1 = alloc() : memref<2x3xf64>

  // Initialize the input buffer with the constant values.
  affine.store %cst, %1[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %1[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %1[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %1[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %1[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %1[1, 2] : memref<2x3xf64>

  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      // Load the transpose value from the input buffer.
      %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>

      // Multiply and store into the output buffer.
      %3 = mulf %2, %2 : f64
      affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Print the value held by the buffer.
  toy.print %0 : memref<3x2xf64>
  dealloc %1 : memref<2x3xf64>
  dealloc %0 : memref<3x2xf64>
  return
}
```

在这里，我们可以看到，删除了冗余分配，融合了两个循环嵌套，并删除了一些不必要的`load`操作。您可以构建`toyc-ch5`并亲自试用：`toyc-ch5 test/examples/Toy/CH5/affine-lowering.mlir -emit=mlir-affine`。我们也可以通过添加`-opt`来检查我们的优化。

在这一章中，我们探讨了局部降低的一些方面，目的是进行优化。在[下一章](zh-Ch-6.md)中，我们将继续讨论方言转换，将LLVM作为代码生成的目标。
