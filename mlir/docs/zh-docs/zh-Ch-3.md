# 第3章：特定于高级语言的分析和转换

[TOC]

创建紧密代表输入语言语义的方言可以实现MLIR中的分析、转换和优化，这些分析、转换和优化需要高级语言信息，并且通常在语言AST上执行。例如，`clang`在C++中执行模板实例化时有一个相当重的mechanism](https://clang.llvm.org/doxygen/classclang_1_1TreeTransform.html)。

我们将编译器转换分为两类：局部和全局。在本章中，我们将重点介绍如何利用玩具方言及其高级语义来执行在LLVM中难以实现的本地模式匹配转换。为此，我们使用MLIR的[通用DAG重写器](../../PatternRewriter.md)。

有两种方法可以实现模式匹配转换：1.命令式，C++模式匹配和重写2.声明性的、基于规则的模式匹配和重写，使用表驱动的[声明性重写规则](../../DeclarativeRewrites.md)(Drr)。请注意，DRR的使用要求使用ODS定义操作，如[第2章](CH-2.md)中所述。

## 使用C++风格的模式匹配和重写优化转置

让我们从一个简单的模式开始，尝试消除两个相互抵消的转置序列：‘transspose(transspose(X))->X’。下面是相应的玩具示例：

```toy
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

它对应于以下IR：

```mlir
func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %1 : tensor<*xf64>
}
```

这是一个很好的转换示例，在Toy IR上很难匹配，但是LLVM很难理解。例如，今天的Clang不能优化掉临时数组，使用朴素转置的计算用以下循环表示：

```c++
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       B[j][i] = A[i][j];
    }
  }
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       A[i][j] = B[j][i];
    }
  }
  sink(A);
}
```

对于一种简单的C++重写方法，包括匹配IR中的树形模式并将其替换为一组不同的操作，我们可以通过实现`RewritePattern`来插入MLIR的`Canonicalizer`过程：

```c++
/// Fold transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

该重写器的实现在`Toyota Combine.cpp`中。[规范化过程](../../Canonicalization.md)以贪婪、迭代的方式应用由操作定义的转换。为了确保规范化过程应用我们的新转换，我们设置[hasCanonicalizer=1](../../OpDefinitions.md#hascanonicalizer)并将模式注册到规范化框架。

```c++
// Register our patterns for rewrite by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyRedundantTranspose>(context);
}
```

我们还需要更新主文件`toyc.cpp`，以添加优化管道。在MLIR中，优化通过`PassManager`进行，方式与LLVM类似：

```c++
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
```

最后，我们可以运行`toyc-ch3 test/Examples/Toy/ch3/transpose_transpose.toy-emit=mlir-opt`并观察我们的模式：

```mlir
func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %arg0 : tensor<*xf64>
}
```

不出所料，我们现在直接返回函数参数，绕过任何转置操作。然而，其中一个转位仍然没有被消除。那不是很理想！发生的情况是，我们的模式用函数输入替换了最后一个转换，留下了现在死的转置输入。Canonicalizer知道清理无效的操作；但是，MLIR保守地假设操作可能有副作用。我们可以通过在我们的`TransposeOp`中添加一个新的特征`NoSideEffect`来修复这个问题：

```tablegen
def TransposeOp : Toy_Op<"transpose", [NoSideEffect]> {...}
```

现在重试`toyc-ch3 test/transpose_transpose.toy-emit=mlir-opt`：

```mlir
func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  toy.return %arg0 : tensor<*xf64>
}
```

太棒了！没有留下‘转置’操作-代码是最优的。

在下一节中，我们将使用DRR进行与重塑操作相关联的模式匹配优化。

## 使用DRR优化整形

基于规则的声明性模式匹配和重写(DRR)是基于DAG的操作声明性重写器，它为模式匹配和重写规则提供基于表的语法：

```tablegen
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
```

类似于SimplifyRedundantTranspose的冗余重塑优化可以更简单地使用DRR表示，如下所示：

```tablegen
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

可以在`path/to/BUILD/tools/mlir/examples/toy/Ch3/ToyCombine.inc`.下找到与每个DRR模式相对应的自动生成的C++代码

DRR还提供了一种方法，用于在转换取决于参数和结果的某些属性时添加参数约束。例如，当重塑是冗余的时(即，当输入和输出形状相同时)，可以消除重塑。

```tablegen
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

某些优化可能需要对指令参数进行额外的转换。这是使用NativeCodeCall实现的，它允许通过调用C++帮助器函数或使用内联C++进行更复杂的转换。这种优化的一个例子是FoldConstantReshape，我们通过就地重塑常量并消除重塑操作来优化常量值的重塑。

```tablegen
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

我们使用下面的trivial_reshape.toy程序演示这些重塑优化：

```c++
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

```mlir
module {
  func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

我们可以尝试运行`toyc-ch3 test/Examples/Toy/ch3/trivial_reshape.toy-emit=mlir-opt`，并实际观察我们的模式：

```mlir
module {
  func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

正如预期的那样，规范化后不再保留任何重塑操作。

有关声明性重写方法的更多详细信息，请参阅[表驱动声明性重写规则(DRR)](../../DeclarativeRewrites.md)。

在本章中，我们了解了如何通过始终可用的钩子使用特定的核心转换。在[下一章](CH-4.md)中，我们将了解如何使用通过Interfaces更好地扩展的通用解决方案。
