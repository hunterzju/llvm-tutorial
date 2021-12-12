#! https://zhuanlan.zhihu.com/p/444428735
# MLIR-Toy-实践-3-Dialect转换
[上篇文章](https://zhuanlan.zhihu.com/p/441471026)为Toy添加了一个新Op（`toy.or`）表示逻辑或。本文介绍如何将OrOp降低到其他方言对应的Op，主要用到了`RewritePattern`和`ConversionPattern`相关的内容。

## RewritePattern与ConversionPattern
MLIR是一种图类型的IR表示，而RewritePattern提供了一个图模式匹配的接口，可以更方便进行图优化。比如[ToyTutorial-ch3](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)中使用的优化pattern：将两个嵌套的transport转换为一个返回输入数据的节点。

![Image](https://pic4.zhimg.com/80/v2-5e9cba8238323d84be4bf983d55554d2.png)

RewritePattern的实现有两种方式，一种是采用c++实现，需要定义一个转换结构体继承`mlir::OpRewritePattern<TransposeOp>`，并重写`matchAndRewrite()`方法，该方法中实现了IR结构的修改逻辑。比如上文中提到的`Transpose`逻辑优化，在transpose嵌套transpose操作时，两次转置操作抵消，直接返回输入参数。定义该Pattern后，创建一个标准化pass（在`toy.cpp`中实现），并将Pattern注册到该Pass中（在`ToyCombine.cpp`中实现）。

```c++
/// ToyCombine.cpp 定义pattern：transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  // ....
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

// toy.cpp 创建pass
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());

// ToyCombine.cpp 注册pattarn
void TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

// 在ops.td中声明允许标准化操作
let hasCanonicalizer = 1;

```
RewritePattern的另一种实现方式是采用DRR描述Pattern，然后通过TableGen来生成c代码。在[ToyTutorial-ch3](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)中采用DRR方式实现了`Reshape`操作的优化。DRR定义规则如下：
```mlir
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
```
比如，定义对`Reshape`操作的优化如下，其中`sourcePattern`是`(ReshapeOp(ReshapeOp $arg))`,`resultPatterns`是`(ReshapeOp $arg)`，约束Constraints和优先级benefits都省略没有定义：
```mlir
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

`ConversionPattern`是一种特殊的`RewritePattern`，用于实现Dialect之间的转换。在Dialect转换过程中，可能会对`Operation`中的操作数做修改，因而`ConversionPattern`和`RewritePattern`一个主要区别是`matchAndRewrite()`接口函数中多了一个`operands`参数，用于对`Operation`中的操作数修改。
```c++
struct MyConversionPattern : public ConversionPattern {
  /// The `matchAndRewrite` hooks on ConversionPatterns take an additional
  /// `operands` parameter, containing the remapped operands of the original
  /// operation.
  virtual LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const;
};
```

## ConversionPattern实现Dialect转换
在[上一篇文章](https://zhuanlan.zhihu.com/p/441471026)中，我们给`Toy Dialect`添加了一个逻辑或操作`OrOp`，下文结合`Conversion Pattern`的使用记录下将`Toy Dialect`中的`OrOp`转换到其他`dialect`的过程。`Dialect`转换需要指定`Conversion Target`（目标方言）和`Rewrite Patterns`（匹配Operation）。

首先指定`Conversion Target`，这里将`MLIR Dialect`转换到`Affine`, `MemRef` and `Standard` 三种`Dialect`，为后续转换到可运行的`LLVM Dialect`做准备。具体实现在`LowerToAffineLoops.cpp`中，指定了合法和非法的Dialect以及Operation：

```c++
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `MemRef` and `Standard` dialects.
  target.addLegalDialect<AffineDialect, memref::MemRefDialect,
                         StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`.
  target.addIllegalDialect<toy::ToyDialect>();
  target.addLegalOp<toy::PrintOp>();
```

接下来指定转换匹配的Pattern，具体实现如上一节描述，先定义一个转换Pattern类，该类继承了`ConversionPattern`；然后重载其中的`matchAndRewrite()`方法来指定转换操作；接下来将这些Pattern添加到转换context中；最后执行转换。
```c++
  // 转换pattern定义
  struct TransposeOpLowering : public ConversionPattern {
    // ...
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter)
    // ...
  ｝
  // 添加pattern到context
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, MulOpLowering,
               ReturnOpLowering, TransposeOpLowering>(&getContext());
  
  // 应用转换过程
  applyPartialConversion(getFunction(), target, std::move(patterns))
```

## OrOp转换
将新添加的`toy.or`进行转换，需要实现一个转换Pattern，并将其添加到转换Context中。参考已经实现的`Add`和`Mul`操作，其都是将`Toy Dialect`先通过`Affine Dialect`将循环展开，然后转换到`Standard Dialect`中的对应Op。这里有一个问题是，在`Standard Dialect`中Add和Mul都有对应的浮点和整型操作，但是Or仅支持整型操作（这是符合运算逻辑的，对于整型逻辑或才有意义），但是输入数据是浮点型F64。因此，OrOp需要做一个浮点转整型的操作。同时由于后续操作都是在浮点上操作的，因此还需要将OrOp的结果操作数从整型转回浮点。
```c++
// 对OrOp添加Float转Int逻辑，并映射到Standard::OrOp
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc, op](OpBuilder &builder, ValueRange memRefOperands,
              ValueRange loopIvs) {
          // ...

          // 对toy.or添加float转int逻辑，利用standard中的FPToUIOp
          auto opname = op->getName();
          if (opname.getStringRef().str() == "toy.or") {
            auto castLhs = builder.create<FPToUIOp>(loc, builder.getI64Type(), loadedLhs);
            auto castRhs = builder.create<FPToUIOp>(loc, builder.getI64Type(), loadedRhs);
            
            return builder.create<LoweredBinaryOp>(loc, castLhs, castRhs);
          }
          // ....
        });
    return success();
  }
};
// OrOp pattern定义
using OrOpLowering = BinaryOpLowering<toy::OrOp, OrOp>;

//
static void lowerOpToLoops(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  // ...
  buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        // 将"toy.or" Op的结果从Int转换为Float
        if(nestedBuilder.getI64Type() == valueToStore.getType()) {
          valueToStore = nestedBuilder.create<UIToFPOp>(loc, nestedBuilder.getF64Type(), valueToStore);
        }
        nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
      });
  // ...
}
```

## 总结
MLIR中基于Pattern对IR图进行操作，提供了一个便捷且标准化的接口，带来了很大便利性，但是也增加了学习成本。Conversion Pattern提供了一套在Dialect间进行转换的通路，别且多个Dialect可以共存，有点类似于插件的感觉。由于目前了解有限，总感觉各个Dialect之间的抽象关系不是很明确，而且暂时没找到一个文档解释对各个Dialect的抽象层级进行一个比较系统的解释，可能是个有待改进的点吧。

（*另外，关于OrOp类型转换的问题，感觉处理得有点野路子，欢迎有想法的朋友指正。*）