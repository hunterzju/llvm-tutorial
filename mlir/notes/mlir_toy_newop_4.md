# MLIR-Toy-实践-4-转换到LLVM IR运行
之前的[文章](https://zhuanlan.zhihu.com/p/444428735)基于MLIR中的Toy教程添加了操作OrOp，并从Toy Dialect降级到了`Standard Op`。本文主要记录了最终降级到`LLVM Dialect`并调用LLVM JIT执行的过程。

![MLIR Pattern]](https://pic4.zhimg.com/80/v2-e7b8fe24b71ad47d664748006f05c260.png)

LLVM中Pattern提供了对IR的便捷操作方式，其中ConversionPattern主要用于Dialect间的转换。而ConversionPatter又分为PartialConversion和FullConversion，上篇文章使用PartialConversion执行了部分降级，本文主要使用了Full Conversion将所有Op降级到LLVM Dialect。

## 降级到LLVM Dialect
当期获得的IR中除了`toy.print`，其余Op都被降级到了MLIR先有的几种Dialect中（`Standard`,`Affine`,`Memref`等），这些Dialect都提供了可以降级到LLVM Dialect的接口，而`toy.print`则需要单独实现从`toy`到`llvm`的转换方法。

### print降级到LLVM
这里需要定义一个`ConversionPattern`实现`toy.print`到`llvm`的转换，方法和之前一样：继承`ConversionPattern`并重写`matchAndRewrite`方法。
```c++
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // ...
  }
}
```
还需要在llvm中创建一个叫做`printf`的`FuncOp`来代替`toy.print`操作，该函数的返回值是int类型，输入参数是指向字符串的指针，具体创建过程如下：
```c++
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }
```

### 其他Dialect降级到LLVM
其他方言的转换可以利用已有的转换方法，直接将转换Pattern添加进去即可，转换Pattern的实现在`void ToyToLLVMLoweringPass::runOnOperation()`中完成：
```c++
void ToyToLLVMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.add<PrintOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
```
这个过程中首先定义了转换目标`LLVMConversionTarget`和`legalOp`，在添加转换Pattern后应用`applyFullConversion()`转换到LLVM Dialect中。

## CodeGen：输出LLVM IR并使用JIT运行
最后就可以从LLVM Dialect导出LLVM IR，然后调用LLVM JIT执行了。

导出LLVM IR过程将MLIR Module转换到LLVM IR表示，可以直接调用已有接口(`toyc.cpp`中`dumpLLVMIR()`实现)：
```c++
auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
```

调用JIT使用了MLIR中的`mlir::ExecutionEngine`，使用和LLVM中类似，具体实现在`toyc.cpp`中的`runJIT()`中：
```c++
int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

## 总结
至此，新加入的Op已经可以从toy源码中解析到MLIR Toy Dialect中，最终转化到LLVM JIT中执行了。测试命令：
```bash
./bin/toyc-ch6 ../../testcode/Ch2/codegen.toy --emit=jit 
```

整个过程主要熟悉了MLIR中Dialect，ODS，Pattern，Pass这些基础概念和功能。作为一种编译基础设施，MLIR的整个可玩性还是很高的，后续会做更多的探索。
