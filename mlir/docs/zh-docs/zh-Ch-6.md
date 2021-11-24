# 第6章：降低到LLVM和代码生成

在[上一章](zh-Ch-5.md)中，我们介绍了[方言转换](../../DialectConversion.md)框架，并将很多`toy`操作部分降为仿射循环嵌套进行优化。在本章中，我们将最终降低到LLVM进行代码生成。

## 降低到LLVM

对于这一下降，我们将再次使用方言转换框架来执行繁琐的工作。但是，这次我们将执行到[LLVM方言](../../方言/LLVM.md)的完全转换。谢天谢地，我们已经降低了所有的`toy`操作，只有一个除外，最后一个是`toy.print`。在完成到LLVM的转换之前，我们先降低`toy.print`操作。我们将此操作降低到为每个元素调用`printf`的非仿射循环嵌套。注意，因为方言转换框架支持[传递性lowering](../getting_started/Glossary.md#transitive-lowering)，所以我们不需要用LLVM方言直接发出操作。通过传递性降低，我们的意思是转换框架可以应用多个模式来使操作完全合法化。在本例中，我们生成的是结构化循环嵌套，而不是LLVM方言中的分支形式。只要我们有一个从循环操作到LLVM的降级，降级仍然会成功。

在降低过程中，我们可以通过如下方式获得或构建printf的声明：

```c++
/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get("printf", context);

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get("printf", context);
}
```

既然已经定义了printf操作的降低，我们可以指定降低所需的组件。这些组件与[上一章](zh-Ch-5.md)中定义的组件基本相同。

### 转换目标

对于此转换，除了顶级模块之外，我们将把所有内容都降低到LLVM方言。

```c++
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
```

### 类型转换器

这种降低还会将当前正在操作的MemRef类型转换为LLVM中的表示形式。要执行此转换，我们使用TypeConverter作为降级的一部分。此转换器指定一种类型如何映射到另一种类型。由于我们正在执行更复杂的涉及block参数的下降，使用转换器是必要的。假设我们没有任何需要降低的特定于Toy方言的类型，那么对于我们的用例来说，默认的转换器就足够了。

```c++
  LLVMTypeConverter typeConverter(&getContext());
```

### 转换模式

既然已经定义了转换目标，我们需要提供用于降低的模式。在编译过程中的这一点上，我们组合了`toy`、`affine`和`std`操作。幸运的是，`std`和`affine`方言已经提供了将它们转换为LLVM方言所需的模式集。这些模式允许通过依赖[传递性lowering](../getting_started/Glossary.md#transitive-lowering)来通过多个阶段降低IR。

```c++
  mlir::OwningRewritePatternList patterns;
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
  mlir::populateLoopToStdConversionPatterns(patterns, &getContext());
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation, to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.insert<PrintOpLowering>(&getContext());
```

### 完全降级

我们希望完全降到LLVM，所以我们使用`FullConversion`。这确保在转换后只保留合法的操作。

```c++
  mlir::ModuleOp module = getOperation();
  if (mlir::failed(mlir::applyFullConversion(module, target, patterns)))
    signalPassFailure();
```

回过头来看我们当前的工作示例：

```mlir
func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

现在，我们可以向下查看LLVM方言，它会生成以下代码：

```mlir
llvm.func @free(!llvm<"i8*">)
llvm.func @printf(!llvm<"i8*">, ...) -> i32
llvm.func @malloc(i64) -> !llvm<"i8*">
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64

  ...

^bb16:
  %221 = llvm.extractvalue %25[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %222 = llvm.mlir.constant(0 : index) : i64
  %223 = llvm.mlir.constant(2 : index) : i64
  %224 = llvm.mul %214, %223 : i64
  %225 = llvm.add %222, %224 : i64
  %226 = llvm.mlir.constant(1 : index) : i64
  %227 = llvm.mul %219, %226 : i64
  %228 = llvm.add %225, %227 : i64
  %229 = llvm.getelementptr %221[%228] : (!llvm."double*">, i64) -> !llvm<"f64*">
  %230 = llvm.load %229 : !llvm<"double*">
  %231 = llvm.call @printf(%207, %230) : (!llvm<"i8*">, f64) -> i32
  %232 = llvm.add %219, %218 : i64
  llvm.br ^bb15(%232 : i64)

  ...

^bb18:
  %235 = llvm.extractvalue %65[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %236 = llvm.bitcast %235 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%236) : (!llvm<"i8*">) -> ()
  %237 = llvm.extractvalue %45[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %238 = llvm.bitcast %237 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%238) : (!llvm<"i8*">) -> ()
  %239 = llvm.extractvalue %25[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %240 = llvm.bitcast %239 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%240) : (!llvm<"i8*">) -> ()
  llvm.return
}
```

有关降低到LLVM方言的更多详细信息，请参见[转换为LLVM IR方言](../../ConversionToLLVMDialect.md)。

## CodeGen：摆脱MLIR

此时，我们正处于代码生成的节骨眼。我们可以用LLVM方言生成代码，所以现在我们只需要导出到LLVM IR并设置一个JIT来运行它。

### 发射LLVM IR

既然我们的模块只包含LLVM方言的操作，我们就可以导出到LLVM IR。要以编程方式完成此操作，我们可以调用以下实用程序：

```c++
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module);
  if (!llvmModule)
    /* ... an error was encountered ... */
```

将我们的模块导出到LLVM IR会生成：

```llvm
define void @main() {
  ...

102:
  %103 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %104 = mul i64 %96, 2
  %105 = add i64 0, %104
  %106 = mul i64 %100, 1
  %107 = add i64 %105, %106
  %108 = getelementptr double, double* %103, i64 %107
  %109 = load double, double* %108
  %110 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double %109)
  %111 = add i64 %100, 1
  br label %99

  ...

115:
  %116 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %24, 0
  %117 = bitcast double* %116 to i8*
  call void @free(i8* %117)
  %118 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %16, 0
  %119 = bitcast double* %118 to i8*
  call void @free(i8* %119)
  %120 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %121 = bitcast double* %120 to i8*
  call void @free(i8* %121)
  ret void
}
```

如果我们对生成的LLVM IR启用优化，我们可以将其大幅削减：

```llvm
define void @main()
  %0 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.000000e+00)
  %1 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.600000e+01)
  %putchar = tail call i32 @putchar(i32 10)
  %2 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 4.000000e+00)
  %3 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 2.500000e+01)
  %putchar.1 = tail call i32 @putchar(i32 10)
  %4 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 9.000000e+00)
  %5 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 3.600000e+01)
  %putchar.2 = tail call i32 @putchar(i32 10)
  ret void
}
```

转储LLVM IR的完整代码清单可在`Examples/toy/ch6/toy.cpp`中的`dumpLLVMIR()`函数中：

```c++

int dumpLLVMIR(mlir::ModuleOp module) {
  // Translate the module, that contains the LLVM dialect, to LLVM IR. Use a
  // fresh LLVM IR context. (Note that LLVM is not thread-safe and any
  // concurrent use of a context requires external locking.)
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

### 设置JIT

可以使用`mlir::ExecutionEngine`基础设施设置JIT来运行包含LLVM方言的模块。这是一个围绕LLVM的JIT的实用程序包装，接受`.mlir`作为输入。设置JIT的完整代码清单可以在`ch6/toyc.cpp`中的`runJit()`函数中找到：

```c++
int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(module, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

您可以从build目录使用它：

```shell
$ echo 'def main() { print([[1, 2], [3, 4]]); }' | ./bin/toyc-ch6 -emit=jit
1.000000 2.000000
3.000000 4.000000
```

您也可以通过`-emit=mlir`，`-emit=mlir-affine`，`-emit=mlir-llvm`，`-emit=llvm`来比较不同等级的IR。还可以尝试像[`--print-ir-after-all`](../../PassManagement.md#ir-printing)这样的选项来跟踪整个流程中IR的演变。

本节使用的示例代码可以在`test/Examples/Toy/ch6/llvm-lowering.mlir`中找到。

到目前为止，我们已经使用了原始数据类型。在[下一章](zh-Ch-7.md)中，我们将添加一个复合的`struct`类型。
