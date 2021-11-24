# 构建JIT：添加优化\--ORC层简介
=

**本教程正在积极开发中。它是不完整的，细节可能会经常变化。** Nonetheless we invite you to try it out as it stands, and we welcome any feedback.

## 第二章绪论
* * *

**警告：本教程当前正在更新，以解决ORC API更改。只有第1章和第2章是最新的。**

**第3章至第5章中的示例代码将编译并运行，但尚未更新**

欢迎学习\“在LLVM中构建基于ORC的JIT\”教程的第2章。在本系列的[第1章](zh-BuildingAJIT1.html)中，我们研究了一个基本的JIT类KaleidoscopeJIT，它可以将LLVM IR模块作为输入并在内存中生成可执行代码。KaleidoscopeJIT通过编写两个现成的*ORCLayer*：IRCompileLayer和ObjectLinkingLayer，用相对较少的代码就能做到这一点，以完成大部分繁重的工作：IRCompileLayer和ObjectLinkingLayer。

在这一层中，我们将通过使用新的层IRTransformLayer来向KaleidoscopeJIT添加IR优化支持，从而更多地了解ORC层的概念。

## 使用IRTransformLayer优化模块
* * *

在\“使用LLVM实现语言\”教程系列的[第4章](LangImpl04.html)中，介绍了llvm*FunctionPassManager*作为优化LLVM IR的一种方法。感兴趣的读者可能会阅读该章以获取详细信息，但简而言之：为了优化模块，我们创建一个llvm::FunctionPassManager实例，使用一组优化对其进行配置，然后在模块上运行PassManager以将其变异为(希望)更优化但语义等价的形式。在最初的教程系列中，FunctionPassManager是在KaleidoscopeJIT之外创建的，模块在添加到它之前进行了优化。在本章中，我们将把优化作为JIT的一个阶段。目前，这将为我们提供学习更多关于ORC层的动力，但从长远来看，使优化成为我们JIT的一部分将产生一个重要的好处：当我们开始懒惰地编译代码时(即，将每个函数的编译推迟到它第一次运行时)，由我们的JIT管理的优化也将允许我们懒惰地进行优化，而不必预先进行所有的优化。

为了给我们的JIT增加优化支持，我们将第一章中的KaleidoscopeJIT放在上面，组成一个ORC*IRTransformLayer*。下面我们将更详细地介绍IRTransformLayer的工作原理，但接口很简单：该层的构造函数引用执行会话和下面的层(与所有层一样)，外加一个*IR优化函数*，它将应用于通过addModule添加的每个模块：

```c++
class KaleidoscopeJIT {
private:
  ExecutionSession ES;
  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer CompileLayer;
  IRTransformLayer TransformLayer;

  DataLayout DL;
  MangleAndInterner Mangle;
  ThreadSafeContext Ctx;

public:

  KaleidoscopeJIT(JITTargetMachineBuilder JTMB, DataLayout DL)
      : ObjectLayer(ES,
                    []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(ES, ObjectLayer, ConcurrentIRCompiler(std::move(JTMB))),
        TransformLayer(ES, CompileLayer, optimizeModule),
        DL(std::move(DL)), Mangle(ES, this->DL),
        Ctx(std::make_unique<LLVMContext>()) {
    ES.getMainJITDylib().setGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
  }
```

我们的扩展KaleidoscopeJIT类开始时与第1章相同，但是在CompileLayer之后，我们引入了一个新成员TransformLayer，它位于CompileLayer之上。我们使用对ExecutionSession和输出层(层的标准实践)的引用以及*变换函数*来初始化OptimizeLayer。对于我们的转换函数，我们提供了我们的类OptimizeModule静电方法。

```c++
// ...
return cantFail(OptimizeLayer.addModule(std::move(M),
                                        std::move(Resolver)));
// ...
```

接下来，我们需要更新addModule方法以替换对
`CompileLayer::add`，改为调用`OptimizeLayer::add`。

```c++
static Expected<ThreadSafeModule>
optimizeModule(ThreadSafeModule M, const MaterializationResponsibility &R) {
  // Create a function pass manager.
  auto FPM = std::make_unique<legacy::FunctionPassManager>(M.get());

  // Add some optimizations.
  FPM->add(createInstructionCombiningPass());
  FPM->add(createReassociatePass());
  FPM->add(createGVNPass());
  FPM->add(createCFGSimplificationPass());
  FPM->doInitialization();

  // Run the optimizations over all functions in the module being added to
  // the JIT.
  for (auto &F : *M)
    FPM->run(F);

  return M;
}
```

在我们的JIT的底部，我们添加了一个私有方法来执行实际的优化：*OptimizeModule*。此函数接受要转换的模块作为输入(作为ThreadSafeModule)以及对新类`MaterializationResponsibility`的引用。MaterializationResponsibility参数可用于查询正在转换的模块的JIT状态，例如JIT代码正在主动尝试调用/访问的模块中的一组定义。目前，我们将忽略此论点，并使用标准优化管道。为此，我们设置一个FunctionPassManager，向其添加一些PASS，在模块中的每个函数上运行它，然后返回变异的模块。具体的优化与“用LLVM实现语言”系列教程的[第4章](LangImpl04.html)中使用的相同。读者可以访问这一章，更深入地讨论这些，以及一般的IR优化。

这就是对KaleidoscopeJIT的更改：当通过addModule添加模块时，OptimizeLayer将在将转换后的模块传递到下面的CompileLayer之前调用我们的OptimizeModule函数。当然，我们可以直接在addModule函数中调用OptimizeModule，而不必费心使用IRTransformLayer，但这样做给了我们另一个机会来了解层是如何组成的。它还为*图层*概念提供了一个简洁的入口点

```c++
// From IRTransformLayer.h:
class IRTransformLayer : public IRLayer {
public:
  using TransformFunction = std::function<Expected<ThreadSafeModule>(
      ThreadSafeModule, const MaterializationResponsibility &R)>;

  IRTransformLayer(ExecutionSession &ES, IRLayer &BaseLayer,
                   TransformFunction Transform = identityTransform);

  void setTransform(TransformFunction Transform) {
    this->Transform = std::move(Transform);
  }

  static ThreadSafeModule
  identityTransform(ThreadSafeModule TSM,
                    const MaterializationResponsibility &R) {
    return TSM;
  }

  void emit(MaterializationResponsibility R, ThreadSafeModule TSM) override;

private:
  IRLayer &BaseLayer;
  TransformFunction Transform;
};

// From IRTransformLayer.cpp:

IRTransformLayer::IRTransformLayer(ExecutionSession &ES,
                                   IRLayer &BaseLayer,
                                   TransformFunction Transform)
    : IRLayer(ES), BaseLayer(BaseLayer), Transform(std::move(Transform)) {}

void IRTransformLayer::emit(MaterializationResponsibility R,
                            ThreadSafeModule TSM) {
  assert(TSM.getModule() && "Module must not be null");

  if (auto TransformedTSM = Transform(std::move(TSM), R))
    BaseLayer.emit(std::move(R), std::move(*TransformedTSM));
  else {
    R.failMaterialization();
    getExecutionSession().reportError(TransformedTSM.takeError());
  }
}
```

这是来自`llvm/include/llvm/ExecutionEngine/Orc/IRTransformLayer.h`和`llvm/lib/ExecutionEngine/Orc/IRTransformLayer.cpp`.的IRTransformLayer的完整定义这个类涉及两个非常简单的工作：(1)通过Transform函数对象运行通过该层发出的每个IR模块，以及(2)实现ORC‘IRLayer`接口(该接口本身符合一般的ORC层概念，详见下文)。该类的大部分内容都很简单：用于转换函数的tyecif、用于初始化成员的构造函数、用于转换函数值的设置器和默认的无操作转换。最重要的方法是`emit`，因为这是我们IRLayer接口的一半。emit方法将我们的转换应用于它调用的每个模块，如果转换成功，则将转换后的模块传递到基本层。如果转换失败，我们的emit函数将调用`MaterializationResponsibility::failMaterialization`(这个jit客户端可能正在等待其他线程，他们知道自己等待的代码编译失败)，并在退出之前用执行会话记录错误。

我们从IRLayer类继承的IRLayer接口的另一半未经修改：

```c++
Error IRLayer::add(JITDylib &JD, ThreadSafeModule TSM, VModuleKey K) {
  return JD.define(std::make_unique<BasicIRLayerMaterializationUnit>(
      *this, std::move(K), std::move(TSM)));
}
```

此代码来自`llvm/lib/ExecutionEngine/orc/layer.cpp`，通过将给定的JITDylib包装在`MaterializationUnit`(本例中为`BasicIRLayerMaterializationUnit`)中，将ThreadSafeModule添加到该JITDylib。从IRLayer派生的大多数层都可以依赖这个`add`方法的默认实现。

这两个操作，`add`和`emit`一起构成了层的概念：层是一种包装编译器流水线的一部分(在本例中是LLVM编译器的\“opt\”阶段)的方式，其API对于ORC是不透明的，该接口允许ORC在需要时调用它。Add方法采用某种输入程序表示形式的模块(在本例中为LLVM IR模块)，并将其存储在目标JITDylib中，以便在请求该模块定义的任何符号时将其传递回层的emit方法。层可以通过调用基本层的\‘emit\’方法来整齐地组合起来，以完成它们的工作。(=例如，在本教程中，IRTransformLayer调用IRCompileLayer来编译转换后的IR，IRCompileLayer反过来调用ObjectLayer来链接编译器生成的目标文件。

到目前为止，我们已经学习了如何优化和编译我们的LLVM IR，但是我们还没有关注编译何时发生。我们当前的REPL非常迫切：每个函数定义只要被任何其他代码引用，无论它是否在运行时被调用，都会立即进行优化和编译。在下一章中，我们将介绍完全懒惰编译，在这种编译中，函数直到在运行时第一次被调用时才被编译。在这一点上，权衡变得更加有趣：我们越懒惰，开始执行第一个函数的速度就越快，但我们将不得不更频繁地暂停编译新遇到的函数。如果我们只是懒惰地生成代码，而急切地进行优化，我们将有更长的启动时间(因为一切都是优化的)，但相对较短的暂停时间，因为每个函数都只是通过代码生成。如果我们同时懒于优化和代码生成，我们可以更快地开始执行第一个函数，但我们会有更长的停顿，因为每个函数在第一次执行时都必须进行优化和代码生成。如果我们考虑过程间优化，如内联，事情就会变得更加有趣，这必须急切地执行。这些都是复杂的权衡，没有万能的解决方案，但通过提供可组合的层，我们将决策留给实现JIT的人，从而使他们很容易进行实验

[下一步：添加每个函数的惰性编译](zh-BuildingAJIT3.md)

## 完整代码列表
* * *

下面是我们的运行示例的完整代码清单，其中添加了IRTransformLayer以实现优化。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

```
../../examples/Kaleidoscope/BuildingAJIT/Chapter2/KaleidoscopeJIT.h
```
