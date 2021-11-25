# 构建JIT：按函数延迟编译
=

**本教程正在积极开发中。它是不完整的，细节可能会经常变化。** Nonetheless we invite you to try it out as it stands, and we welcome any feedback.

## 第三章绪论
* * *

**警告：由于ORC API更新，此文本当前已过期。**

**示例代码已经更新，可以使用了。一旦API的波动平息下来，文本将会更新。**

欢迎学习`在LLVM中构建基于ORC的JIT`教程的第3章。本章讨论惰性JITing，并向您展示如何通过添加[第2章](zh-BuildingAJIT2.md)中的JIT的ORC CompileOnDemand层来启用它。

## 懒惰编译
* * *

当我们将模块添加到第2章中的KaleidoscopeJIT类中时，它将分别由IRTransformLayer、IRCompileLayer和RTDyldObjectLinkingLayer分别为我们优化、编译和链接。在这个方案中，使模块可执行的所有工作都是预先完成的，它很容易理解，而且其性能特征也很容易推断。但是，如果要编译的代码量很大，会导致非常长的启动时间，如果在运行时只调用几个编译过的函数，还可能会进行大量不必要的编译。一个真正的`即时`编译器应该允许我们将任何给定函数的编译推迟到该函数第一次被调用的那一刻，从而缩短启动时间并消除多余的工作。事实上，ORCAPI为我们提供了一个用于延迟编译编译IR的层：*CompileOnDemandLayer Layer OnDemandLayer*。

CompileOnDemandLayer类符合第2章中描述的Layer接口，但是它的addModule方法的行为与我们到目前为止看到的层有很大的不同：它只扫描要添加的模块，并安排它们中的每个函数在第一次调用时编译，而不是预先做任何工作。为此，CompileOnDemandLayer为它扫描的每个函数创建两个小实用程序：a*存根*和a*编译回调*。存根是一对函数指针(一旦编译完函数，它将指向函数的实现)和通过指针的间接跳转。通过在程序的生存期内固定间接跳转的地址，我们可以给函数一个永久的“有效地址”，即使函数的实现从未编译过，或者如果它被编译了多次(例如，由于在更高的优化级别上重新编译函数)并更改了地址，该地址也可以安全地用于间接和函数指针比较。第二个实用程序是编译回调，它表示从程序到编译器的重新入口点，它将触发函数的编译和执行。通过将函数的存根初始化为指向函数的编译回调，我们启用了惰性编译：第一次尝试调用函数将跟随函数指针并触发编译回调。编译回调将编译函数，更新存根的函数指针，然后执行函数。在对该函数的所有后续调用中，函数指针将指向已编译的函数，因此编译器不会产生更多开销。我们将在本教程的下一章中更详细地研究这个过程，但现在我们将信任CompileOnDemandLayer来为我们设置所有的存根和回调。我们所需要做的就是将CompileOnDemandLayer添加到堆栈的顶部，我们将获得懒惰编译的好处。我们只需要对源代码做几处更改：

```c++
...
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
...

...
class KaleidoscopeJIT {
private:
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompileLayer;

  using OptimizeFunction =
      std::function<std::shared_ptr<Module>(std::shared_ptr<Module>)>;

  IRTransformLayer<decltype(CompileLayer), OptimizeFunction> OptimizeLayer;

  std::unique_ptr<JITCompileCallbackManager> CompileCallbackManager;
  CompileOnDemandLayer<decltype(OptimizeLayer)> CODLayer;

public:
  using ModuleHandle = decltype(CODLayer)::ModuleHandleT;
```

首先，我们需要包括CompileOnDemandLayer.h头，然后向我们的类添加两个新成员：std::Unique\_ptr\<JITCompileCallbackManager\>和CompileOnDemandLayer。CompileOnDemandLayer使用CompileCallbackManager成员创建每个函数所需的编译回调。

```c++
KaleidoscopeJIT()
    : TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
      ObjectLayer([]() { return std::make_shared<SectionMemoryManager>(); }),
      CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
      OptimizeLayer(CompileLayer,
                    [this](std::shared_ptr<Module> M) {
                      return optimizeModule(std::move(M));
                    }),
      CompileCallbackManager(
          orc::createLocalCompileCallbackManager(TM->getTargetTriple(), 0)),
      CODLayer(OptimizeLayer,
               [this](Function &F) { return std::set<Function*>({&F}); },
               *CompileCallbackManager,
               orc::createLocalIndirectStubsManagerBuilder(
                 TM->getTargetTriple())) {
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
}
```

接下来，我们必须更新构造函数以初始化新成员。为了创建适当的编译回调管理器，我们使用createLocalCompileCallbackManager函数，该函数在收到编译未知函数的请求时接受TargetMachine和JITTargetAddress调用。在我们的简单JIT中，这种情况不太可能出现，所以我们将作弊并在这里传递\‘0\’。在产品质量JIT中，您可以给出抛出异常的函数的地址，以便展开JIT代码的堆栈。

现在我们可以构造CompileOnDemandLayer了。按照前几层的模式，我们首先将引用传递到堆栈中向下的下一层--OptimizeLayer。接下来，我们需要提供一个`Partitioning Function`：当调用一个尚未编译的函数时，CompileOnDemandLayer将调用该函数来询问我们要编译什么。我们至少需要编译被调用的函数(由分区函数的参数给出)，但我们也可以请求CompileOnDemandLayer编译从被调用的函数无条件调用(或极有可能被调用)的其他函数。对于KaleidoscopeJIT，我们将保持简单，只请求编译所调用的函数。接下来，我们将一个引用传递给我们的CompileCallbackManager。最后，我们需要提供一个`间接存根管理器构建器`：一个构造IndirectStubManagers的实用函数，而IndirectStubManagers又用于为每个模块中的函数构建存根。CompileOnDemandLayer将为每个对addModule的调用调用一次间接存根管理器构建器，并使用生成的间接存根管理器为集合中所有模块中的所有函数创建存根。如果/当模块集从JIT中删除时，间接存根管理器将被删除，从而释放分配给存根的任何内存。我们通过使用createLocalIndirectStubsManagerBuilder实用程序提供此函数。

```c++
// ...
        if (auto Sym = CODLayer.findSymbol(Name, false))
// ...
return cantFail(CODLayer.addModule(std::move(Ms),
                                   std::move(Resolver)));
// ...

// ...
return CODLayer.findSymbol(MangledNameStream.str(), true);
// ...

// ...
CODLayer.removeModule(H);
// ...
```

最后，我们需要替换addModule、findSymbol和removeModule方法中对OptimizeLayer的引用。有了这些，我们就可以开工了。

**待完成的工作：**

\*\*章节结论。\*\*

## 完整代码列表
* * *

下面是我们的运行示例的完整代码清单，其中包含
添加CompileOnDemand图层以启用一次延迟功能
编译。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

```
../../examples/Kaleidoscope/BuildingAJIT/Chapter3/KaleidoscopeJIT.h
```

[下一步：极度懒惰\--直接从AST使用JIT的编译回调](zh-BuildingAJIT4.md)
