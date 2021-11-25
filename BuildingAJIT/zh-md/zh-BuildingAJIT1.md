# 构建JIT：从KaleidoscopeJIT开始
=

## 第一章绪论
* * *

**警告：本教程当前正在更新，以解决ORC API更改。只有第1章和第2章是最新的。**

**第3章至第5章中的示例代码将编译并运行，但尚未更新**

欢迎学习\“在LLVM中构建基于ORC的JIT\”教程的第1章。本教程贯穿使用LLVM的按请求编译(ORC)API的JIT编译器的实现过程，首先介绍[使用LLVM实现语言](LangImpl01.html)教程中使用的KaleidoscopeJIT类的简化版本，然后介绍并发编译、优化、延迟编译和远程执行等新特性。

本教程的目标是向您介绍LLVM的ORC JIT API，展示这些API如何与LLVM的其他部分交互，并教您如何重新组合它们来构建适合您的用例的自定义JIT。

本教程的结构如下：

- 第一章：研究简单的KaleidoscopeJIT类。这将介绍ORC JIT API的一些基本概念，包括ORC*图层*的概念。
- [第2章](zh-BuildingAJIT2.html)：通过添加一个新的层来扩展基本的KaleidoscopeJIT，该层将优化IR和生成的代码。
- [第3章](zh-BuildingAJIT3.html)：通过添加按需编译层来延迟编译IR，从而进一步扩展JIT。
- [第4章](zh-BuildingAJIT4.html)：通过将按需编译层替换为直接使用ORC编译回调API以将IR生成推迟到调用函数的定制层，来改进JIT的惰性。
- [第5章](zh-BuildingAJIT5.html)：使用JIT远程API将JITing代码添加到权限降低的远程进程中。

为了为我们的JIT提供输入，我们将使用\“在LLVM教程中实现语言\”的[第7章](LangImpl07.html)中略微修改的万花筒REPL版本。

最后，关于API生成：ORC是LLVM JIT API的第三代。在此之前是MCJIT，在此之前是遗留的JIT(现已删除)。这些教程不假定您有使用这些早期API的任何经验，但熟悉它们的读者会看到许多熟悉的元素。在适当的情况下，我们将明确说明与早期API的这种联系，以帮助从它们过渡到ORC的人们。

## JIT API基础知识
* * *

即时编译器的目的是根据需要“在飞翔”编译代码，而不是像传统编译器那样提前将整个程序编译到磁盘上。为了支持这一目标，我们最初的基本JIT API将只有两个函数：

`Error addModule(std::Unique_ptr<Module>M)`：使给定的IR模块可供执行。
`Expted<JITEvaluatedSymbol>lookup()`：搜索指向已添加到JIT的符号(函数或变量)的指针。

此API的一个基本用例是从模块执行\‘main\’函数，如下所示：

```c++
JIT J;
J.addModule(buildModule());
auto *Main = (int(*)(int, char*[]))J.lookup("main").getAddress();
int Result = Main();
```

我们在这些教程中构建的API都是这个简单主题的变体。在此API背后，我们将改进JIT的实现，以添加对并发编译、优化和延迟编译的支持。最终，我们将扩展API本身，以允许将更高级别的程序表示(例如，AST)添加到JIT中。

## 万花筒JIT-万花筒JIT
* * *

在上一节中，我们描述了我们的API，现在我们研究它的一个简单实现：[使用LLVM实现语言](LangImpl01.html)教程中使用的KaleidoscopeJIT类[^1]。我们将使用该教程[第7章](LangImpl07.html)中的REPL代码为我们的JIT提供输入：每次用户输入表达式时，REPL都会将包含该表达式代码的新IR模块添加到JIT中。如果表达式是像`1+1`或`sin(X)`这样的顶级表达式，则REPL还将使用我们的JIT类的查找方法查找并执行该表达式的代码。在本教程后面的章节中，我们将修改REPL以启用与我们的JIT类的新交互，但现在我们将把这个设置视为理所当然，并将注意力集中在JIT本身的实现上。

我们的KaleidoscopeJIT类在KaleidoscopeJIT.h头中定义。在通常包括Guard和\#includes[^2](+-+-+)，之后，我们得到了我们类的定义：

```c++
#ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
#define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include <memory>

namespace llvm {
namespace orc {

class KaleidoscopeJIT {
private:
  ExecutionSession ES;
  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer CompileLayer;

  DataLayout DL;
  MangleAndInterner Mangle;
  ThreadSafeContext Ctx;

public:
  KaleidoscopeJIT(JITTargetMachineBuilder JTMB, DataLayout DL)
      : ObjectLayer(ES,
                    []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(ES, ObjectLayer, ConcurrentIRCompiler(std::move(JTMB))),
        DL(std::move(DL)), Mangle(ES, this->DL),
        Ctx(std::make_unique<LLVMContext>()) {
    ES.getMainJITDylib().setGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
  }
```

我们的类从六个成员变量开始：ExecutionSession成员`ES`，它为我们运行的JIT\‘d代码(包括字符串池、全局互斥和错误报告工具)提供上下文；RTDyldObjectLinkingLayer，`ObjectLayer`，它可以用来向我们的JIT添加对象文件(尽管我们不会直接使用它)；IRCompileLayer，`CompileLayer`，它可以用来添加LLVM ModLayer`

接下来，我们有一个类构造函数，它接受IRCompiler将使用的JITTargetMachineBuilder\`，以及我们将用来初始化DL成员的`DataLayout‘。构造函数首先初始化ObjectLayer。ObjectLayer需要一个对ExecutionSession的引用，以及一个将为添加的每个模块构建JIT内存管理器的函数对象(JIT内存管理器管理内存分配、内存权限和JIT代码的异常处理程序的注册)。为此，我们使用了一个返回SectionMemoryManager的lambda，这是一个现成的实用程序，它提供了本章所需的所有基本内存管理功能。接下来，我们初始化CompileLayer。CompileLayer需要三样东西：(1)对ExecutionSession的引用，(2)对对象层的引用，以及(3)用于执行从IR到目标文件的实际编译的编译器实例。我们使用现成的ConcurrentIRCompiler实用程序作为编译器，它是使用此构造函数的JITTargetMachineBuilder参数构造的。ConcurrentIRCompiler实用程序将根据编译需要使用JITTargetMachineBuilder构建llvm TargetMachines(不是线程安全的)。之后，我们分别使用输入的DataLayout、ExecutionSession和DL成员以及新的默认构造的LLVMContext初始化我们的支持成员：`DL`、`Mangler`和`Ctx`。既然我们的成员已经初始化，那么剩下的一件事就是调整我们将在其中存储代码的*JITDylib*的配置。我们希望修改此dylib，使其不仅包含添加到其中的符号，还包含REPL过程中的符号。为此，我们使用the`DynamicLibrarySearchGenerator::GetForCurrentProcess`方法附加一个`DynamicLibrarySearchGenerator`实例。

```c++
static Expected<std::unique_ptr<KaleidoscopeJIT>> Create() {
  auto JTMB = JITTargetMachineBuilder::detectHost();

  if (!JTMB)
    return JTMB.takeError();

  auto DL = JTMB->getDefaultDataLayoutForTarget();
  if (!DL)
    return DL.takeError();

  return std::make_unique<KaleidoscopeJIT>(std::move(*JTMB), std::move(*DL));
}

const DataLayout &getDataLayout() const { return DL; }

LLVMContext &getContext() { return *Ctx.getContext(); }
```

接下来，我们有一个命名构造函数`Create`，它将构建一个KaleidoscopeJIT实例，该实例被配置为为我们的宿主进程生成代码。这是通过以下方式实现的：首先使用类的DetectHost方法生成一个JITTargetMachineBuilder实例，然后使用该实例为目标流程生成数据布局。这些操作中的每一个都可能失败，因此每个操作都返回包装在期望值[^3]中的结果，我们必须检查该值是否有错误，然后才能继续。如果两个操作都成功，我们可以解开它们的结果(使用解引用操作符)，并将它们传递到函数最后一行上的KaleidoscopeJIT的构造函数中。

在命名构造函数之后，我们有`getDataLayout()`和`getContext()`方法。它们用于使由JIT(特别是LLVMContext)创建和管理的数据结构可供构建IR模块的REPL代码使用。

```c++
void addModule(std::unique_ptr<Module> M) {
  cantFail(CompileLayer.add(ES.getMainJITDylib(),
                            ThreadSafeModule(std::move(M), Ctx)));
}

Expected<JITEvaluatedSymbol> lookup(StringRef Name) {
  return ES.lookup({&ES.getMainJITDylib()}, Mangle(Name.str()));
}
```

现在我们来看第一个JIT API方法：addModule。此方法负责将IR添加到JIT并使其可供执行。在我们的JIT的这个初始实现中，我们将通过将模块添加到CompileLayer来使我们的模块“可供执行”，该CompileLayer将把模块存储在主JITDylib中。此过程将在JITDylib中为模块中的每个定义创建新的符号表条目，并将模块的编译推迟到查找其任何定义之后。请注意，这不是懒惰编译：只需引用一个定义，即使它从未使用过，也足以触发编译。在后面的章节中，我们将教我们的JIT将函数的编译推迟到它们被实际调用的时候。要添加我们的模块，我们必须首先将它包装在一个ThreadSafeModule实例中，该实例以线程友好的方式管理模块的LLVMContext(我们的CTX成员)的生存期。在我们的示例中，所有模块都将共享CTX成员，该成员将在JIT期间存在。一旦我们在后面的章节中切换到并发编译，我们将为每个模块使用一个新的上下文。

最后一个方法是`lookup`，它允许我们根据符号名称查找添加到JIT的函数和变量定义的地址。如上所述，查找将隐式触发尚未编译的任何符号的编译。我们的查找方法调用ExecutionSession::Lookup，传入要搜索的dylib列表(在本例中只是主dylib)和要搜索的符号名称，但有一点不同：我们必须先*压榨机*我们要搜索的符号的名称。ORC JIT组件像静电编译器和链接器一样在内部使用损坏的符号，而不是使用纯IR符号名称。这使得JIT代码可以轻松地与应用程序或共享库中的预编译代码进行互操作。损坏的类型将取决于DataLayout，而DataLayout又取决于目标平台。为了允许我们保持可移植性并基于未损坏的名称进行搜索，我们只需使用我们的“Mangle`”成员函数对象重新生成此损坏。

这将我们带到构建JIT的第1章的末尾。现在，您已经有了一个基本但功能齐全的JIT堆栈，您可以使用它来获取LLVM IR，并使其在JIT流程的上下文中可执行。在下一章中，我们将研究如何扩展这种JIT来生成更高质量的代码，并在此过程中更深入地研究ORC层的概念。

[下一步：扩展KaleidoscopeJIT](zh-BuildingAJIT2.md)

## 完整代码列表
* * *

下面是我们的运行示例的完整代码清单。为了建造这个
例如，使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

```
../../examples/Kaleidoscope/BuildingAJIT/Chapter1/KaleidoscopeJIT.h
```

[^1]：实际上我们使用的是KaleidoscopeJIT的精简版本，它做了一个简化的假设：符号不能重新定义。这将使重新定义REPL中的符号变得不可能，但将使我们的符号查找逻辑变得更简单。重新引入对符号重新定义的支持留给读者作为练习。(原始教程中使用的KaleidoscopeJIT.h将是有用的参考)。

[^2]:
    | File                     | Reason for inclusion                      |
    +==========================+===========================================+
    | > JITSymbol.h            | Defines the lookup result type            |
    |                          | JITEvaluatedSymbol                        |
    +--------------------------+-------------------------------------------+
    | > CompileUtils.h         | Provides the SimpleCompiler class.        |
    +--------------------------+-------------------------------------------+
    | > Core.h                 | Core utilities such as ExecutionSession   |
    |                          | and JITDylib.                             |
    +--------------------------+-------------------------------------------+
    | > ExecutionUtils.h       | Provides the                              |
    |                          | DynamicLibrarySearchGenerator class.      |
    +--------------------------+-------------------------------------------+
    | > IRCompileLayer.h       | Provides the IRCompileLayer class.        |
    +--------------------------+-------------------------------------------+
    | > JITTargetMachineBuilde | Provides the JITTargetMachineBuilder      |
    | r.h                      | class.                                    |
    +--------------------------+-------------------------------------------+
    | RTDyldObjectLinkingLayer | Provides the RTDyldObjectLinkingLayer     |
    | .h                       | class.                                    |
    +--------------------------+-------------------------------------------+
    | > SectionMemoryManager.h | Provides the SectionMemoryManager class.  |
    +--------------------------+-------------------------------------------+
    | > DataLayout.h           | Provides the DataLayout class.            |
    +--------------------------+-------------------------------------------+
    | > LLVMContext.h          | Provides the LLVMContext class.           |
    +--------------------------+-------------------------------------------+

[^3]：请参阅LLVM程序员手册中的ErrorHandling部分
(<https://llvm.org/docs/ProgrammersManual.html#error-handling>)
