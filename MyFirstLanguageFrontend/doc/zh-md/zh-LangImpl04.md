# 万花筒：添加JIT和优化器支持

：{.content local=“”}
**

## 第四章绪论

欢迎学习\“的第4章[使用实现语言
LLVM](index.html)\“教程。第1-3章描述了实现
一种简单的语言，并添加了对生成LLVM IR的支持。这
本章介绍两种新技术：将优化器支持添加到
语言，并添加了JIT编译器支持。这些新增功能将
演示如何为万花筒获得漂亮、高效的代码
语言。

## 平凡常数折叠

我们在第3章中的演示是优雅的，并且易于扩展。
不幸的是，它不能生成出色的代码。IRBuilder，
但是，在编译简单代码时，确实给了我们明显的优化：

    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            ret double %addtmp
    }

此代码不是通过解析
输入。那就是：

    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 2.000000e+00, 1.000000e+00
            %addtmp1 = fadd double %addtmp, %x
            ret double %addtmp1
    }

如上所述，常量折叠尤其是一种非常常见的
非常重要的优化：如此之多以至于许多语言实现者
在它们的AST表示中实现常量折叠支持。

使用LLVM，您在AST中不需要此支持。
构建LLVM IR通过LLVM IR构建器，检查构建器本身
看看你打电话的时候是否有持续的折叠机会。如果
因此，它只是进行常量折叠并返回常量，而不是
创建指令。

嗯，这很简单：)。在实践中，我们建议始终使用
生成这样的代码时使用`IRBuilder`。它没有\“句法
开销\“用于它的使用(您不必用以下命令丑化您的编译器
无处不在的持续检查)，它可以极大地减少
在某些情况下生成的LLVM IR(特别是对于具有
宏预处理器或使用大量常量的)。

另一方面，“IRBuilder”受到这样的事实的限制
它的所有分析都与构建时的代码保持一致。如果你拿一个
稍微复杂一点的示例：

    ready> def test(x) (1+2+x)*(x+(1+2));
    ready> Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            %addtmp1 = fadd double %x, 3.000000e+00
            %multmp = fmul double %addtmp, %addtmp1
            ret double %multmp
    }

在这种情况下，乘法的LHS和RHS是相同的值。
我们真的很希望看到这件事
\“`tmp=x+3；result=tmp*tmp；`\”，而不是计算\“`x+3`\”两次。

不幸的是，任何数量的本地分析都无法检测到
改正这个。这需要两个转换：重新关联
表达式(使Add的词汇相同)和Common
子表达式消除(CSE)删除冗余加法指令。
幸运的是，LLVM提供了广泛的优化，您可以
以\“通行证\”的形式使用。

## LLVM优化通过

**：警告
：标题
警告
**：

由于过渡到新的PassManager基础架构，
教程基于`llvm：：Legacy：：FunctionPassManager`，它可以是
在以下位置找到
[LegacyPassManager.h](https://llvm.org/doxygen/classllvm_1_1legacy_1_1FunctionPassManager.html).
出于本教程的目的，应该使用上面的内容，直到
过程管理器过渡已完成。
**：

LLVM提供了许多优化过程，它们执行许多不同类型的
并且有不同的权衡。与其他系统不同，LLVM不
坚持错误的观念，认为一组优化适用于
所有语言，适用于所有情况。LLVM允许编译器实现器
要完全决定要使用哪些优化，在
秩序，以及在什么情况下。

作为一个具体示例，LLVM支持两个\“整个模块\”过程，即
查看尽可能大的代码体(通常是整个文件，
但如果在链接时运行，这可能是整体的很大一部分
程序)。它还支持并包括\“按函数\”传递，该传递
一次只操作一个函数，而不查看其他函数
功能。有关通道及其运行方式的更多信息，请参见
[如何写PASS](../../WritingAnLLVMPass.html)文档和
[LLVM通道列表](../../Passes.html)。

对于万花筒，我们目前正在飞翔上生成函数，一个
一次，当用户键入它们时。我们不是在为
在此设置中的终极优化体验，但我们也希望
在可能的情况下抓住容易和快速的东西。因此，我们将选择
要在用户键入函数时运行一些针对每个函数的优化，请执行以下操作
在……里面。如果我们想做一个\“静电万花筒编译器\”，我们会
完全使用我们现在拥有的代码，只是我们会推迟运行
优化器，直到解析完整个文件。

为了进行每个函数的优化，我们需要设置一个
[FunctionPassManager](../../WritingAnLLVMPass.html#what-passmanager-doesr)
来保存和组织我们要运行的LLVM优化。一旦我们
有了这些，我们就可以添加一组优化来运行。我们需要一个新的
要优化的每个模块的FunctionPassManager，因此我们将
编写一个函数来创建和初始化模块和传递
我们的经理：

```c++
void InitializeModuleAndPassManager(void) {
  // Open a new module.
  TheModule = std::make_unique<Module>("my cool jit", TheContext);

  // Create a new pass manager attached to it.
  TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());

  // Do simple "peephole" optimizations and bit-twiddling optzns.
  TheFPM->add(createInstructionCombiningPass());
  // Reassociate expressions.
  TheFPM->add(createReassociatePass());
  // Eliminate Common SubExpressions.
  TheFPM->add(createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  TheFPM->add(createCFGSimplificationPass());

  TheFPM->doInitialization();
}
```

此代码初始化全局模块`TheModule`和函数
传入管理器`TheFPM`，与`TheModule`关联。一旦通行证
管理器设置后，我们使用一系列\“add\”调用来添加一组
LLVM通过。

在本例中，我们选择添加四个优化过程。我们的通行证
选择此处是一组非常标准的\“清理\”优化
对于各种代码都很有用。我不会深入研究他们的所作所为
但是，相信我，它们是一个很好的起点：)。

一旦设置了PassManager，我们就需要使用它。我们是通过以下方式做到这一点的
在构造新创建的函数(在中)之后运行它
`FunctionAST：：codegen()`)，但在返回给客户端之前：

```c++
if (Value *RetVal = Body->codegen()) {
  // Finish off the function.
  Builder.CreateRet(RetVal);

  // Validate the generated code, checking for consistency.
  verifyFunction(*TheFunction);

  // Optimize the function.
  TheFPM->run(*TheFunction);

  return TheFunction;
}
```

如您所见，这非常简单。这个
`FunctionPassManager`优化并更新中的LLVM函数\*
地方，改善(希望)它的身体。有了这个，我们可以试一试
我们的测试再次如上：

    ready> def test(x) (1+2+x)*(x+(1+2));
    ready> Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double %x, 3.000000e+00
            %multmp = fmul double %addtmp, %addtmp
            ret double %multmp
    }

不出所料，我们现在得到了经过良好优化的代码，从而节省了一个浮点数
每次执行此函数时的点数相加指令。

LLVM提供了多种可用于
在某些情况下。一些[关于各种
PASS](../../Passes.html)可用，但不是非常完整。
另一个好的想法来源可以来自于查看传球
“叮当”跑着开始。您可以使用\“`opt`\”工具进行实验
使用来自命令行的传递，所以您可以看到它们是否做了什么。

现在我们有了来自前端的合理代码，让我们
谈到执行它！

## 添加JIT编译器

LLVM IR中可用的代码可以有多种工具
适用于它。例如，您可以对其运行优化(如我们所做的
如上所述)，您可以将其转储为文本或二进制形式，您可以编译
某些目标的程序集文件(.s)的代码，或者您可以JIT
编译它。LLVM IR表示的好处是它
是编译器的许多不同部分之间的\“通用货币\”。

在本节中，我们将向解释器添加JIT编译器支持。
我们想要的万花筒的基本思想是让用户输入
函数体，但立即计算顶级
他们键入的表达式。例如，如果他们键入\“1+2；\”，我们
应该求值并打印出来3。如果他们定义了一个函数，他们应该
能够从命令行调用它。

为了做到这一点，我们首先准备好为其创建代码的环境
当前本机目标，并声明和初始化JIT。这是
通过调用一些`InitializeNativeTarget\*`函数并添加一个
全局变量`TheJIT`，并在`main`中初始化：

```c++
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
...
int main() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = std::make_unique<KaleidoscopeJIT>();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
```

我们还需要设置JIT的数据布局：

```c++
void InitializeModuleAndPassManager(void) {
  // Open a new module.
  TheModule = std::make_unique<Module>("my cool jit", TheContext);
  TheModule->setDataLayout(TheJIT->getTargetMachine().createDataLayout());

  // Create a new pass manager attached to it.
  TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());
  ...
```

KaleidoscopeJIT类是专门为这些组件构建的简单JIT类
LLVM源代码内的教程，网址为
llvm-src/examples/Kaleidoscope/include/KaleidoscopeJIT.h.在以后
章节我们将了解它的工作原理，并使用新功能对其进行扩展，
但现在我们会把它当成是给与的。它的接口非常简单：
`addModule`将LLVM IR模块添加到JIT，使其函数
可供执行；`removeModule`删除模块，释放所有
与该模块中的代码相关联的内存；并且`findSymbol`允许
来查找指向编译代码的指针。

我们可以使用这个简单的API并更改顶层解析的代码
表达式如下所示：

```c++
static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (FnAST->codegen()) {

      // JIT the module containing the anonymous expression, keeping a handle so
      // we can free it later.
      auto H = TheJIT->addModule(std::move(TheModule));
      InitializeModuleAndPassManager();

      // Search the JIT for the __anon_expr symbol.
      auto ExprSymbol = TheJIT->findSymbol("__anon_expr");
      assert(ExprSymbol && "Function not found");

      // Get the symbol's address and cast it to the right type (takes no
      // arguments, returns a double) so we can call it as a native function.
      double (*FP)() = (double (*)())(intptr_t)ExprSymbol.getAddress();
      fprintf(stderr, "Evaluated to %f\n", FP());

      // Delete the anonymous expression module from the JIT.
      TheJIT->removeModule(H);
    }
```

如果解析和代码生成成功，则下一步是添加模块
包含JIT的顶级表达式的。我们是通过打电话来做到这一点的
addModule，它触发
模块，并返回一个句柄，该句柄可用于从
晚些时候的JIT。一旦将模块添加到JIT，它就不能
因此，我们还打开一个新模块来保存后续代码
调用`InitializeModuleAndPassManager()`。

将模块添加到JIT后，需要获取指向
最终生成的代码。为此，我们调用JIT的findSymbol
方法，并传递顶级表达式函数的名称：
`__anon_expr`。由于我们刚刚添加了此函数，因此我们断言
findSymbol返回结果。

接下来，我们通过以下方式获得`__anon_expr`函数的内存地址
在符号上调用`getAddress()`。回想一下，我们在顶层编译
表达式转换为不带参数的自含式LLVM函数
并返回计算出的双精度。因为LLVM JIT编译器匹配
本机平台ABI，这意味着您可以只强制转换结果
指向该类型的函数指针的指针，并直接调用它。这
也就是说，JIT编译代码和本机代码没有区别
静态链接到应用程序的机器码。

最后，由于我们不支持重新计算顶级表达式，
当我们完成释放模块时，我们将模块从JIT中移除
关联内存。不过，回想一下，我们创建的几个模块
早些时候的线路(通过`InitializeModuleAndPassManager`)仍然开放，并且
正在等待添加新代码。

仅凭这两个变化，让我们看看万花筒现在是如何工作的！

    ready> 4+5;
    Read top-level expression:
    define double @0() {
    entry:
      ret double 9.000000e+00
    }
    
    Evaluated to 9.000000

嗯，这看起来基本上是有效的。函数的转储
显示了\“始终返回DOUBLE的无参数函数\”
为键入的每个顶级表达式合成。这
演示了非常基本的功能，但我们还能做更多吗？

    ready> def testfunc(x y) x + y*2;
    Read function definition:
    define double @testfunc(double %x, double %y) {
    entry:
      %multmp = fmul double %y, 2.000000e+00
      %addtmp = fadd double %multmp, %x
      ret double %addtmp
    }
    
    ready> testfunc(4, 10);
    Read top-level expression:
    define double @1() {
    entry:
      %calltmp = call double @testfunc(double 4.000000e+00, double 1.000000e+01)
      ret double %calltmp
    }
    
    Evaluated to 24.000000
    
    ready> testfunc(5, 10);
    ready> LLVM ERROR: Program used external function 'testfunc' which could not be resolved!

函数定义和调用也可以工作，但出现了非常错误的情况
在最后一条线上。电话看起来有效，发生了什么事？尽管你可能会
从API猜测模块是JIT的分配单位，
而testfunc是同一模块的一部分，该模块包含匿名
表达式。当我们从JIT中删除该模块以释放内存时
对于匿名表达式，我们删除了`testfunc`的定义
随之而来的是。然后，当我们尝试第二次调用testfunc时，
JIT再也找不到它了。

解决此问题的最简单方法是将匿名表达式放在
将模块与睡觉的函数定义分开。JIT将会
跨模块边界愉快地解决函数调用，只要每个
被调用的函数中有一个原型，并且在此之前被添加到JIT中
它被称为。通过将匿名表达式放在不同的模块中
我们可以在不影响睡觉功能的情况下删除它。

事实上，我们要走得更远，把每一个功能都放在它的
拥有自己的模块。这样做可以让我们利用
KaleidoscopeJIT将使我们的环境更像REPL：函数
可以多次添加到JIT中(与模块不同，在JIT中
函数必须有唯一的定义)。当您在中查找符号时
KaleidoscopeJIT它将始终返回最新的定义：

    ready> def foo(x) x + 1;
    Read function definition:
    define double @foo(double %x) {
    entry:
      %addtmp = fadd double %x, 1.000000e+00
      ret double %addtmp
    }
    
    ready> foo(2);
    Evaluated to 3.000000
    
    ready> def foo(x) x + 2;
    define double @foo(double %x) {
    entry:
      %addtmp = fadd double %x, 2.000000e+00
      ret double %addtmp
    }
    
    ready> foo(2);
    Evaluated to 4.000000

为了让每个函数都存在于它自己的模块中，我们需要一种方法来
将以前的函数声明重新生成到我们打开的每个新模块中：

```c++
static std::unique_ptr<KaleidoscopeJIT> TheJIT;

...

Function *getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto *F = TheModule->getFunction(Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  // If no existing prototype exists, return null.
  return nullptr;
}

...

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *CalleeF = getFunction(Callee);

...

Function *FunctionAST::codegen() {
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function *TheFunction = getFunction(P.getName());
  if (!TheFunction)
    return nullptr;
```

要实现这一点，我们将首先添加一个新的全局变量`FunctionProtos`，
它保存每个函数的最新原型。我们还将添加
一个方便的方法，`getFunction()`，用于替换对
`TheModule->getFunction()`。我们的便捷方法搜索`TheModule`
对于现有函数声明，回退到生成新的
如果找不到FunctionProtos，则从FunctionProtos声明。
`CallExprAST：：codegen()`我们只需要将对
`TheModule->getFunction()`。在`FunctionAST：：codegen()`中，我们需要
首先更新FunctionProtos映射，然后调用`getFunction()`。使用
这样做之后，我们始终可以在当前
任何先前声明的函数的模块。

我们还需要更新HandleDefinition和HandleExtern：

```c++
static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read function definition:");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      TheJIT->addModule(std::move(TheModule));
      InitializeModuleAndPassManager();
    }
  } else {
    // Skip token for error recovery.
     getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern: ");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}
```

在HandleDefinition中，我们添加两行来传输新定义的
函数添加到JIT并打开一个新模块。在HandleExtern中，我们只需要
若要添加一行以将原型添加到FunctionProtos，请执行以下操作。

完成这些更改后，让我们再次尝试我们的REPL(我删除了转储
对于这次的匿名函数，您现在应该已经了解了：)
：

    ready> def foo(x) x + 1;
    ready> foo(2);
    Evaluated to 3.000000
    
    ready> def foo(x) x + 2;
    ready> foo(2);
    Evaluated to 4.000000

它起作用了!

即使使用这个简单的代码，我们也会得到一些令人惊讶的强大功能
功能-查看以下内容：

    ready> extern sin(x);
    Read extern:
    declare double @sin(double)
    
    ready> extern cos(x);
    Read extern:
    declare double @cos(double)
    
    ready> sin(1.0);
    Read top-level expression:
    define double @2() {
    entry:
      ret double 0x3FEAED548F090CEE
    }
    
    Evaluated to 0.841471
    
    ready> def foo(x) sin(x)*sin(x) + cos(x)*cos(x);
    Read function definition:
    define double @foo(double %x) {
    entry:
      %calltmp = call double @sin(double %x)
      %multmp = fmul double %calltmp, %calltmp
      %calltmp2 = call double @cos(double %x)
      %multmp4 = fmul double %calltmp2, %calltmp2
      %addtmp = fadd double %multmp, %multmp4
      ret double %addtmp
    }
    
    ready> foo(4.0);
    Read top-level expression:
    define double @3() {
    entry:
      %calltmp = call double @foo(double 4.000000e+00)
      ret double %calltmp
    }
    
    Evaluated to 1.000000

哇，JIT怎么知道罪孽和COS的？答案是
出奇的简单：KaleidoscopeJIT有一个简单的符号
用于查找中不可用的符号的解析规则
任何给定的模块：首先，它搜索所有已经
已添加到JIT中，从最新到最旧，以查找
最新定义。如果在JIT中找不到定义，它将失败
返回到对万花筒进程本身调用\“`dlsym(”sin“)`\”。
因为\“`sin`\”是在JIT的地址空间中定义的，所以它只
修补模块中的调用，以调用`sin`的libm版本
直接去吧。但在某些情况下，这甚至走得更远：就像罪和罪一样
标准数学函数的名称，常量文件夹将直接
使用调用函数时，对正确结果的函数调用进行评估
常量，如上面的\“`sin(1.0)`\”。

将来，我们将看看如何调整此符号解析规则
用于启用各种有用的功能，从安全(限制
可用于JIT代码的符号集)，用于动态代码生成
基于符号名称，甚至懒于编译。

符号解析规则的一个直接好处是，我们现在可以
通过编写任意C++代码来扩展该语言以实现
运营部。例如，如果我们添加：

```c++
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double X) {
  fputc((char)X, stderr);
  return 0;
}
```

请注意，对于Windows，我们需要实际导出函数，因为
动态符号加载器将使用GetProcAddress查找符号。

现在，我们可以使用如下内容向控制台生成简单的输出：
\“`外部putchard(X)；putchard(120)；`\”，打印小写的\‘x\’
在控制台上(120是\‘x\’的ASCII代码)。类似的代码可能是
用于实施文件I/O、控制台输入和许多其他功能
在万花筒里。

这就完成了万花筒的JIT和优化器一章
教程。在这一点上，我们可以编译一个非图灵完成
编程语言，以用户驱动的方式进行优化和JIT编译。
接下来，我们将研究[使用控制流扩展语言
构造](LangImpl05.html)，处理一些有趣的LLVM IR问题
一路走来。

## 完整代码列表

以下是我们的运行示例的完整代码清单，增强了
LLVM JIT和优化器。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

如果您在Linux上编译此程序，请确保添加\“-rdynamic\”
也可以选择。这可确保解析外部函数
在运行时正确执行。

以下是代码：

：{.writalinclude language=“c++”}
../examples/Kaleidoscope/Chapter4/toy.cpp
**：

[下一步：扩展语言：控制流](LangImpl05.html)
