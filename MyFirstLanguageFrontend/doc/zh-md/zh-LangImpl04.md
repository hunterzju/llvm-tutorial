# Kaleidoscope：添加JIT和优化器支持

## 第四章绪论

欢迎阅读“[使用LLVM实现语言](index.md)”教程的第4章。第1-3章描述了简单语言的实现，并添加了对生成LLVM IR的支持。本章介绍两种新技术：向语言添加优化器支持和添加JIT编译器支持。这些新增内容将演示如何为Kaleidoscope语言获得漂亮、高效的代码。

## 琐碎的常数折叠

我们在第3章中的演示是优雅的，并且易于扩展。不幸的是，它不能生成出色的代码。但是，在编译简单代码时，IRBuilder确实为我们提供了明显的优化：

```
    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            ret double %addtmp
    }
```

此代码不是通过解析输入构建的AST的文字转录。那就是：

```
    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 2.000000e+00, 1.000000e+00
            %addtmp1 = fadd double %addtmp, %x
            ret double %addtmp1
    }
```

特别是如上所述，常量折叠是一种非常常见且非常重要的优化：如此之多，以至于许多语言实现者在其AST表示中实现了常量折叠支持。

使用LLVM，您在AST中不需要这种支持。因为构建LLVM IR的所有调用都要通过LLVM IR生成器，所以当您调用它时，生成器本身会检查是否存在常量折叠机会。如果有，它只执行常量折叠并返回常量，而不是创建指令。

嗯，这很简单：)。实际上，我们建议在生成这样的代码时始终使用`IRBuilder`。它的使用没有“语法开销”(您不必在任何地方通过常量检查使编译器丑化)，并且它可以极大地减少在某些情况下生成的LLVM IR的数量(特别是对于带有宏预处理器的语言或使用大量常量的语言)。

另一方面，“IRBuilder”受到这样一个事实的限制，即它在构建时所有的分析都与代码内联。如果您举一个稍微复杂一点的示例：

```
    ready> def test(x) (1+2+x)*(x+(1+2));
    ready> Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            %addtmp1 = fadd double %x, 3.000000e+00
            %multmp = fmul double %addtmp, %addtmp1
            ret double %multmp
    }
```

在这种情况下，乘法的LHS和RHS是相同的值。我们非常希望看到它生成“`tmp=x+3；result=tmp*tmp；`”，而不是计算“`x+3`”两次。

不幸的是，任何数量的本地分析都无法检测和纠正这一点。这需要两个转换：表达式的重新关联(以使加法的词法相同)和公共子表达式消除(CSE)以删除冗余的加法指令。幸运的是，LLVM以“PASS”的形式提供了一系列可以使用的优化。

## LLVM优化通过

> 警告：由于已过渡到新的PassManager基础结构，因此本教程基于`llvm：：Legacy：：FunctionPassManager`(可以在[LegacyPassManager.h](https://llvm.org/doxygen/classllvm_1_1legacy_1_1FunctionPassManager.html)中找到).在完成PASS管理器过渡之前，应一直使用上述PassManager。

LLVM提供了许多优化通道，它们可以做很多不同的事情，有不同的权衡。与其他系统不同的是，LLVM不会错误地认为一组优化对所有语言和所有情况都是正确的。LLVM允许编译器实现者完全决定使用什么优化、以什么顺序和在什么情况下使用。

作为一个具体示例，LLVM支持两个“整个模块（whole module）”passes，这两个过程都能看到尽可能完整的代码体(通常是整个文件，但如果在链接时运行，这可能是整个程序的重要部分)。它还支持并包含“每个函数（per function）”passes，这些传递一次只在一个函数上操作，而不查看其他函数。有关pass及其运行方式的更多信息，请参阅[如何编写pass](https://llvm.org/docs/WritingAnLLVMPass.html)文档和[LLVM pass列表](https://llvm.org/docs/Passes.html)。

对于Kaleidoscope来说，我们目前正在动态（on the fly）生成函数，随着用户输入函数，一次生成一个函数。我们的目标不是在这种设置下获得终极优化体验，但我们也希望尽可能捕捉到简单快捷的东西。因此，我们将选择在用户键入函数时针对每个函数运行一些优化。如果我们想要创建一个“静态Kaleidoscope编译器”，我们将完全使用现在拥有的代码，只是我们将推迟运行优化器，直到解析完整个文件。

为了运行每个函数的优化，我们需要设置一个[FunctionPassManager](https://llvm.org/docs/WritingAnLLVMPass.html#what-passmanager-doesr)来保存和组织我们想要运行的LLVM优化。一旦我们有了这些，我们就可以添加一组要运行的优化。我们需要为每个要优化的模块创建一个新的FunctionPassManager，因此我们将编写一个函数来为我们创建和初始化模块和Pass管理器：

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

此代码初始化全局模块`TheModule`，以及`TheModule`附带的函数pass管理器`TheFPM`。一旦设置了PASS管理器，我们将使用一系列的“add”调用来添加一组LLVM PASS。

在本例中，我们选择添加四个优化过程。我们在这里选择的通道是一组非常标准的“清理”优化，对各种代码都很有用。我不会深入研究他们做了什么，但相信我，他们是一个很好的起点：)。

一旦设置了PassManager，我们就需要使用它。我们在构造新创建的函数之后(在`FunctionAST::codegen()`中)，在返回给客户端之前运行：

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

如您所见，这非常简单。`FunctionPassManager`就地优化和更新LLVM函数\*，改进(希望如此)它的主体。准备就绪后，我们可以再次尝试上面的测试：

```
    ready> def test(x) (1+2+x)*(x+(1+2));
    ready> Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double %x, 3.000000e+00
            %multmp = fmul double %addtmp, %addtmp
            ret double %multmp
    }
```

不出所料，我们现在得到了经过良好优化的代码，每次执行此函数时都会保存一条浮点加法指令。

LLVM提供了可在某些情况下使用的各种优化。虽然有一些[各种pass的文档](https://llvm.org/docs/Passes.html)，但不是很完整。另一个很好的想法来源是查看`Clang`开始运行的pass来学习pass。“`opt`”工具允许您从命令行尝试pass，这样您就可以看到它们是否有什么作用。

现在我们有了来自前端的合理代码，让我们来讨论一下如何执行它！

## 添加JIT编译器

LLVM IR中提供的代码可以应用多种工具。例如，您可以对其运行优化(如上所述)，可以将其转储为文本或二进制形式，可以将代码编译为某个目标的汇编文件(.s)，也可以对其进行JIT编译。LLVM IR表示的好处是它是编译器许多不同部分之间的“通用货币”。

在本节中，我们将在我们的解释器中添加JIT编译器支持。我们希望Kaleidoscope的基本思想是让用户像现在一样输入函数体，但立即计算他们键入的顶层表达式。例如，如果他们键入“1+2；”，我们应该计算并打印出3。如果他们定义了函数，他们应该能够从命令行调用该函数。

为此，我们首先准备环境为当前本机目标创建代码，并声明和初始化JIT。方法是调用一些`InitializeNativeTarget\*`函数，添加一个全局变量`TheJIT`，在`main`中初始化：

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

KaleidoscopeJIT类是专门为这些教程构建的简单JIT类，可在llvm-src/examples/Kaleidoscope/include/KaleidoscopeJIT.h.的LLVM源代码中找到。在后面的章节中，我们将看看它是如何工作的，并用新功能对其进行扩展，但现在我们将把它当作给定的。它的接口非常简单：`addModule`将LLVM IR模块添加到JIT中，使其函数可供执行；`removeModule`移除模块，释放与该模块中的代码关联的所有内存；`findSymbol`允许我们查找指向编译后代码的指针。

我们可以使用这个简单的API，并将解析顶级表达式的代码更改为如下所示：

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

如果解析和编码生成成功，则下一步是将包含顶级表达式的模块添加到JIT。我们通过调用addModule来实现这一点，addModule触发模块中所有函数的代码生成，并返回一个句柄，该句柄可用于稍后从JIT中删除模块。模块一旦添加到JIT中就不能再修改，所以我们还会通过调用`InitializeModuleAndPassManager()`打开一个新模块来存放后续代码。

将模块添加到JIT后，我们需要获取指向最终生成的代码的指针。为此，我们调用JIT的findSymbol方法，并传递顶层表达式函数的名称：`__anon_expr`。由于我们刚刚添加了此函数，因此我们断言findSymbol返回了一个结果。

接下来，我们通过对符号调用`getAddress()`来获取`__anon_expr`函数的内存地址。回想一下，我们将顶级表达式编译成一个不带参数并返回计算出的双精度值的自包含LLVM函数。因为LLVM JIT编译器匹配本机平台ABI，这意味着您只需将结果指针转换为该类型的函数指针并直接调用它。这意味着，JIT编译代码和静态链接到应用程序中的本机代码之间没有区别。

最后，因为我们不支持顶级表达式的重新求值，所以当我们完成释放相关内存时，我们会从JIT中删除该模块。但是，回想一下，我们在前面几行创建的模块(通过`InitializeModuleAndPassManager`)仍然处于打开状态，并等待添加新代码。

仅凭这两个变化，让我们看看Kaleidoscope现在是如何工作的！

```
    ready> 4+5;
    Read top-level expression:
    define double @0() {
    entry:
      ret double 9.000000e+00
    }

    Evaluated to 9.000000
```

嗯，这看起来基本上是有效的。函数的转储显示了我们为每个键入的顶级表达式合成的“总是返回双精度的无参数函数”。这演示了非常基本的功能，但是我们能做更多吗？

```
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
```

函数定义和调用也可以工作，但最后一行出现了非常错误的情况。函数调用看起来有效，但是出现报错，发生了什么事？正如您可能从API中猜到的那样，Module是JIT的分配单元，而testfunc是包含匿名表达式的同一模块的一部分。当我们从JIT中删除该模块以释放用于匿名表达式的内存时，我们同时删除了`testfunc`的定义。然后，当我们试图第二次调用testfunc时，JIT再也找不到它了。

解决此问题的最简单方法是将匿名表达式放在与剩余函数定义的不同的模块中。JIT将愉快地跨模块边界解决函数调用，只要每个被调用的函数都有一个原型，并且在调用之前被添加到JIT中。通过将匿名表达式放在不同的模块中，我们可以删除它，而不会影响剩余的函数。

事实上，我们将更进一步，将每个函数都放在它自己的模块中。这样做可以利用KaleidoscopeJIT的一个有用属性，这将使我们的环境更像REPL（Read–eval–print loop）：函数可以多次添加到JIT中(不同于每个函数都必须有唯一定义的模块)。当您在KaleidoscopeJIT中查找符号时，它将始终返回最新的定义：

```
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
```

要允许每个函数驻留在其自己的模块中，我们需要一种方法将以前的函数声明重新生成到我们打开的每个新模块中：

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

要实现这一点，我们将从添加一个新的全局`FunctionProtos`开始，它保存每个函数的最新原型。我们还将添加一个方便的方法`getFunction()`来替换对`TheModule->getFunction()`的调用。我们的便捷方法在`TheModule`中搜索现有的函数声明，如果没有找到，则退回到从FunctionProtos生成新的声明。在`CallExprAST：：codegen()`中，我们只需要替换对`TheModule->getFunction()`的调用。在`FunctionAST：：codegen()`中，我们需要先更新FunctionProtos映射，然后再调用`getFunction()`。完成此操作后，我们始终可以在当前模块中为任何先前声明的函数获取函数声明。

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

在HandleDefinition中，我们添加两行代码来将新定义的函数传递给JIT并打开一个新模块。在HandleExtern中，我们只需要添加一行将原型添加到FunctionProtos。

完成这些更改后，让我们再次尝试我们的REPL(这次我删除了匿名函数的转储，您现在应该明白了)：
```
    ready> def foo(x) x + 1;
    ready> foo(2);
    Evaluated to 3.000000

    ready> def foo(x) x + 2;
    ready> foo(2);
    Evaluated to 4.000000
```

它是有效的!

即使采用了这么简单的代码，我们收获了令人惊讶的强大能力 - 来看看下面示例：

```
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
```

哇，JIT怎么知道SIN和COS的？答案出奇的简单：KaleidoscopeJIT有一个简单明了的符号解析规则，它用来查找任何给定模块中没有的符号：首先，它搜索已经添加到JIT的所有模块(从最新到最旧)，以找到最新的定义。如果在JIT中找不到定义，它将退回到在Kaleidoscope进程本身上调用“`dlsym(”sin“)`”。因为“`sin`”是在JIT的地址空间中定义的，所以它只是给模块中的调用打了补丁，直接调用`sin`的libm版本。但在某些情况下，这甚至会更进一步：因为sin和cos是标准数学函数的名称，所以当使用常量调用函数时，Constant folder将直接计算函数调用的正确结果，就像上面的“`sin(1.0)`”一样。

在未来，我们将看到调整此符号解析规则能够被用来启用各种有用的功能，从安全性(限制可用于JIT代码的符号集)到基于符号名称的动态代码生成，甚至惰性编译（lazy compilation）。

符号解析规则的一个直接好处是，我们现在可以通过编写任意C++代码来实现来扩展语言操作符operation。例如，如果我们添加：

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

请注意，对于Windows，我们需要实际导出函数，因为动态符号加载器将使用GetProcAddress查找符号。

现在，我们可以使用以下命令向控制台生成简单的输出：“`extern putchard(X)；putchard(120)；`”，它在控制台上打印小写的‘x’(120是‘x’的ASCII代码)。类似的代码可用于在Kaleidoscope中实现文件I/O、控制台输入和许多其他功能。

这就完成了Kaleidoscope教程的JIT和优化器一章。在这一点上，我们可以编译一种非图灵完全的编程语言，并以用户驱动的方式对其进行优化和JIT编译。接下来，我们将研究[使用控制流构造扩展语言](LangImpl05.md)，解决一些有趣的LLVM IR问题。

## 完整代码列表

下面是我们的运行示例的完整代码清单，并使用LLVM JIT和优化器进行了增强。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

如果在Linux上编译，请确保还添加了“-rdynamic”选项。这确保在运行时正确解析外部函数。

以下是代码：
[https://github.com/llvm/llvm-project/blob/main/llvm/examples/Kaleidoscope/Chapter4/toy.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/examples/Kaleidoscope/Chapter4/toy.cpp)

[下一步：扩展语言：控制流](LangImpl05.md)
