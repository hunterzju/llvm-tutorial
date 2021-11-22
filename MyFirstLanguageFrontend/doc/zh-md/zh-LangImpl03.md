# Kaleidoscope：LLVM IR的代码生成

## 第三章绪论

欢迎阅读“[使用LLVM实现语言](zh-index.md)”教程的第3章。本章介绍如何将第2章中构建的[抽象语法树](zh-LangImpl02.md)转换为LLVM IR。这将教您一些关于LLVM是如何做事情的知识，并演示它的易用性。与生成LLVM IR代码相比，构建词法分析器和解析器的工作要多得多。：)

**请注意**：本章及以后的代码需要LLVM3.7或更高版本。LLVM 3.6和更早版本将不能与其配合使用。还要注意，您需要使用与您的LLVM发行版相匹配的本教程版本：如果您使用的是正式的LLVM发行版，请使用发行版中包含的文档版本或在[llvm.org发行版页面](https://llvm.org/releases/)中的版本。

## 代码生成设置

为了生成LLVM IR，我们需要一些简单的设置。首先，我们在每个AST类中定义虚拟代码生成(Codegen)方法：

```c++
/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() {}
  virtual Value *codegen() = 0;
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}
  virtual Value *codegen();
};
...
```

codegen()方法表示为该AST节点产生IR以及它所依赖的所有内容，并且它们都返回一个LLVM值对象。Value是用来表示LLVM中的“[静态单赋值(SSA)](http://en.wikipedia.org/wiki/Static_single_assignment_form)寄存器”或“SSA值”的类。SSA值最明显的方面是，它们的值是在相关指令执行时计算的，并且直到(如果)指令重新执行时才会获得新值。换句话说，没有办法“更改”SSA值。欲了解更多信息，请阅读[静态单赋值](http://en.wikipedia.org/wiki/Static_single_assignment_form) - 一旦你去研究，这些概念就真的很自然了。

请注意，除了将虚方法添加到ExprAST类层次结构中，使用[访问者模式](http://en.wikipedia.org/wiki/Visitor_pattern)或其他方式对此进行建模也是有意义的。重申一下，本教程不会详述好的软件工程实践：就我们的目的而言，添加虚拟方法是最简单的。

我们需要的第二件事是“LogError”方法，就像我们用于解析器一样，它将用于报告在代码生成过程中发现的错误(例如，使用未声明的参数)：

```c++
static LLVMContext TheContext;
static IRBuilder<> Builder(TheContext);
static std::unique_ptr<Module> TheModule;
static std::map<std::string, Value *> NamedValues;

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}
```

静态变量将在代码生成期间使用。`TheContext`是一个不透明的对象，拥有大量的LLVM核心数据结构，比如类型表和常量值表。我们不需要详细了解它，我们只需要一个实例来传递给需要它的API。

`Builder`对象是一个帮助对象，可以轻松生成LLVM指令。[IRBuilder](https://llvm.org/doxygen/IRBuilder_8h_source.html)类模板的实例跟踪当前插入指令的位置，并具有创建新指令的方法。

`TheModule`是包含函数和全局变量的LLVM结构。在许多方面，它是LLVM IR用来包含代码的顶层结构。它将拥有我们生成的所有IR的内存，这就是codegen()方法返回raw Value\*而不是unique_ptr\<Value\>的原因。

`NamedValues`映射跟踪在当前作用域中定义了哪些值，以及它们的LLVM表示是什么。(换句话说，它是代码的符号表)。在这种形式的Kaleidoscope中，唯一可以引用的是函数参数。因此，在为函数主体生成代码时，函数参数将在此映射中。

有了这些基础知识后，我们就可以开始讨论如何为每个表达式生成代码了。请注意，这假设`Builder`已设置为生成代码*变成*什么(译者注：即生成目标代码类型，比如x86的汇编还是ARM汇编)。现在，我们假设这已经完成了，我们将只使用它来发出代码。

## 表达式代码生成

为表达式节点生成LLVM代码非常简单：所有四个表达式节点加上注释代码不到45行。首先，我们要做的是数字常量：

```c++
Value *NumberExprAST::codegen() {
  return ConstantFP::get(TheContext, APFloat(Val));
}
```

在LLVM IR中，数值常量由`ConstantFP`类表示，该类在内部保存`APFloat`中的数值(`APFloat`可以保存任意精度的浮点常量)。这段代码基本上只是创建并返回一个`ConstantFP`。请注意，在LLVM IR中，所有常量都是唯一的，并且都是共享的。为此，API使用了“foo::get(\.)”习惯用法，而不是“new foo(..)”或“foo::create(..)”。

```c++
Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value *V = NamedValues[Name];
  if (!V)
    LogErrorV("Unknown variable name");
  return V;
}
```

使用LLVM引用变量也非常简单。在简单版本的Kaleidoscope中，我们假设变量已经在某个地方发出，并且它的值是可用的。实际上，`NamedValues`映射中唯一可以出现的值是函数参数。这段代码只是检查映射中是否有指定的名称(如果没有，则表示引用了一个未知变量)并返回该变量的值。在以后的章节中，我们将添加对符号表中的[循环指示变量(LOOP induction variables)](zh-LangImpl05.md#for-loop-expression)]和[本地变量(LOCAL variables)](zh-LangImpl07.md#user-defined-local-variables)的支持。

```c++
Value *BinaryExprAST::codegen() {
  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  if (!L || !R)
    return nullptr;

  switch (Op) {
  case '+':
    return Builder.CreateFAdd(L, R, "addtmp");
  case '-':
    return Builder.CreateFSub(L, R, "subtmp");
  case '*':
    return Builder.CreateFMul(L, R, "multmp");
  case '<':
    L = Builder.CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return Builder.CreateUIToFP(L, Type::getDoubleTy(TheContext),
                                "booltmp");
  default:
    return LogErrorV("invalid binary operator");
  }
}
```

二元运算符开始变得更加有趣。这里的基本思想是，我们递归地发出表达式左侧的代码，然后是右侧的代码，然后计算二元表达式的结果。在这段代码中，我们简单地替换操作码以创建正确的LLVM指令。

在上面的示例中，LLVM构建器类开始显示其价值。IRBuilder知道插入新创建的指令的位置，您只需指定要创建的指令(例如，使用`CreateFAdd`)、要使用的操作数(这里是`L`和`R`)，并可选择为生成的指令提供名称。

LLVM的一个优点是名称只是一个提示。例如，如果上面的代码发出多个“addtmp”变量，LLVM将自动为每个变量提供一个递增的唯一数字后缀。指令的本地值名称纯粹是可选的，但它使读取IR转储变得容易得多。

[LLVM instructions](https://llvm.org/docs/LangRef.html#instruction-reference)有严格的规则约束：例如，[Add instruction](https://llvm.org/docs/LangRef.html#add-instruction)的左运算符和右运算符必须具有相同的类型，并且Add的结果类型必须与操作数类型匹配。因为Kaleidoscope中的所有值都是双精度的，所以这使得加法、减法和乘法的代码非常简单。

另一方面，llvm指定[fcmp instruction](https://llvm.org/docs/LangRef.html#fcmp-instruction)总是返回‘i1’值(一位整数)。这样做的问题是Kaleidoscope希望该值是0.0或1.0。为了获得这些语义，我们将fcmp指令与[uitofp instruction](https://llvm.org/docs/LangRef.html#uitofp-to-instruction)组合在一起。此指令通过将输入视为无符号值，将其输入整数转换为浮点值。相反，如果我们使用[Sitofp instruction](https://llvm.org/docs/LangRef.html#sitofp-to-instruction)，则根据输入值的不同，Kaleidoscope‘\<’运算符将返回0.0和-1.0。

```c++
Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *CalleeF = TheModule->getFunction(Callee);
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->codegen());
    if (!ArgsV.back())
      return nullptr;
  }

  return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}
```

使用LLVM，函数调用的代码生成非常简单。上面的代码最初在LLVM模块的符号表中查找函数名。回想一下，LLVM模块是保存我们正在JIT的函数的容器。通过赋予每个函数与用户指定的名称相同的名称，我们可以使用LLVM符号表为我们解析函数名。

一旦我们有了要调用的函数，我们就递归地对要传入的每个参数进行编码，并创建一个llvm[调用instruction](https://llvm.org/docs/LangRef.html#call-instruction).请注意，默认情况下，LLVM使用原生C调用约定，允许这些调用还可以调用标准库函数(如“sin”和“cos”)，而不需要额外的工作。

到目前为止，我们对Kaleidoscope中的四个基本表达式的处理到此结束。请随意进去，再加一些。例如，通过浏览[LLVM Language Reference](https://llvm.org/docs/LangRef.html)，您会发现其他几个有趣的指令，它们非常容易插入到我们的基本框架中。

## 函数代码生成

原型和函数的代码生成必须处理许多细节，这些细节使它们的代码不如表达式代码生成美观，但允许我们说明一些重要的点。首先，让我们讨论一下原型的代码生成：它们既用于函数体，也用于外部函数声明。代码如下：

```c++
Function *PrototypeAST::codegen() {
  // Make the function type:  double(double,double) etc.
  std::vector<Type*> Doubles(Args.size(),
                             Type::getDoubleTy(TheContext));
  FunctionType *FT =
    FunctionType::get(Type::getDoubleTy(TheContext), Doubles, false);

  Function *F =
    Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());
```

此代码将大量功能打包到几行中。首先请注意，此函数返回”function\*”，而不是”value\*”。因为”Prototype”实际上谈论的是函数的外部接口(而不是表达式计算的值)，所以当codegen‘d时，它返回与之对应的LLVM函数是有意义的。

对`FunctionType::get`的调用创建了应该用于给定原型的`FunctionType`。因为Kaleidoscope中的所有函数参数都是DOUBLE类型，所以第一行创建了一个”N”LLVM DOUBLE类型的向量。然后使用`Functiontype::get`方法创建一个函数类型，该函数类型以”N”双精度值作为参数，返回一个双精度值作为结果，并且不是vararg(false参数表示这一点)。请注意，LLVM中的类型与常量一样是唯一的，因此您不会“新建”类型，而是“获取”它。

上面的最后一行实际上创建了与原型相对应的IR函数。这指示要使用的类型、链接和名称，以及要插入的模块。”[外部链接](https://llvm.org/docs/LangRef.html#linkage)”表示函数可以在当前模块外部定义和/或可以由模块外部的函数调用。传入的名称是用户指定的名称：由于指定了”`TheModule`”，所以该名称注册在”`TheModule`”的符号表中。

```c++
// Set names for all arguments.
unsigned Idx = 0;
for (auto &Arg : F->args())
  Arg.setName(Args[Idx++]);

return F;
```

最后，我们根据原型中给出的名称设置每个函数参数的名称。这一步并不是严格必要的，但是保持名称的一致性会使IR更具可读性，并且允许后续代码直接引用它们的名称的参数，而不必在原型AST中查找它们。

此时，我们有了一个没有函数体的函数原型。这就是LLVM IR表示函数声明的方式。对于Kaleidoscope中的外部（extern）语句，这就是我们需要做的。然而，对于函数定义，我们需要编码生成并附加一个函数体。

```c++
Function *FunctionAST::codegen() {
    // First, check for an existing function from a previous 'extern' declaration.
  Function *TheFunction = TheModule->getFunction(Proto->getName());

  if (!TheFunction)
    TheFunction = Proto->codegen();

  if (!TheFunction)
    return nullptr;

  if (!TheFunction->empty())
    return (Function*)LogErrorV("Function cannot be redefined.");
```

对于函数定义，我们首先在模块的符号表中搜索此函数的现有版本(如果已经使用‘extern’语句创建了一个版本)。如果Module::getFunction返回NULL，则不存在以前的版本，因此我们将从原型中编码生成一个。在任何一种情况下，我们都希望在开始之前断言函数为空(即还没有主体)。

```c++
// Create a new basic block to start insertion into.
BasicBlock *BB = BasicBlock::Create(TheContext, "entry", TheFunction);
Builder.SetInsertPoint(BB);

// Record the function arguments in the NamedValues map.
NamedValues.clear();
for (auto &Arg : TheFunction->args())
  NamedValues[Arg.getName()] = &Arg;
```

现在我们到了设置`Builder`的地方。第一行创建一个新的[basic block](http://en.wikipedia.org/wiki/Basic_block)”插入到`TheFunction`中。然后第二行告诉构建器，应该在新的`Basic block`的末尾插入新的指令。LLVM中的基本块是定义[控制流Graph](http://en.wikipedia.org/wiki/Control_flow_graph)的函数的重要部分。因为我们没有任何控制流，所以我们的函数此时将只包含一个block。我们将在[第5章](zh-LangImpl05.md)中解决这个问题：)。

接下来，我们将函数参数添加到NamedValues映射中(在其清除之后)，以便`VariableExprAST`节点可以访问它们。

```c++
if (Value *RetVal = Body->codegen()) {
  // Finish off the function.
  Builder.CreateRet(RetVal);

  // Validate the generated code, checking for consistency.
  verifyFunction(*TheFunction);

  return TheFunction;
}
```

一旦设置了插入点并填充了NamedValues映射，我们就会为函数的根表达式调用`codegen()`方法。如果没有发生错误，这将发出代码来计算表达式添加到entry block，并返回计算出的值。假设没有错误，我们会创建一个完成该功能的llvm [ret instruction](https://llvm.org/docs/LangRef.html#ret-instruction)。函数构建完成后，调用LLVM提供的`verifyFunction`。此函数对生成的代码执行各种一致性检查，以确定我们的编译器是否一切正常。使用它很重要：它可以捕获很多错误。一旦函数完成并经过验证，我们就会返回它。

```c++
// Error reading body, remove function.
TheFunction->eraseFromParent();
  return nullptr;
}
```

这里剩下的唯一部分就是错误情况的处理。为简单起见，我们只需使用`eraseFromParent`方法删除生成的函数即可处理此问题。这允许用户重新定义他们以前错误键入的函数：如果我们不删除它，它将与函数体一起存在于符号表中，防止将来重新定义。

不过，此代码确实有一个缺陷：如果`FunctionAST::codegen()`方法找到一个现有的IR函数，它不会根据定义自己的原型验证其签名。这意味着较早的‘extern’声明将优先于函数定义的签名，这可能会导致codegen失败，例如，如果函数参数命名不同。有很多方法可以修复此缺陷，看看您能想到什么！下面是一个测试用例：

```
    extern foo(a);     # ok, defines foo.
    def foo(b) b;      # Error: Unknown variable name. (decl using 'a' takes precedence).
```

## 驱动程序更改和结束思路

目前，LLVM的代码生成并没有给我们带来多少好处，除了我们可以查看漂亮的IR调用之外。示例代码将codegen的调用插入到”`HandleDefinition`”、”`HandleExtern`”等函数中，然后转储LLVM IR。这为查看简单函数的LLVM IR提供了一个很好的方法。例如：
```
    ready> 4+5;
    Read top-level expression:
    define double @0() {
    entry:
      ret double 9.000000e+00
    }
```

请注意解析器如何为我们将顶层表达式转换为匿名函数。当我们在下一章中添加[JIT support](zh-LangImpl04.md#adding-a-jit-compiler)]时，这将非常方便。还要注意的是，代码是按字面意思转录的，除了IRBuilder执行的简单常量折叠外，没有执行任何优化。我们将在下一章中[显式添加optimizations](zh-LangImpl04.md#trivial-constant-folding)。

```
    ready> def foo(a b) a*a + 2*a*b + b*b;
    Read function definition:
    define double @foo(double %a, double %b) {
    entry:
      %multmp = fmul double %a, %a
      %multmp1 = fmul double 2.000000e+00, %a
      %multmp2 = fmul double %multmp1, %b
      %addtmp = fadd double %multmp, %multmp2
      %multmp3 = fmul double %b, %b
      %addtmp4 = fadd double %addtmp, %multmp3
      ret double %addtmp4
    }
```

这显示了一些简单的算术运算。请注意，它与我们用来创建指令的LLVM构建器调用有惊人的相似之处。

```
    ready> def bar(a) foo(a, 4.0) + bar(31337);
    Read function definition:
    define double @bar(double %a) {
    entry:
      %calltmp = call double @foo(double %a, double 4.000000e+00)
      %calltmp1 = call double @bar(double 3.133700e+04)
      %addtmp = fadd double %calltmp, %calltmp1
      ret double %addtmp
    }
```

这显示了一些函数调用。请注意，如果调用此函数，将需要很长的执行时间。在将来，我们将添加条件控制流以使递归真正有用：)。

```
    ready> extern cos(x);
    Read extern:
    declare double @cos(double)

    ready> cos(1.234);
    Read top-level expression:
    define double @1() {
    entry:
      %calltmp = call double @cos(double 1.234000e+00)
      ret double %calltmp
    }
```

这显示了一个extern函数libm”cos”函数，以及对它的调用。

```
    ready> ^D
    ; ModuleID = 'my cool jit'

    define double @0() {
    entry:
      %addtmp = fadd double 4.000000e+00, 5.000000e+00
      ret double %addtmp
    }

    define double @foo(double %a, double %b) {
    entry:
      %multmp = fmul double %a, %a
      %multmp1 = fmul double 2.000000e+00, %a
      %multmp2 = fmul double %multmp1, %b
      %addtmp = fadd double %multmp, %multmp2
      %multmp3 = fmul double %b, %b
      %addtmp4 = fadd double %addtmp, %multmp3
      ret double %addtmp4
    }

    define double @bar(double %a) {
    entry:
      %calltmp = call double @foo(double %a, double 4.000000e+00)
      %calltmp1 = call double @bar(double 3.133700e+04)
      %addtmp = fadd double %calltmp, %calltmp1
      ret double %addtmp
    }

    declare double @cos(double)

    define double @1() {
    entry:
      %calltmp = call double @cos(double 1.234000e+00)
      ret double %calltmp
    }
```

当您退出当前演示(在Linux上通过CTRL+D发送EOF，在Windows上通过CTRL+Z并回车)时，它会转储生成的整个模块的IR。在这里，您可以看到所有函数相互引用的整体情况。

这结束了Kaleidoscope教程的第三章。接下来，我们将描述如何[添加JIT代码生成和优化器支持](zh-LangImpl04.md)，这样我们就可以真正开始运行代码了！

## 完整代码列表

下面是我们的运行示例的完整代码清单，并通过LLVM代码生成器进行了增强。因为它使用LLVM库，所以我们需要链接它们。为此，我们使用[llvm-config](https://llvm.org/cmds/llvm-config.html)工具通知生成文件/命令行要使用哪些选项：

```bash
# Compile
clang++ -g -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy
# Run
./toy
```

以下是代码：
[https://github.com/llvm/llvm-project/blob/main/llvm/examples/Kaleidoscope/Chapter3/toy.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/examples/Kaleidoscope/Chapter3/toy.cpp)

[下一步：增加JIT和优化器支持](zh-LangImpl04.md)

## 后记：心得体会
1. 静态单赋值：https://blog.csdn.net/qq_38876114/article/details/111461727
2. llvm::LLVMContext使用;
3. llvm::Module使用;
4. llvm::IRBuilder使用;