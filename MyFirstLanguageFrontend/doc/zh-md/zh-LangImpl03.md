# 万花筒：LLVM IR的代码生成

：{.content local=“”}
**

## 第三章绪论

欢迎学习\“的第3章[使用实现语言
LLVM](index.html)\“教程。本章介绍如何转换
第2章中内置的[抽象语法树](LangImpl02.html)
LLVM IR。这将教给您一些关于LLVM是如何做事情的知识，如下所示
同时也展示了它的易用性。它有更多的工作要做
一个词法分析器和解析器，比它更能生成LLVM IR代码。：)

**请注意**：本章及更高版本的代码需要LLVM3.7或
后来。LLVM 3.6和更早版本将不能与其配合使用。另请注意，您需要
需要使用与您的LLVM版本匹配的本教程版本：
如果您使用的是官方LLVM发行版，请使用
随您的发行版一起提供或在[llvm.org发行版]上提供的文档
页面](https://llvm.org/releases/).

## 代码一代

为了生成LLVM IR，我们需要一些简单的设置。
首先，我们在每个AST中定义虚拟代码生成(Codegen)方法
班级：

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

codegen()方法表示为AST节点发出IR以及所有
它所依赖的对象，并且它们都返回一个LLVM值对象。
\“Value\”是用于表示\“[静电单项分配]的类
(SSA)](http://en.wikipedia.org/wiki/Static_single_assignment_form)
在LLVM中注册\“或\”SSA值\“。SSA最明显的方面
值是将它们的值作为相关指令进行计算
执行，并且在(并且如果)指令之前不会获得新值
重新执行。换句话说，没有办法\“更改\”SSA值。
欲了解更多信息，请阅读[静电单曲
Assignment](http://en.wikipedia.org/wiki/Static_single_assignment_form)
- 一旦你摸索了概念，这些概念就真的很自然了。

请注意，不是将虚拟方法添加到ExprAST类
层次结构，所以使用[访问者]也是有意义的
pattern](http://en.wikipedia.org/wiki/Visitor_pattern)或其他方式
来模拟这个。同样，本教程不会详述好的软件
工程实践：出于我们的目的，添加虚拟方法是
最简单的。

我们需要的第二件事是\“LogError\”方法，就像我们用于
解析器，它将用于报告在代码生成过程中发现的错误
(例如，使用未声明的参数)：

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

静电变量将在代码生成期间使用。`TheContext`
是一个不透明的对象，它拥有许多核心LLVM数据结构，例如
作为类型表和常量值表。我们不需要理解它
具体地说，我们只需要一个实例来传递给需要
它。

“Builder`”对象是一个帮助器对象，它使得生成
LLVM指令。的实例
[IRBuilder](https://llvm.org/doxygen/IRBuilder_8h_source.html)类
模板跟踪要插入指令的当前位置，并具有
方法来创建新指令。

`TheModule`是一个LLVM结构，它包含函数和全局
变量。在许多方面，它是LLVM IR的顶层结构
用于包含代码。它将拥有我们所有IR的内存
生成，这就是codegen()方法返回原始值\*的原因。
而不是UNIQUE_PTR\<VALUE>。

`NamedValues`映射跟踪在
当前作用域及其LLVM表示是什么。(换句话说，它
是代码的符号表)。在这种形式的万花筒中，唯一的
可以引用的是函数参数。因此，函数
参数在为其函数生成代码时将位于此映射中
身体。

有了这些基础知识，我们就可以开始讨论如何生成
每个表达式的代码。请注意，这假设`Builder`具有
已经设置为生成代码*变成*之类的东西。目前，我们假设
这项工作已经完成，我们将使用它来发出代码。

## 表达式代码生成

为表达式节点生成LLVM代码非常简单：更少
我们所有四个表达式节点的注释代码超过45行。
首先，我们要做的是数字文字：

```c++
Value *NumberExprAST::codegen() {
  return ConstantFP::get(TheContext, APFloat(Val));
}
```

在LLVM IR中，数字常量用`ConstantFP`表示
类，该类在内部保存`APFloat`中的数值
(`APFloat`具有保存浮点常量的能力
任意精度)。这段代码基本上只是创建并返回一个
`ConstantFP`。请注意，在LLVM IR中，所有常量都是唯一的
一起分享。因此，API使用
\“foo：：get(\.)\”习惯用法而不是\“new foo(..)\”或
\“foo：：create(..)\”。

```c++
Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value *V = NamedValues[Name];
  if (!V)
    LogErrorV("Unknown variable name");
  return V;
}
```

使用LLVM引用变量也非常简单。在简单的情况下
版本的万花筒，我们假设变量已经
在某处发出，并且其值可用。在实践中，唯一的
可以在`NamedValues`映射中的值是函数参数。这
代码简单地检查以查看指定的名称是否在地图中(如果没有，
未知变量被引用)，并返回其值。
在后续章节中，我们将添加对[循环归纳]的支持
符号表中的variables](LangImpl05.html#for-loop-expression)，以及
对于[本地variables](LangImpl07.html#user-defined-local-variables).

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

二元运算符开始变得更加有趣。这里的基本思想是
我们递归地发出表达式左侧的代码，
然后在右手边，然后我们计算二进制的结果
表达式。在此代码中，我们简单地切换操作码以创建
正确的LLVM指令。

在上面的示例中，LLVM构建器类开始显示其
价值。IRBuilder知道在哪里插入新创建的指令，
您只需指定要创建的指令(例如，使用
`CreateFAdd`)、要使用的操作数(此处为`L`和`R`)以及可选的
为生成的指令提供名称。

LLVM的一个优点是名称只是一个提示。例如,
如果上面的代码发出多个\“addtmp\”变量，LLVM将
自动为每个对象提供递增的唯一数字
后缀。指令的本地值名称纯粹是可选的，但它
使得读取红外线转储变得容易得多。

[LLVM instructions](../../LangRef.html#instruction-reference)是
受严格规则约束：例如，左运算符和右运算符
的[Add instruction](../../LangRef.html#add-instruction)必须具有
类型相同，并且加法的结果类型必须与操作数匹配
类型。因为万花筒中的所有值都是双精度的，所以这使得
非常简单的加法、减法和乘法代码。

另一方面，LLVM指定[fcmp
instruction](../../LangRef.html#fcmp-instruction)始终返回一个
\‘i1\’值(一位整数)。这样做的问题是
万花筒希望该值为0.0或1.0。为了得到
这些语义，我们将fcmp指令与[uitofp
instruction](../../LangRef.html#uitofp-to-instruction).本说明书
方法将其输入整数转换为浮点值。
作为无符号值输入。相反，如果我们使用[sitofp
instruction](../../LangRef.html#sitofp-to-instruction)，万花筒
根据输入值，运算符\‘\<\’将返回0.0和-1.0。

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

使用LLVM，函数调用的代码生成非常简单。
上面的代码最初在LLVM中查找函数名
模块的符号表。回想一下，LLVM模块是容器
它包含我们正在JIT\‘的函数。通过为每个函数赋予
与用户指定的名称相同，我们可以使用LLVM符号表
为我们解析函数名称。

一旦我们有了要调用的函数，我们就递归地编码生成每个参数
它将被传入，并创建一个LLVM[调用
instruction](../../LangRef.html#call-instruction).请注意，LLVM使用
默认情况下，本机C调用约定允许这些调用
还可以使用以下命令调用标准库函数，如\“sin\”和\“cos\”
不需要额外的努力。

这就结束了我们对四个基本表达式的处理
远在万花筒里。请随意进去，再加一些。例如,
通过浏览[LLVM Language Reference](../../LangRef.html)，您将
找到其他几个非常容易插入的有趣指令
进入我们的基本框架。

## 函数代码生成

原型和函数的代码生成必须处理许多
详细信息，这些细节使它们的代码不如表达式代码美观
这是一种新的概念，但也让我们能够说明一些重要的观点。第一,
让我们讨论一下原型的代码生成：它们都用于
函数体和外部函数声明。代码启动
使用：

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

此代码将大量功能打包到几行中。首先要注意的是，这个
函数返回\“函数\*\”，而不是\“值\*\”。因为一个
\“Prototype\”实际上谈论的是函数的外部接口
(不是由表达式计算的值)，则它有意义的是
当codegen\‘d时，返回它对应的LLVM函数。

对`FunctionType：：get`的调用创建的`FunctionType`应该
用于给定的原型。由于中的所有函数参数
万花筒是双精度类型的，第一行创建一个矢量
\“N\”LLVM双类型。然后，它使用`Functiontype：：get`方法
创建以\“N\”双精度值作为参数的函数类型，返回
结果是一个DOUBLE，并且不是vararg(false参数
表示这一点)。请注意，LLVM中的类型与常量一样是唯一的
是，所以你不是“新”的类型，你“得到”它。

上面的最后一行实际上创建了与以下内容相对应的IR函数
原型机。这还指示要使用的类型、链接和名称
作为要插入到哪个模块中。\“[外部
链接](../../LangRef.html#link)\“表示函数可以是
在当前模块外部定义和/或它可由
模块外部的功能。传入的名称是用户的名称
已指定：由于指定了\“`TheModule`\”，因此将注册此名称
在\“`TheModule`\”的符号表中。

```c++
// Set names for all arguments.
unsigned Idx = 0;
for (auto &Arg : F->args())
  Arg.setName(Args[Idx++]);

return F;
```

最后，我们根据以下内容设置每个函数参数的名称
原型中给出的名字。这一步并不严格
有必要，但保持名称一致会使IR更具可读性，
并允许后续代码直接引用其
名称，而不必在原型AST中查找它们。

在这一点上，我们有了一个没有主体的函数原型。这就是为什么
LLVM IR表示函数声明。中的外部语句
万花筒，这就是我们要去的地方。对于函数定义
但是，我们需要编码生成并附加一个函数体。

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

对于函数定义，我们首先搜索模块的符号
表，以获取此函数的现有版本(如果已经
是使用\‘extern\’语句创建的。如果模块：：getFunction
如果以前的版本不存在，则返回NULL，因此我们将从
原型机。在这两种情况下，我们都希望断言该函数是
在我们开始之前，你的身体是空的(也就是说还没有身体)。

```c++
// Create a new basic block to start insertion into.
BasicBlock *BB = BasicBlock::Create(TheContext, "entry", TheFunction);
Builder.SetInsertPoint(BB);

// Record the function arguments in the NamedValues map.
NamedValues.clear();
for (auto &Arg : TheFunction->args())
  NamedValues[Arg.getName()] = &Arg;
```

现在我们到了设置`Builder`的地方。第一行
创建新的[基本block](http://en.wikipedia.org/wiki/Basic_block)
(命名为\“entry\”)，插入到`TheFunction`中。第二行
然后告诉生成器应该将新指令插入到
新基础挡路收官。LLVM中的基本块是一个重要部分
定义[控制流]的函数
Graph](http://en.wikipedia.org/wiki/Control_flow_graph).既然我们没有
没有任何控制流，我们的函数在这里只包含一个挡路
重点。我们将在[第5章](LangImpl05.html)：)中解决此问题。

接下来，我们将函数参数添加到NamedValues映射(在第一个之后
清除它)，以便`VariableExprAST‘节点可以访问它们。

```c++
if (Value *RetVal = Body->codegen()) {
  // Finish off the function.
  Builder.CreateRet(RetVal);

  // Validate the generated code, checking for consistency.
  verifyFunction(*TheFunction);

  return TheFunction;
}
```

一旦设置了插入点并且NamedValues映射
填充后，我们调用`codegen()`方法作为
功能。如果没有发生错误，则会发出计算表达式的代码
添加到条目挡路中，并返回计算出的值。假设
没有错误，然后我们创建一个LLVM[ret
instruction](../../LangRef.html#ret-instruction)，，它完成了
功能。函数构建完成后，我们调用`verifyFunction`，它是
由LLVM提供。此函数对以下各项执行各种一致性检查
生成的代码，以确定我们的编译器是否正在执行所有操作
正确的。使用它很重要：它可以捕获很多错误。一旦
函数完成并验证后，我们返回它。

```c++
// Error reading body, remove function.
TheFunction->eraseFromParent();
return nullptr;
}
```

这里剩下的唯一部分就是错误情况的处理。为简单起见，
我们只需删除使用
`eraseFromParent`方法。这允许用户重新定义函数
他们之前输入的错误：如果我们不删除它，它会
住在符号表里，带着身体，防止将来重新定义。

不过，这段代码确实有一个错误：如果`FunctionAST：：codegen()`
方法查找现有的IR函数，它不验证其签名
与定义自己的原型进行比较。这意味着早先的
\‘extern\’声明将优先于函数
定义的签名，例如，这可能导致代码生成失败
如果函数参数的名称不同，则。这里有很多
修复这个错误的方法，看看你能想出什么！下面是一个测试案例：

    extern foo(a);     # ok, defines foo.
    def foo(b) b;      # Error: Unknown variable name. (decl using 'a' takes precedence).

## 驱动程序更改和结束思路

就目前而言，LLVM的代码生成并没有给我们带来多少好处，除了
我们可以看看漂亮的红外线通话。示例代码插入调用
要编码生成\“`HandleDefinition`\”、\“`HandleExtern`\”等
函数，然后转储LLVM IR。这样看起来很不错。
用于简单功能的LLVM IR。例如：

    ready> 4+5;
    Read top-level expression:
    define double @0() {
    entry:
      ret double 9.000000e+00
    }

请注意解析器如何将顶级表达式转换为匿名表达式
为我们服务。当我们添加[JIT]时，这将非常方便
support](LangImpl04.html#adding-a-jit-compiler)将在下一章中介绍。
还要注意的是，代码是按字面意思转录的，没有优化
除了由IRBuilder执行的简单常量折叠之外。我们
将[添加optimizations](LangImpl04.html#trivial-constant-folding)
请在下一章中明确说明。

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

这显示了一些简单的算术运算。请注意，这与我们的作品有着惊人的相似之处
我们用来创建指令的LLVM构建器调用。

    ready> def bar(a) foo(a, 4.0) + bar(31337);
    Read function definition:
    define double @bar(double %a) {
    entry:
      %calltmp = call double @foo(double %a, double 4.000000e+00)
      %calltmp1 = call double @bar(double 3.133700e+04)
      %addtmp = fadd double %calltmp, %calltmp1
      ret double %addtmp
    }

这显示了一些函数调用。请注意，此函数将需要很长时间
如果你叫它的话就该执行了。以后我们会增加有条件的
控制流以实际使递归有用：)。

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

这显示了libm\“cos\”函数的外部，以及对它的调用。

**：TODO
放弃Pygments\‘可怕的[llvm]{.title-ref}词法分析器。这完全是
由于第一行的原因，放弃突出显示此内容。
**：

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

当您退出当前演示时(通过在Linux上通过CTRL+D发送EOF或
在Windows上按Ctrl+Z和Enter)，则会转储整个模块的IR
已生成。在这里，您可以看到包含所有功能的全景图
相互参照。

这结束了万花筒教程的第三章。接下来，
我们将介绍如何[添加JIT代码生成器和优化器
支持](LangImpl04.html)，这样我们就可以真正开始运行代码了！

## 完整代码列表

以下是我们的运行示例的完整代码清单，增强了
LLVM代码生成器。因为它使用LLVM库，所以我们需要
把他们联系起来。为此，我们使用
[llvm-config](https://llvm.org/cmds/llvm-config.html)工具用于通知我们的
有关使用哪些选项的Makefile/命令行：

```bash
# Compile
clang++ -g -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy
# Run
./toy
```

以下是代码：

：{.writalinclude language=“c++”}
../examples/Kaleidoscope/Chapter3/toy.cpp
**：

[下一步：增加JIT和优化器支持](LangImpl04.html)
