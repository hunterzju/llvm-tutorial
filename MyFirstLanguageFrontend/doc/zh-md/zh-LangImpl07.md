# 万花筒：扩展语言：可变变量

：{.content local=“”}
**

## 第七章简介

欢迎学习\“的第7章[使用实现语言
llvm](index.html)\“教程。在第1章到第6章中，我们构建了
非常受人尊敬，尽管很简单，[函数式编程
language](http://en.wikipedia.org/wiki/Functional_programming).在我们的
旅途中，我们学习了一些解析技术，如何构建和表示
AST，如何构建LLVM IR，以及如何将结果代码优化为
那就由JIT来编译吧。

虽然万花筒作为一种函数式语言很有趣，但事实是
它的功能性使得为其生成LLVM IR\“太容易\”。
特别是，函数式语言使得构建LLVM IR变得非常容易
直接在[SSA]中
form](http://en.wikipedia.org/wiki/Static_single_assignment_form).因为
LLVM要求输入代码为SSA格式，这是一个非常好的
属性，新手通常不清楚如何为
一种具有可变变量的命令式语言。

本章简短(且愉快)的总结是没有必要
为您的前端构建SSA表单：LLVM提供高度调优的
对此的支持经过了很好的测试，尽管它的工作方式有点
对某些人来说是意想不到的。

## 为什么这是一个很难解决的问题？

要了解可变变量为何会导致SSA中的复杂性，请执行以下操作
构造，请考虑这个极其简单的C示例：

```c
int G, H;
int test(_Bool Condition) {
  int X;
  if (Condition)
    X = G;
  else
    X = H;
  return X;
}
```

在本例中，我们有变量\“X\”，它的值取决于
在程序中执行的路径。因为有两种不同的可能
值，则在返回指令之前将PHI节点插入到
合并这两个值。我们在本例中需要的LLVM IR如下所示
如下所示：

```llvm
@G = weak global i32 0   ; type of @G is i32*
@H = weak global i32 0   ; type of @H is i32*

define i32 @test(i1 %Condition) {
entry:
  br i1 %Condition, label %cond_true, label %cond_false

cond_true:
  %X.0 = load i32, i32* @G
  br label %cond_next

cond_false:
  %X.1 = load i32, i32* @H
  br label %cond_next

cond_next:
  %X.2 = phi i32 [ %X.1, %cond_false ], [ %X.0, %cond_true ]
  ret i32 %X.2
}
```

在此示例中，来自G和H全局变量的负载为
在LLVM IR中显式，并且它们位于
IF语句(COND_TRUE/COND_FALSE)。为了合并传入的
值，则COND_NEXT挡路中的X.2φ节点选择正确的值
根据控制流的来源使用：如果控制流来自
从cond_false挡路中，X.2获得X.1的值。或者，如果
控制流来自cond_true，它获取X.0的值。其意图是
本章的内容不是对SSA表的详细说明。了解更多信息
有关信息，请参阅众多[在线]中的一个
references](http://en.wikipedia.org/wiki/Static_single_assignment_form).

本文的问题是\“在以下情况下，谁放置φ节点
降低对可变变量的赋值？\“。这里的问题是
llvm*需要*其IR为ssa形式：没有\“非ssa\”模式
为了它。然而，SSA构造需要非平凡的算法和
数据结构，所以对每个前端来说都是不方便和浪费的。
必须重现这一逻辑。

## LLVM中的内存

这里的\“诀窍\”是，虽然LLVM确实需要所有寄存器值
要以SSA形式存在，它不要求(或允许)内存对象
以SSA形式。在上面的示例中，请注意来自G和H的载荷为
直接访问G和H：它们不会重命名或版本化。这
与其他一些编译器系统不同，其他编译器系统确实会尝试对内存进行版本化
对象。在LLVM中，不是将内存数据流分析编码到
LLVM IR，它由[分析]处理
路径](../../WritingAnLLVMPass.html)，按需计算。

考虑到这一点，高级想法是我们想要创建一个堆栈
变量(它驻留在内存中，因为它在堆栈上)
函数中的可变对象。要利用这个把戏，我们需要
讨论LLVM如何表示堆栈变量。

在LLVM中，所有存储器访问都是用加载/存储指令显式进行的，
而且它被精心设计成不具有(或需要)\“地址\”
接线员。请注意\@G/\@H全局变量的类型是
实际上是\“I32\*\”，即使变量定义为\“I32\”。什么
这意味着\@G在全局数据中为I32定义了*空间
区域，但其*名字*实际上指的是该空间的地址。
堆栈变量的工作方式与此相同，不同之处在于不是声明堆栈变量
对于全局变量定义，它们是用[LLVM]声明的
阿洛卡instruction](../../LangRef.html#alloca-instruction)：

```llvm
define i32 @example() {
entry:
  %X = alloca i32           ; type of %X is i32*.
  ...
  %tmp = load i32, i32* %X  ; load the stack value %X from the stack.
  %tmp2 = add i32 %tmp, 1   ; increment it
  store i32 %tmp2, i32* %X  ; store it back
  ...
```

此代码显示如何声明和操作堆栈的示例
LLVM IR中的变量。使用Alloca分配的堆栈内存
指令是完全通用的：您可以传递堆栈槽的地址
对于函数，您可以将其存储在其他变量中，等等。
在上面，我们可以重写该示例以使用alloca技术来避免
使用PHI节点：

```llvm
@G = weak global i32 0   ; type of @G is i32*
@H = weak global i32 0   ; type of @H is i32*

define i32 @test(i1 %Condition) {
entry:
  %X = alloca i32           ; type of %X is i32*.
  br i1 %Condition, label %cond_true, label %cond_false

cond_true:
  %X.0 = load i32, i32* @G
  store i32 %X.0, i32* %X   ; Update X
  br label %cond_next

cond_false:
  %X.1 = load i32, i32* @H
  store i32 %X.1, i32* %X   ; Update X
  br label %cond_next

cond_next:
  %X.2 = load i32, i32* %X  ; Read X
  ret i32 %X.2
}
```

这样，我们就发现了一种处理任意可变变量的方法
变量，而根本不需要创建φ节点：

每个可变变量都成为堆栈分配。
每次读取变量都会成为堆栈中的加载。
变量的每次更新都会成为堆栈的存储。
获取变量的地址仅使用堆栈地址
直接去吧。

虽然这个解决方案解决了我们眼前的问题，但它引入了
另一个：我们现在显然引入了大量堆栈流量
对于非常简单和常见的操作，这是一个主要的性能问题。
对我们来说幸运的是，LLVM优化器具有高度调优的优化
名为\“mem2reg\”的传递处理此情况，将分配提升为
这将插入到SSA寄存器中，并根据需要插入φ节点。如果你跑
例如，通过此示例，您将获得：

```bash
$ llvm-as < example.ll | opt -mem2reg | llvm-dis
@G = weak global i32 0
@H = weak global i32 0

define i32 @test(i1 %Condition) {
entry:
  br i1 %Condition, label %cond_true, label %cond_false

cond_true:
  %X.0 = load i32, i32* @G
  br label %cond_next

cond_false:
  %X.1 = load i32, i32* @H
  br label %cond_next

cond_next:
  %X.01 = phi i32 [ %X.1, %cond_false ], [ %X.0, %cond_true ]
  ret i32 %X.01
}
```

mem2reg传递实现了标准的\“迭代优势边界\”
构造SSA表单的算法，并有许多优化
这加速了(非常常见的)退化案例。mem2reg优化
PASS是处理可变变量的答案，我们高度重视
建议您依赖它。请注意，mem2reg仅适用于
某些情况下的变量：

mem2reg是由alloca驱动的：它查找alloca以及它是否可以处理
他们，它提升了他们。它不适用于全局变量或
堆分配。
mem2reg仅在条目挡路中查找alloca指令
功能。在挡路的入口保证了阿洛卡是
只执行一次，这使得分析更简单。
mem2reg仅提升其用途为直接加载和
商店。如果堆栈对象的地址被传递给函数，
或者，如果涉及任何有趣的指针算法，则alloca将不会
被提拔。
mem2reg仅适用于[First]的分配
class](../../LangRef.html#first-class-type)值(如
指针、标量和向量)，并且仅当
分配为%1(或.ll文件中缺少)。mem2reg不支持
将结构或数组提升到寄存器。请注意，\“sroa\”
PASS功能更强大，可以升级结构、\“联合\”和
在许多情况下是数组。

对于大多数命令性要求，所有这些属性都很容易满足
语言，我们将在下面用万花筒进行说明。
你可能会问的问题是：我应该为我的
前端？如果我只做SSA建造不是更好吗
直接，避免使用mem2reg优化通道？简而言之，我们
强烈建议您使用此技术构建SSA表单，
除非有非常好的理由不这么做。使用这项技术
是：

- 经过验证和良好测试：Clang将此技术用于局部可变
变量。因此，LLVM最常见的客户端使用的是
来处理它们的大部分变量。你可以肯定虫子是
发现得快，修得早。
- 速度极快：mem2reg有许多特殊情况
在普通情况下快速，在完全通用的情况下也是如此。例如，它有
仅在单个挡路中使用的变量的快速路径，
只有一个赋值点的变量，很好的启发式方法
避免插入不需要的φ节点等。
- 生成调试信息所需的信息：[调试信息位于
LLVM](../../SourceLevelDebugging.html)依赖于拥有地址
公开的变量的属性，以便可以将调试信息附加到该变量。
这项技术与这种调试风格非常自然地吻合
信息。

如果没有其他问题，这将使您的前端和
运行，并且实现起来非常简单。让我们扩展一下万花筒
现在使用可变变量！

## 万花筒中的可变变量

现在我们知道了撞击要解决的问题是什么，让我们来看看是什么问题
这看起来像是在我们的小万花筒语言的上下文中。
我们将添加两个功能：

使用\‘=\’运算符变异变量的能力。
定义新变量的能力。

虽然第一个项目确实是关于这个的，但我们只有
用于传入参数和归纳变量的变量，以及
重新定义这些内容仅限于此：)。此外，还可以定义新的
无论您是否会发生变异，变量都是有用的
他们。下面是一个鼓舞人心的例子，它展示了我们如何使用这些：

    # Define ':' for sequencing: as a low-precedence operator that ignores operands
    # and just returns the RHS.
    def binary : 1 (x y) y;
    
    # Recursive fib, we could do this before.
    def fib(x)
      if (x < 3) then
        1
      else
        fib(x-1)+fib(x-2);
    
    # Iterative fib.
    def fibi(x)
      var a = 1, b = 1, c in
      (for i = 3, i < x in
         c = a + b :
         a = b :
         b = c) :
      b;
    
    # Call it.
    fibi(10);

为了改变变量，我们必须改变现有的变量
使用\“Alloca把戏\”。一旦我们有了它，我们就会添加我们的新
运算符，然后扩展万花筒以支持新的变量定义。

## 调整现有变量以进行突变

万花筒中的符号表在代码生成时由
\‘`NamedValues`\’映射。此映射当前跟踪LLVM
\“value\*\”，保存命名变量的双精度值。按顺序
为了支持突变，我们需要稍微改变这一点，以便
`NamedValues`保存相关变量的*内存位置*。
请注意，此更改是重构：它更改了
代码，但(本身)不会更改编译器的行为。全
这些变化都是在万花筒代码生成器中隔离的。

在万花筒开发的这一点上，它只支持变量
用于两件事：函数的传入参数和归纳
\‘for\’循环的变量。为保持一致性，我们将允许
这些变量以及其他用户定义的变量。这意味着
这两个都需要存储位置。

为了开始我们对万花筒的改造，我们将更改
NamedValues映射，以便它映射到AllocaInst\*而不是Value\*。一次
我们这样做，C++编译器会告诉我们需要哪些代码部分
要更新，请执行以下操作：

```c++
static std::map<std::string, AllocaInst*> NamedValues;
```

此外，由于我们将需要创建这些分配，因此我们将使用帮助器
确保在的条目挡路中创建分配的函数
功能：

```c++
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
static AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          const std::string &VarName) {
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                 TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(Type::getDoubleTy(TheContext), 0,
                           VarName.c_str());
}
```

这段看起来很滑稽的代码创建了一个IRBuilder对象，该对象指向
条目挡路的第一条指令(.Begin())。然后，它创建一个
具有预期名称的alloca，并返回它。因为中的所有值
万花筒都是双面的，不需要传入类型即可使用。

有了这一点，我们想要进行的第一个功能更改
属于变量引用。在我们的新方案中，变量位于
堆栈，所以生成对它们的引用的代码实际上需要生成
来自堆栈槽的加载：

```c++
Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value *V = NamedValues[Name];
  if (!V)
    return LogErrorV("Unknown variable name");

  // Load the value.
  return Builder.CreateLoad(V, Name.c_str());
}
```

如您所见，这非常简单。现在我们需要更新
定义设置Alloca的变量的东西。我们要开始了
使用`ForExprAST：：codegen()`(参见[完整代码清单](#Id1)了解
未删节的代码)：

```c++
Function *TheFunction = Builder.GetInsertBlock()->getParent();

// Create an alloca for the variable in the entry block.
AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

// Emit the start code first, without 'variable' in scope.
Value *StartVal = Start->codegen();
if (!StartVal)
  return nullptr;

// Store the value into the alloca.
Builder.CreateStore(StartVal, Alloca);
...

// Compute the end condition.
Value *EndCond = End->codegen();
if (!EndCond)
  return nullptr;

// Reload, increment, and restore the alloca.  This handles the case where
// the body of the loop mutates the variable.
Value *CurVar = Builder.CreateLoad(Alloca);
Value *NextVar = Builder.CreateFAdd(CurVar, StepVal, "nextvar");
Builder.CreateStore(NextVar, Alloca);
...
```

此代码实际上与[在我们允许可变变量之前]的代码相同
variables](LangImpl05.html#code-generation-for-the-for-loop).大的
不同之处在于，我们不再需要构建PHI节点，而是使用
根据需要加载/存储以访问变量。

为了支持可变参数变量，我们还需要为
他们。这方面的代码也非常简单：

```c++
Function *FunctionAST::codegen() {
  ...
  Builder.SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.
    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());

    // Store the initial value into the alloca.
    Builder.CreateStore(&Arg, Alloca);

    // Add arguments to variable symbol table.
    NamedValues[Arg.getName()] = Alloca;
  }

  if (Value *RetVal = Body->codegen()) {
    ...
```

对于每个参数，我们创建一个alloca，将输入值存储到
函数添加到alloca中，并将alloca注册为内存位置
为这场争论做准备。此方法由`FunctionAST：：codegen()`调用
就在它为该功能设置了入口挡路之后。

最后缺少的部分是添加mem2reg传递，它允许我们
再次获得好的编解码器：

```c++
// Promote allocas to registers.
TheFPM->add(createPromoteMemoryToRegisterPass());
// Do simple "peephole" optimizations and bit-twiddling optzns.
TheFPM->add(createInstructionCombiningPass());
// Reassociate expressions.
TheFPM->add(createReassociatePass());
...
```

方法之前和之后的代码是什么样子是很有趣的。
mem2reg优化运行。例如，这是之前/之后的代码
用于我们的递归fib函数。优化前：

```llvm
define double @fib(double %x) {
entry:
  %x1 = alloca double
  store double %x, double* %x1
  %x2 = load double, double* %x1
  %cmptmp = fcmp ult double %x2, 3.000000e+00
  %booltmp = uitofp i1 %cmptmp to double
  %ifcond = fcmp one double %booltmp, 0.000000e+00
  br i1 %ifcond, label %then, label %else

then:       ; preds = %entry
  br label %ifcont

else:       ; preds = %entry
  %x3 = load double, double* %x1
  %subtmp = fsub double %x3, 1.000000e+00
  %calltmp = call double @fib(double %subtmp)
  %x4 = load double, double* %x1
  %subtmp5 = fsub double %x4, 2.000000e+00
  %calltmp6 = call double @fib(double %subtmp5)
  %addtmp = fadd double %calltmp, %calltmp6
  br label %ifcont

ifcont:     ; preds = %else, %then
  %iftmp = phi double [ 1.000000e+00, %then ], [ %addtmp, %else ]
  ret double %iftmp
}
```

这里只有一个变量(x，输入参数)，但是您可以
仍然可以看到我们头脑极其简单的代码生成策略
使用。在条目挡路中，将创建一个分配，初始输入
值存储在其中。对该变量的每个引用都会进行重新加载
从堆栈中取出。还要注意，我们没有修改IF/THEN/ELSE
表达式，因此它仍然插入PHI节点。而我们可以做一个
对于它的alloca，实际上为它创建一个PHI节点更容易，所以我们
还是做好PHI就行了。

以下是mem2reg传递运行后的代码：

```llvm
define double @fib(double %x) {
entry:
  %cmptmp = fcmp ult double %x, 3.000000e+00
  %booltmp = uitofp i1 %cmptmp to double
  %ifcond = fcmp one double %booltmp, 0.000000e+00
  br i1 %ifcond, label %then, label %else

then:
  br label %ifcont

else:
  %subtmp = fsub double %x, 1.000000e+00
  %calltmp = call double @fib(double %subtmp)
  %subtmp5 = fsub double %x, 2.000000e+00
  %calltmp6 = call double @fib(double %subtmp5)
  %addtmp = fadd double %calltmp, %calltmp6
  br label %ifcont

ifcont:     ; preds = %else, %then
  %iftmp = phi double [ 1.000000e+00, %then ], [ %addtmp, %else ]
  ret double %iftmp
}
```

对于mem2reg来说，这是一个微不足道的案例，因为没有重新定义
变量。展示这个就是为了平息你的紧张情绪
插入这种明目张胆的低效：)。

优化器的睡觉运行后，我们得到：

```llvm
define double @fib(double %x) {
entry:
  %cmptmp = fcmp ult double %x, 3.000000e+00
  %booltmp = uitofp i1 %cmptmp to double
  %ifcond = fcmp ueq double %booltmp, 0.000000e+00
  br i1 %ifcond, label %else, label %ifcont

else:
  %subtmp = fsub double %x, 1.000000e+00
  %calltmp = call double @fib(double %subtmp)
  %subtmp5 = fsub double %x, 2.000000e+00
  %calltmp6 = call double @fib(double %subtmp5)
  %addtmp = fadd double %calltmp, %calltmp6
  ret double %addtmp

ifcont:
  ret double 1.000000e+00
}
```

在这里我们可以看到，simplifycfg传递决定克隆返回
说明插入到\‘Else\’挡路的末尾。这使得它可以
删除一些分支和PHI节点。

既然所有符号表引用都被更新为使用堆栈变量，
我们将添加赋值运算符。

## 新建赋值运算符

在我们当前的框架中，添加一个新的赋值操作符
很简单。我们将像解析任何其他二元运算符一样解析它，但处理
它在内部(而不是允许用户定义它)。第一个
步骤是设置优先级：

```c++
int main() {
  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
```

既然解析器知道二元运算符的优先级，它
负责所有的解析和AST生成。我们只需要
为赋值运算符实现codegen。这看起来像是：

```c++
Value *BinaryExprAST::codegen() {
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    // Assignment requires the LHS to be an identifier.
    VariableExprAST *LHSE = dynamic_cast<VariableExprAST*>(LHS.get());
    if (!LHSE)
      return LogErrorV("destination of '=' must be a variable");
```

与睡觉的二元运算符不同，我们的赋值运算符
没有遵循\“发射LHS，发射RHS，做计算\”的模型，因为
这样，它在其他二元运算符之前作为特殊情况处理
已经处理好了。另一件奇怪的事是，它要求LHS是一个
变量。具有\“(x+1)=expr\”-仅包含\“x\”这样的内容是无效的
=expr\“是允许的。

```c++
// Codegen the RHS.
Value *Val = RHS->codegen();
if (!Val)
  return nullptr;

// Look up the name.
Value *Variable = NamedValues[LHSE->getName()];
if (!Variable)
  return LogErrorV("Unknown variable name");

Builder.CreateStore(Val, Variable);
return Val;
}
...
```

一旦我们有了变量，赋值的代码生成就是
简单地说：我们发出分配的RHS，创建一个存储，然后
返回计算值。返回值允许链式
像\“X=(Y=Z)\”这样的任务。

现在我们有了赋值运算符，我们可以变异循环变量
和争论。例如，我们现在可以运行如下代码：

    # Function to print a double.
    extern printd(x);
    
    # Define ':' for sequencing: as a low-precedence operator that ignores operands
    # and just returns the RHS.
    def binary : 1 (x y) y;
    
    def test(x)
      printd(x) :
      x = 4 :
      printd(x);
    
    test(123);

运行时，此示例打印\“123\”，然后打印\“4\”，显示我们
真的改变了价值！好的，我们现在已经正式实施了
我们的目标：要实现这一点，总体上需要进行ssa建设。
箱子。但是，要真正有用，我们希望能够定义我们的
自己的局部变量，接下来让我们添加这个！

## 用户定义的局部变量

添加var/in就像我们对万花筒所做的任何其他扩展一样：
我们扩展了词法分析器、解析器、AST和代码生成器。这个
添加新的\‘var/in\’结构的第一步是扩展
莱克瑟。与前面一样，这非常简单，代码如下所示：

```c++
enum Token {
  ...
  // var definition
  tok_var = -13
...
}
...
static int gettok() {
...
    if (IdentifierStr == "in")
      return tok_in;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "var")
      return tok_var;
    return tok_identifier;
...
```

下一步是定义我们将构造的AST节点。为
var/in，如下所示：

```c++
/// VarExprAST - Expression class for var/in
class VarExprAST : public ExprAST {
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::unique_ptr<ExprAST> Body;

public:
  VarExprAST(std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
             std::unique_ptr<ExprAST> Body)
    : VarNames(std::move(VarNames)), Body(std::move(Body)) {}

  Value *codegen() override;
};
```

var/in允许一次定义所有名称列表，每个名称
可以有选择地具有初始值设定项值。因此，我们捕捉到了这一点
VarNames矢量中的信息。另外，var/in有一个身体，这个身体
允许访问由var/in定义的变量。

有了这些，我们就可以定义解析器部分了。我们首先要做的就是
DO是将其添加为主表达式：

```c++
/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
  switch (CurTok) {
  default:
    return LogError("unknown token when expecting an expression");
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr();
  case tok_for:
    return ParseForExpr();
  case tok_var:
    return ParseVarExpr();
  }
}
```

接下来，我们定义ParseVarExpr：

```c++
/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr() {
  getNextToken();  // eat the var.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("expected identifier after var");
```

此代码的第一部分解析标识符/expr对的列表
添加到本地的“VarNames`”向量中。

```c++
while (1) {
  std::string Name = IdentifierStr;
  getNextToken();  // eat identifier.

  // Read the optional initializer.
  std::unique_ptr<ExprAST> Init;
  if (CurTok == '=') {
    getNextToken(); // eat the '='.

    Init = ParseExpression();
    if (!Init) return nullptr;
  }

  VarNames.push_back(std::make_pair(Name, std::move(Init)));

  // End of var list, exit loop.
  if (CurTok != ',') break;
  getNextToken(); // eat the ','.

  if (CurTok != tok_identifier)
    return LogError("expected identifier list after var");
}
```

一旦解析完所有变量，我们就解析正文并创建
AST节点：

```c++
// At this point, we have to have 'in'.
if (CurTok != tok_in)
  return LogError("expected 'in' keyword after 'var'");
getNextToken();  // eat 'in'.

auto Body = ParseExpression();
if (!Body)
  return nullptr;

return std::make_unique<VarExprAST>(std::move(VarNames),
                                     std::move(Body));
}
```

现在我们可以解析和表示代码了，我们需要支持
它的LLVM IR的发射。此代码以以下代码开头：

```c++
Value *VarExprAST::codegen() {
  std::vector<AllocaInst *> OldBindings;

  Function *TheFunction = Builder.GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();
```

基本上，它循环遍历所有变量，一次安装一个变量
时间到了。对于我们放到符号表中的每个变量，我们记住
我们在OldBindings中替换的上一个值。

```c++
// Emit the initializer before adding the variable to scope, this prevents
// the initializer from referencing the variable itself, and permits stuff
// like this:
//  var a = 1 in
//    var a = a in ...   # refers to outer 'a'.
Value *InitVal;
if (Init) {
  InitVal = Init->codegen();
  if (!InitVal)
    return nullptr;
} else { // If not specified, use 0.0.
  InitVal = ConstantFP::get(TheContext, APFloat(0.0));
}

AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
Builder.CreateStore(InitVal, Alloca);

// Remember the old variable binding so that we can restore the binding when
// we unrecurse.
OldBindings.push_back(NamedValues[VarName]);

// Remember this binding.
NamedValues[VarName] = Alloca;
}
```

这里的注释比代码多。基本的想法是我们排放出
初始值设定项，创建分配项，然后更新符号表以
指向它。一旦所有变量都安装在符号表中，
我们计算var/in表达式的主体：

```c++
// Codegen the body, now that all vars are in scope.
Value *BodyVal = Body->codegen();
if (!BodyVal)
  return nullptr;
```

最后，在返回之前，我们恢复前面的变量绑定：

```c++
// Pop all our variables from scope.
for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
  NamedValues[VarNames[i].first] = OldBindings[i];

// Return the body computation.
return BodyVal;
}
```

所有这一切的最终结果是我们获得了正确的作用域变量
定义，我们甚至(微不足道地)允许对它们进行更改：)。

有了这个，我们就完成了我们开始要做的事情。我们美好的迭代谎言
简介中的示例编译并运行良好。mem2reg通行证
优化SSA寄存器中的所有堆栈变量，插入PHI
节点在需要的地方，并且我们的前端保持简单：无\“迭代
优势边界\“计算在视线范围内的任何地方。

## 完整代码列表

以下是我们的运行示例的完整代码清单，增强了
可变变量和var/in支持。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

：{.writalinclude language=“c++”}
../examples/Kaleidoscope/Chapter7/toy.cpp
**：

[下一步：编译为对象代码](LangImpl08.html)
