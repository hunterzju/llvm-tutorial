# Kaleidoscope：扩展语言：可变变量

## 第七章简介

欢迎阅读“[使用LLVM实现语言](index.html)”教程的第7章。在第1章到第6章中，我们已经构建了一个非常值得尊敬的[函数式编程语言](http://en.wikipedia.org/wiki/Functional_programming).]。在我们的旅程中，我们学习了一些解析技术，如何构建和表示一个AST，如何构建LLVMIR，以及如何优化结果代码和即时编译它。

虽然Kaleidoscope作为一种函数式语言很有趣，但它是函数式的这一事实使得为它生成LLVMIR“太容易”了。特别是，函数式语言使得直接在[ssa form](http://en.wikipedia.org/wiki/Static_single_assignment_form)中构建LLVMIR变得非常容易由于LLVM要求输入代码采用SSA形式，这是一个非常好的属性，新手通常不清楚如何为具有可变变量的命令式语言生成代码。

本章的简短(令人愉快的)总结是，您的前端不需要构建SSA表单：LLVM为此提供了高度调优和经过良好测试的支持，尽管它的工作方式对某些人来说有点出乎意料。

## 为什么这是一个很难解决的问题？

要理解为什么可变变量会导致SSA构造的复杂性，请考虑下面这个极其简单的C示例：

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

在本例中，我们有变量“X”，它的值取决于程序中执行的路径。因为在返回指令之前X有两个不同的可能值，所以插入一个PHI节点来合并这两个值。本例需要的LLVM IR如下所示：

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

在本例中，来自G和H全局变量的加载在LLVM IR中是显式的，它们位于if语句(cond_true/cond_false)的THEN/ELSE分支中。为了合并传入的值，COND_NEXT block中的X.2 φ节点根据控制流来自何处选择要使用的正确值：如果控制流来自COND_FALSE Block，则X.2获取X.1的值。或者，如果控制流来自cond_true，它将获得X.0的值。本章的目的不是解释SSA表单的细节。有关详细信息，请参阅众多[线上参考资料](http://en.wikipedia.org/wiki/Static_single_assignment_form).]中的一个。

本文的问题是“对可变变量赋值降维时，谁放置φ节点？”。这里的问题是llvm*需要*它的IR必须是ssa形式的：它没有“非ssa”模式。但是，SSA的构建需要不平凡的算法和数据结构，所以每个前端都要重现这个逻辑是浪费并且不方便的。

## LLVM中的内存

这里的“诀窍”是，虽然LLVM确实要求所有寄存器值都采用SSA格式，但它并不要求(或允许)内存对象采用SSA格式。在上面的示例中，请注意来自G和H的载荷是对G和H的直接访问：它们没有重命名或版本化。这与其他一些编译器系统不同，其他编译器系统确实会尝试对内存对象进行版本化。在LLVM中，不是将内存的数据流分析编码到LLVM IR中，而是使用按需计算的[分析通道(Analysis Passes)](../../WritingAnLLVMPass.html)进行处理。

考虑到这一点，高级想法是我们希望为函数中的每个可变对象创建一个堆栈变量(它驻留在内存中，因为它在堆栈上)。要利用此技巧，我们需要讨论LLVM如何表示堆栈变量。

在LLVM中，所有内存访问都是使用加载/存储指令显式进行的，并且它被精心设计为不具有(或不需要)“address-of”运算符。请注意，即使变量定义为“I32”，\@G/\@H全局变量的类型实际上也是“I32\*”。这意味着，\@G在全局数据区域中为I32定义了*空间*，但它的*名字*实际上是指该空间的地址。堆栈变量的工作方式相同，不同之处在于它们不是使用全局变量定义声明的，而是使用[LLVM Alloca instruction](../../LangRef.html#alloca-instruction)：]声明的

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

此代码显示了如何在LLVM IR中声明和操作堆栈变量的示例。使用alloca指令分配的堆栈内存是完全通用的：您可以将堆栈槽的地址传递给函数，也可以将其存储在其他变量中，依此类推。在上面的示例中，我们可以重写示例以使用alloca技术来避免使用PHI节点：

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

这样，我们就发现了一种处理任意可变变量的方法，而根本不需要创建Phi节点：

1. 每个可变变量都由堆栈分配。
2. 每次读取变量都会成为堆栈中的加载load。
3. 变量的每次更新都会成为堆栈的存储store。
4. 获取变量的地址只需直接使用堆栈地址。

虽然这个解决方案解决了我们眼前的问题，但它引入了另一个问题：我们现在显然为非常简单和常见的操作引入了大量堆栈流量，这是一个主要的性能问题。对我们来说幸运的是，LLVM优化器有一个名为“mem2reg”的高度调优的优化通道来处理这种情况，它会将这样的分配提升到SSA寄存器中，并在适当的时候插入Phi节点。例如，如果通过该过程运行此示例，您将获得：

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

mem2reg pass实现了用于构建SSA表单的标准“迭代优势边界(iterated dominance frontier)”算法，并进行了许多优化以加速(非常常见的)退化情况。mem2reg优化通道是处理可变变量的答案，我们强烈建议您依赖它。请注意，mem2reg仅在某些情况下适用于变量：

1. mem2reg是由alloca驱动的：它查找alloca，如果它能处理它们，它就会提升它们。它不适用于全局变量或堆分配。
2. mem2reg只在函数的entry Block中查找alloca指令。在entry Block中可以保证alloca只执行一次，这使得分析更简单。
3. mem2reg仅提升用途是直接加载和存储的alloca。如果将堆栈对象的地址传递给函数，或者如果涉及任何有趣的指针算法，则不会提升alloca。
4. mem2reg仅适用于[First class](../../LangRef.html#first-class-type)值的alloca(如指针、标量和向量)，并且仅当allocation的数组大小为1(或.ll文件中缺少)时才有效。mem2reg不能将结构或数组提升到寄存器。请注意，“sroa”通道功能更强大，在许多情况下可以提升struct、“union”和array。

对于大多数命令式语言来说，所有这些属性都很容易满足，我们将在下面用Kaleidoscope进行说明。您可能会问的最后一个问题是：我是否应该在前端进行这种无意义的折腾？如果我直接进行SSA构造，避免使用mem2reg优化通道，不是更好吗？简而言之，我们强烈建议您使用此技术来构建SSA表单，除非有非常好的理由不这样做。使用此技术是：

- 经过验证和良好测试：Clang将此技术用于局部可变变量。因此，LLVM最常见的客户端使用它来处理它们的大部分变量。您可以确保快速发现并及早修复错误。
- 极快：mem2reg有许多特殊情况，这使得它在普通情况下和完全通用情况下都很快。例如，它具有只在单个Block中使用的变量的快速路径，只有一个赋值点的变量，避免插入不需要的φ节点的良好启发式方法，等等。
- 生成调试信息所需：[LLVM中的调试信息](../../SourceLevelDebugging.html)依赖于公开变量的地址，以便可以附加调试信息。这种技术与这种风格的调试信息非常自然地吻合。

如果没有其他问题，这将使您的前端更容易启动和运行，并且实现起来非常简单。现在让我们用可变变量来扩展Kaleidoscope！

## Kaleidoscope中的可变变量

现在我们知道了我们想要解决的问题类型，让我们看看这在我们的Kaleidoscope语言的上下文中是什么样子。我们将添加两个功能：

1. 使用‘=’运算符修改变量的能力。
2. 定义新变量的能力。

尽管第一项实际上是关于这一点的，但我们只有用于传入参数和推导变量的变量，重新定义这些变量也就到此为止了：)。此外，定义新变量的能力是一件很有用的事情，无论您是否要对它们进行修改。下面是一个鼓舞人心的例子，它展示了我们如何使用这些：
```
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
```

为了使变量发生改变，我们必须更改现有变量以使用“alloca技巧”。完成后，我们将添加新的运算符，然后扩展Kaleidoscope以支持新的变量定义。

## 调整现有变量以进行改变

Kaleidoscope中的符号表在代码生成时由‘`NamedValues`’映射管理。此映射当前跟踪保存已命名变量的双精度值的LLVM“value\*”。为了支持修改，我们需要稍微更改一下，以便`NamedValues`保存需要修改变量的*内存位置*。请注意，此更改是一种重构：它更改了代码的结构，但(本身)不更改编译器的行为。所有这些更改都隔离在Kaleidoscope代码生成器中。

在Kaleidoscope开发的这一点上，它只支持两件事的变量：函数的传入参数和‘for’循环的推导变量。为了保持一致性，除了其他用户定义的变量外，我们还允许这些变量的改变。这意味着这些变量都需要内存位置。

要开始转换Kaleidoscope，我们将更改NamedValues映射，使其映射到AllocaInst\*而不是Value\*。完成此操作后，C++编译器将告诉我们需要更新代码的哪些部分：

```c++
static std::map<std::string, AllocaInst*> NamedValues;
```

另外，由于我们将需要创建这些allocas，因此我们将使用一个助手函数来确保在函数的entry Block中创建allocas：

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

这段看起来很滑稽的代码创建了一个IRBuilder对象，该对象指向Blockentry 的第一条指令(.Begin())。然后，它创建一个具有预期名称的alloca并返回它。因为Kaleidoscope中的所有值都是双精度值，所以不需要传入类型即可使用。

有了这一点，我们要进行的第一个功能更改属于变量引用。在我们的新方案中，变量驻留在堆栈中，因此生成对它们的引用的代码实际上需要从堆栈插槽生成加载：

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

如您所见，这非常简单。现在我们需要更新定义变量的内容来设置alloca。我们将从`ForExprAST::codegen()`开始(未删节的代码参见[完整代码清单](#Id1))：

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

此代码实际上与[在我们允许可变variables](zh-LangImpl05.html#code-generation-for-the-for-loop).之前]的代码相同。最大的区别在于，我们不再需要构造PHI节点，而是根据需要使用加载(load)/存储(store)来访问变量。

为了支持可变参数变量，我们还需要为它们进行分配。这方面的代码也非常简单：

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

对于每个参数，我们创建一个Alloca，将函数的输入值存储到Alloca中，并将Alloca注册为参数的内存位置。此方法由`FunctionAST::codegen()`在为函数设置entry Block后立即调用。

最后缺少的部分是添加mem2reg pass，它允许我们再次获得良好的编解码器：

```c++
// Promote allocas to registers.
TheFPM->add(createPromoteMemoryToRegisterPass());
// Do simple "peephole" optimizations and bit-twiddling optzns.
TheFPM->add(createInstructionCombiningPass());
// Reassociate expressions.
TheFPM->add(createReassociatePass());
...
```

看看mem2reg优化运行前后的代码是什么样子是很有趣的。例如，这是我们的递归fib函数的前后代码。优化前：

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

这里只有一个变量(x，输入参数)，但是您仍然可以看到我们正在使用的极其简单的代码生成策略。在entry Block中，创建一个alloca，并将初始输入值存储在其中。每个对变量的引用都会从堆栈重新加载一次。另外，请注意，我们没有修改if/Then/Else表达式，所以它仍然插入一个PHI节点。虽然我们可以为它创建一个alloca，但实际上为它创建一个PHI节点更容易，所以我们仍然只创建PHI。

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

对于mem2reg来说，这是一个微不足道的例子，因为没有重新定义变量。展示这一点的目的是为了平息你对插入这种明显的低效行为的紧张情绪：)。

优化器的剩余部分运行后，我们得到：

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

在这里我们可以看到，simplifycfg pass决定将返回指令克隆到‘Else’Block的末尾。这允许它消除一些分支和PHI节点。

现在所有符号表引用都更新为使用堆栈变量，我们将添加赋值运算符。

## 新建赋值运算符

使用我们当前的框架，添加一个新的赋值操作符非常简单。我们将像解析任何其他二元运算符一样解析它，但在内部处理它(而不是允许用户定义它)。第一步是设置优先级：

```c++
int main() {
  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
```

既然解析器知道二元运算符的优先级，它就负责所有的解析和AST生成。我们只需要为赋值操作符实现codegen。这看起来像下文这样：

```c++
Value *BinaryExprAST::codegen() {
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    // Assignment requires the LHS to be an identifier.
    VariableExprAST *LHSE = dynamic_cast<VariableExprAST*>(LHS.get());
    if (!LHSE)
      return LogErrorV("destination of '=' must be a variable");
```

与其他的二元运算符不同，我们的赋值运算符没有遵循“发出lhs，发出rh，做计算”的模型，所以在处理其他二元运算符之前，会将其作为特例来处理。另一个奇怪的事情是，它要求lhs是一个变量。有“(x+1)=expr”是无效的-只允许“x=expr”这样的东西。

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

一旦我们有了变量，赋值的代码生成就很简单了：我们发出赋值的RHS，创建一个存储，并返回计算值。返回值允许像“X=(Y=Z)”这样的链式赋值。

现在我们有了赋值操作符，我们可以改变循环变量和参数。例如，我们现在可以运行如下代码：
```
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
```

运行时，此示例打印“123”，然后打印“4”，表明我们确实改变了值！好的，我们现在已经正式实现了我们的目标：要想让它正常工作，一般情况下需要SSA构建。然而，为了真正有用，我们希望能够定义我们自己的局部变量，接下来让我们添加这个！

## 用户定义的局部变量

添加var/in就像我们对Kaleidoscope所做的任何其他扩展一样：我们扩展了词法分析器、解析器、AST和代码生成器。添加新的‘var/in’结构的第一步是扩展词法分析器。与前面一样，这非常简单，代码如下所示：

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

下一步是定义我们将构造的AST节点。对于var/in，如下所示：

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

var/in允许一次定义所有名称列表，并且每个名称可以有一个可选的初始值。这样，我们在VarNames矢量中捕获此信息。另外，var/in有一个主体（Body），这个主体允许访问由var/in定义的变量。

有了这些，我们就可以定义解析器部分了。我们要做的第一件事是将其添加为主表达式：

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

这段代码的第一部分将标识符/表达式对的列表解析为本地的“VarNames`”向量。

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

一旦解析完所有变量，我们就解析正文并创建AST节点：

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

现在我们可以解析和表示代码了，我们需要支持它的LLVM IR发射。此代码以以下代码开头：

```c++
Value *VarExprAST::codegen() {
  std::vector<AllocaInst *> OldBindings;

  Function *TheFunction = Builder.GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();
```

基本上，它循环所有变量，一次安装一个变量。对于我们放到符号表中的每个变量，我们都会记住在OldBindings中替换的前一个值。

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

这里的注释比代码多。基本思想是发出初始值设定项，创建alloca，然后更新符号表以指向它。一旦所有变量都安装到符号表中，我们将计算var/in表达式的主体（body）：

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

所有这一切的最终结果是我们得到了适当范围的变量定义，并且我们甚至(微不足道地)允许对它们进行修改：)。

有了这个，我们就完成了我们开始要做的事情。我们从开头给出的漂亮的迭代fib示例编译得并运行得很好。mem2reg pass优化了SSA寄存器中的所有堆栈变量，在需要的地方插入PHI节点，并且我们的前端仍然很简单：在任何地方都看不到“迭代优势边界(iterated dominance frontier)”计算。

## 完整代码列表

下面是我们的运行示例的完整代码清单，增强了可变变量和var/in支持。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

[下一步：编译为对象代码](zh-LangImpl08.html)
