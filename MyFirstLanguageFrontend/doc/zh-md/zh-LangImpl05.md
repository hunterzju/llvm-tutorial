# Kaleidoscope：扩展语言：控制流

## 第五章绪论

欢迎阅读“[使用LLVM实现语言](zh-index.md)”教程的第5章。第1-4部分描述了简单Kaleidoscope语言的实现，包括对生成LLVM IR的支持，随后是优化和JIT编译器。不幸的是，正如所展示的那样，Kaleidoscope几乎毫无用处：除了调用和返回之外，它没有任何控制流。这意味着你在代码中不能有条件分支，这大大限制了它的功能。在“构建编译器”的这一集中，我们将扩展Kaleidoscope，使其有一个if/Then/Else表达式和一个简单的‘for’循环。

## IF/THEN/ELSE

扩展Kaleidoscope以支持IF/THEN/ELSE非常简单。它基本上需要向词法分析器、解析器、AST和LLVM代码发射器添加对这个“新”概念的支持。这个例子很不错，因为它展示了随着时间的推移“扩展”一门语言是多么容易，随着新思想的发现而逐渐扩展。

在我们继续“如何”添加此扩展之前，让我们先来讨论一下我们想要什么。基本思想是我们希望能够编写这样的东西：
```
    def fib(x)
      if x < 3 then
        1
      else
        fib(x-1)+fib(x-2);
```

在Kaleidoscope中，每个结构都是一个表达式(expression)：没有语句(statement)。因此，IF/THEN/ELSE表达式需要像其他表达式一样返回值。因为我们使用的主要是函数形式，所以我们将让它评估其条件，然后根据条件的解决方式返回‘THEN’或‘ELSE’值。这与C“？：”表达式非常相似。

IF/THEN/ELSE表达式的语义是它将条件计算为布尔相等的值：0.0被认为是假的，而其他一切都被认为是真的。如果条件为TRUE，则计算并返回第一个子表达式；如果条件为FALSE，则计算并返回第二个子表达式。因为Kaleidoscope允许[side effect](https://en.wikipedia.org/wiki/Side_effect_(computer_science)#:~:text=In%20computer%20science%2C%20an%20operation,the%20invoker%20of%20the%20operation.)，所以这一行为对于确定分支是很重要的。

既然我们知道了我们“想要”什么，让我们把它分解成几个组成部分。

### IF/THEN/ELSE的词法分析器扩展

词法分析器扩展很简单。首先，我们为相关令牌添加新的枚举值：

```c++
// control
tok_if = -6,
tok_then = -7,
tok_else = -8,
```

一旦我们有了它，我们就可以识别词法分析器中的新关键字。这是非常简单的东西：

```c++
...
if (IdentifierStr == "def")
  return tok_def;
if (IdentifierStr == "extern")
  return tok_extern;
if (IdentifierStr == "if")
  return tok_if;
if (IdentifierStr == "then")
  return tok_then;
if (IdentifierStr == "else")
  return tok_else;
return tok_identifier;
```

### IF/THEN/ELSE的AST扩展

为了表示新表达式，我们为其添加一个新的AST节点：

```c++
/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond, Then, Else;

public:
  IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then,
            std::unique_ptr<ExprAST> Else)
    : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}

  Value *codegen() override;
};
```

AST节点只有指向各种子表达式的指针。

### IF/THEN/ELSE的解析器扩展

既然我们有了来自词法分析器的相关令牌，也有了要构建的AST节点，我们的解析逻辑就相对简单了。首先，我们定义一个新的解析函数：

```c++
/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr() {
  getNextToken();  // eat the if.

  // condition.
  auto Cond = ParseExpression();
  if (!Cond)
    return nullptr;

  if (CurTok != tok_then)
    return LogError("expected then");
  getNextToken();  // eat the then

  auto Then = ParseExpression();
  if (!Then)
    return nullptr;

  if (CurTok != tok_else)
    return LogError("expected else");

  getNextToken();

  auto Else = ParseExpression();
  if (!Else)
    return nullptr;

  return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
}
```

接下来，我们将其作为主表达式连接起来：

```c++
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
  }
}
```

### IF/THEN/ELSE的LLVM IR

现在我们已经有了解析和构建AST的功能，最后一部分是添加LLVM代码生成支持。这是IF/THEN/ELSE示例中最有趣的部分，因为这是引入新概念的开始。上面的所有代码都在前面的章节中进行了详细描述。

为了激发我们想要生成的代码，让我们来看一个简单的例子。考虑一下：
```
    extern foo();
    extern bar();
    def baz(x) if x then foo() else bar();
```

如果禁用优化，您将(很快)从Kaleidoscope获得的代码如下所示：

```llvm
declare double @foo()

declare double @bar()

define double @baz(double %x) {
entry:
  %ifcond = fcmp one double %x, 0.000000e+00
  br i1 %ifcond, label %then, label %else

then:       ; preds = %entry
  %calltmp = call double @foo()
  br label %ifcont

else:       ; preds = %entry
  %calltmp1 = call double @bar()
  br label %ifcont

ifcont:     ; preds = %else, %then
  %iftmp = phi double [ %calltmp, %then ], [ %calltmp1, %else ]
  ret double %iftmp
}
```

要可视化控制流图，您可以使用LLVM[OPT](https://llvm.org/cmds/opt.html)工具的一个很好的特性。如果您将此LLVMIR放入“t.ll”并运行“`llvm-as < t.ll | OPT-ANALYLE-VIEW-Cfg`”，[将弹出一个窗口up](https://llvm.org/docs/ProgrammersManual.html#viewing-graphs-while-debugging-code)，您将看到此图形：

！[示例配置](../pics/zh-LangImpl05-cfg.png){.ign-center}

实现这一点的另一种方法是调用“`F->viewCFG()`”或“`F->viewCFGOnly()`”(其中F是“`Function*`”)，方法是将实际调用插入代码并重新编译，或者在调试器中调用它们。LLVM有许多用于可视化各种图形的很好的特性。

返回到生成的代码，它相当简单：entry block计算条件表达式(在我们的示例中是“x”)，并用“`fcmp one`”指令(‘one’is“Ordered and Not Equity”)将结果与0.0进行比较。根据该表达式的结果，代码跳转到“THEN”或“ELSE”块，这两个块包含TRUE/FALSE情况的表达式。

THEN/ELSE块执行完毕后，它们都会分支回‘ifcont’块，以执行IF/THEN/ELSE之后发生的代码。在这种情况下，剩下的唯一要做的事情就是返回到函数的调用方。然后问题就变成了：代码如何知道要返回哪个表达式？

这个问题的答案涉及到一个重要的SSA操作：[Phi operation](http://en.wikipedia.org/wiki/Static_single_assignment_form).如果你不熟悉ssa，[维基百科article](http://en.wikipedia.org/wiki/Static_single_assignment_form)是一个很好的介绍，在你最喜欢的搜索引擎上有各种各样的其他介绍。简而言之，“执行”φ操作需要“记住”哪个block控件是从何而来的。φ操作采用与input control block相对应的值。在本例中，如果控制权来自“THEN”block，它将获得“calltmp”的值。如果控制权来自“Else”block，则获取“calltmp1”的值。

在这一点上，您可能开始想“哦，不！这意味着我的简单而优雅的前端必须开始生成SSA表单才能使用LLVM！”幸运的是，情况并非如此，我们强烈建议*不*在您的前端实现SSA构建算法，除非有令人惊讶的好理由。实际上，在为一般命令式编程语言编写的代码中，有两种值待计算的值可能需要φ节点：

1. 涉及用户变量的代码：`x=1；x=x+1；`
2. 隐含在AST结构中的值，如在本例中为Phi节点。

在本教程(“可变变量”)的[第7章](zh-LangImpl07.md)中，我们将深入讨论#1。现在，请相信我，您不需要使用SSA构造来处理这种情况。对于#2，您可以选择使用我们将在#1中描述的技术，也可以在方便的情况下直接插入Phi节点。在这种情况下，生成Phi节点非常容易，所以我们选择直接执行。

好了，动机和概述到此为止，让我们生成代码吧！

### IF/THEN/ELSE的代码生成

为了生成代码，我们为`IfExprAST`实现了`codegen`方法：

```c++
Value *IfExprAST::codegen() {
  Value *CondV = Cond->codegen();
  if (!CondV)
    return nullptr;

  // Convert condition to a bool by comparing non-equal to 0.0.
  CondV = Builder.CreateFCmpONE(
      CondV, ConstantFP::get(TheContext, APFloat(0.0)), "ifcond");
```

这段代码简单明了，与我们之前看到的类似。我们发出该条件的表达式，然后将该值与零进行比较，以获得1位(布尔值)形式的真值。

```c++
Function *TheFunction = Builder.GetInsertBlock()->getParent();

// Create blocks for the then and else cases.  Insert the 'then' block at the
// end of the function.
BasicBlock *ThenBB =
    BasicBlock::Create(TheContext, "then", TheFunction);
BasicBlock *ElseBB = BasicBlock::Create(TheContext, "else");
BasicBlock *MergeBB = BasicBlock::Create(TheContext, "ifcont");

Builder.CreateCondBr(CondV, ThenBB, ElseBB);
```

此代码创建与IF/THEN/ELSE语句相关的基本块，并直接对应于上面示例中的块。第一行获取正在构建的当前函数对象。它通过向构建器询问当前的BasicBlock，并向block询问它的“父节点”(它当前嵌入到其中的函数)来实现这一点。

一旦有了它，它就会创建三个块。注意，它将“TheFunction”传递给“THEN”block的构造函数。这会使构造函数自动将新block插入到指定函数的末尾。其他两个块已创建，但尚未插入到函数中。

一旦创建了块，我们就可以发出在它们之间进行选择的条件分支。请注意，创建新块不会隐式影响IRBuilder，因此它仍会插入到条件进入的block中。还要注意的是，它正在创建一个指向“THEN”block和“ELSE”block的分支，尽管“ELSE”block还没有插入到函数中。这一切都没问题：这是LLVM支持正向引用的标准方式。

```c++
// Emit then value.
Builder.SetInsertPoint(ThenBB);

Value *ThenV = Then->codegen();
if (!ThenV)
  return nullptr;

Builder.CreateBr(MergeBB);
// Codegen of 'Then' can change the current block, update ThenBB for the PHI.
ThenBB = Builder.GetInsertBlock();
```

在插入条件分支之后，我们移动构建器以开始插入到“THEN”block中。严格地说，此调用将插入点移动到指定block的末尾。不过，由于“THEN”block是空的，所以也是从插入block开头开始的。：)

一旦设置了插入点，我们就从AST递归地编码生成“THEN”表达式。为了完成“THEN”block，我们创建一个无条件分支来合并block。LLVM IR的一个有趣(也是非常重要的)方面是，它要求所有基本块都使用一个[`控制流指令`](https://llvm.org/docs/LangRef.html#terminators)(如return或分支)“终止”。这意味着所有控制流*包括fall-through*必须在LLVMIR中显式显示。如果您违反此规则，验证器将发出错误。

这里的最后一行相当微妙，但非常重要。基本问题是，当我们在合并block中创建phi节点时，我们需要设置block/value对，以指示phi将如何工作。重要的是，phi节点希望在cfg中为block的每个前驱都有一个条目。那么，为什么我们刚刚将block设置为以上5行，就会得到当前的block呢？问题是，then block中可能实际上会修改生成器Builder发送到if中的block，比如then表达式中包含嵌套的“IF/THEN/ELSE”表达式。因为递归调用`codegen()`可能会任意改变当前block的概念，所以我们需要获取最新值，赋值给设置Phi节点的代码。

```c++
// Emit else block.
TheFunction->getBasicBlockList().push_back(ElseBB);
Builder.SetInsertPoint(ElseBB);

Value *ElseV = Else->codegen();
if (!ElseV)
  return nullptr;

Builder.CreateBr(MergeBB);
// codegen of 'Else' can change the current block, update ElseBB for the PHI.
ElseBB = Builder.GetInsertBlock();
```

Elseblock的代码生成与then block的代码生成基本相同。唯一显著的区别是第一行，它将‘Else’block添加到函数中。回想一下，前面已经创建了‘Else’block，但没有添加到函数中。现在已经发出了‘THEN’和‘ELSE’块，我们可以完成合并代码：

```c++
// Emit merge block.
TheFunction->getBasicBlockList().push_back(MergeBB);
Builder.SetInsertPoint(MergeBB);
PHINode *PN =
  Builder.CreatePHI(Type::getDoubleTy(TheContext), 2, "iftmp");

PN->addIncoming(ThenV, ThenBB);
PN->addIncoming(ElseV, ElseBB);
return PN;
}
```

这里的前两行现在很熟悉：第一行将“Merge”block添加到函数对象中(它以前是浮点的，就像上面的Elseblock一样)。第二个更改插入点，以便新创建的代码将进入“Merge”block。完成后，我们需要创建PHI节点并为PHI设置block/value对。

最后，CodeGen函数将phi节点作为IF/THEN/ELSE表达式计算的值返回。在上面的示例中，此返回值将提供给顶层函数的代码，该代码将创建返回指令。

总体而言，我们现在能够在Kaleidoscope中执行条件代码。有了这个扩展，Kaleidoscope是一种相当完整的语言，可以计算各种各样的数值函数。接下来，我们将添加另一个在非函数式语言中熟悉的有用表达式.

## ‘for’循环表达式

既然我们知道了如何将基本的控制流结构添加到语言中，我们就有了工具来添加更强大的东西。让我们添加一些更具攻击性的东西，‘for’表达式：
```
    extern putchard(char);
    def printstar(n)
      for i = 1, i < n, 1.0 in
        putchard(42);  # ascii 42 = '*'

    # print 100 '*' characters
    printstar(100);
```

该表达式定义了一个从起始值迭代的新变量(在本例中为“i”)，而条件(在本例中为“i < n”)为真，递增一个可选的步长值(在本例中为“1.0”)。如果省略步长值，则默认为1.0。当循环为真时，它执行其主体表达式。因为我们没有更好的返回，所以我们将循环定义为总是返回0.0。将来当我们有可变变量时，它会变得更有用。

像以前一样，让我们来讨论一下我们需要对Kaleidoscope进行哪些更改来支持这一点。

### ‘for’循环的词法分析器扩展

词法分析器扩展与IF/THEN/ELSE相同：

```c++
... in enum Token ...
// control
tok_if = -6, tok_then = -7, tok_else = -8,
tok_for = -9, tok_in = -10

... in gettok ...
if (IdentifierStr == "def")
  return tok_def;
if (IdentifierStr == "extern")
  return tok_extern;
if (IdentifierStr == "if")
  return tok_if;
if (IdentifierStr == "then")
  return tok_then;
if (IdentifierStr == "else")
  return tok_else;
if (IdentifierStr == "for")
  return tok_for;
if (IdentifierStr == "in")
  return tok_in;
return tok_identifier;
```

### ‘for’循环的AST扩展

AST节点也同样简单。它基本上归结为捕获节点中的变量名和组成表达式。

```c++
/// ForExprAST - Expression class for for/in.
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
  ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
             std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
             std::unique_ptr<ExprAST> Body)
    : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
      Step(std::move(Step)), Body(std::move(Body)) {}

  Value *codegen() override;
};
```

### ‘for’循环的解析器扩展

解析器代码也相当标准。这里唯一有趣的事情是处理可选的步长值。解析器代码通过检查第二个逗号是否存在来处理它。如果不是，则在AST节点中将步长值设置为NULL：

```c++
/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr() {
  getNextToken();  // eat the for.

  if (CurTok != tok_identifier)
    return LogError("expected identifier after for");

  std::string IdName = IdentifierStr;
  getNextToken();  // eat identifier.

  if (CurTok != '=')
    return LogError("expected '=' after for");
  getNextToken();  // eat '='.


  auto Start = ParseExpression();
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("expected ',' after for start value");
  getNextToken();

  auto End = ParseExpression();
  if (!End)
    return nullptr;

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok == ',') {
    getNextToken();
    Step = ParseExpression();
    if (!Step)
      return nullptr;
  }

  if (CurTok != tok_in)
    return LogError("expected 'in' after for");
  getNextToken();  // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<ForExprAST>(IdName, std::move(Start),
                                       std::move(End), std::move(Step),
                                       std::move(Body));
}
```

我们再一次把它作为一个主要的表达式：

```c++
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
  }
}
```

### ‘for’循环的LLVM IR

现在我们来看好的部分：我们想要为这件事生成的LLVM IR。通过上面的简单示例，我们将获得此LLVM IR(请注意，为清晰起见，生成此转储时禁用了优化)：

```llvm
declare double @putchard(double)

define double @printstar(double %n) {
entry:
  ; initial value = 1.0 (inlined into phi)
  br label %loop

loop:       ; preds = %loop, %entry
  %i = phi double [ 1.000000e+00, %entry ], [ %nextvar, %loop ]
  ; body
  %calltmp = call double @putchard(double 4.200000e+01)
  ; increment
  %nextvar = fadd double %i, 1.000000e+00

  ; termination test
  %cmptmp = fcmp ult double %i, %n
  %booltmp = uitofp i1 %cmptmp to double
  %loopcond = fcmp one double %booltmp, 0.000000e+00
  br i1 %loopcond, label %loop, label %afterloop

afterloop:      ; preds = %loop
  ; loop always returns 0.0
  ret double 0.000000e+00
}
```

这个循环包含我们以前看到的所有相同的结构：一个phi节点、几个表达式和一些基本块。让我们看看这两个组件是如何搭配在一起的。

### ‘for’循环的代码生成

codegen的第一部分非常简单：我们只输出循环值的开始表达式：

```c++
Value *ForExprAST::codegen() {
  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen();
  if (!StartVal)
    return nullptr;
```

这样就解决了问题，下一步是为循环体的开始设置LLVM Basicblock。在上面的例子中，整个循环体是一个block，但请记住，体代码本身可以由多个块组成(例如，如果它包含IF/THEN/ELSE或FOR/in表达式)。

```c++
// Make the new basic block for the loop header, inserting after current
// block.
Function *TheFunction = Builder.GetInsertBlock()->getParent();
BasicBlock *PreheaderBB = Builder.GetInsertBlock();
BasicBlock *LoopBB =
    BasicBlock::Create(TheContext, "loop", TheFunction);

// Insert an explicit fall through from the current block to the LoopBB.
Builder.CreateBr(LoopBB);
```

此代码类似于我们在if/Then/Else中看到的代码。因为我们将需要它来创建Phi节点，所以我们记住了落入循环中的block。完成后，我们将创建实际的block来启动循环，并为两个块之间的fall-through创建无条件的分支。

```c++
// Start insertion in LoopBB.
Builder.SetInsertPoint(LoopBB);

// Start the PHI node with an entry for Start.
PHINode *Variable = Builder.CreatePHI(Type::getDoubleTy(TheContext),
                                      2, VarName.c_str());
Variable->addIncoming(StartVal, PreheaderBB);
```

现在已经设置了循环的“preheader”，我们切换到为循环体发送代码。首先，我们移动插入点并为loop induction变量创建PHI节点。因为我们已经知道起始值的传入值，所以我们将其添加到phi节点。注意，phi最终将为backedge获得第二个值，但是我们还不能设置它(因为它不存在！)。

```c++
// Within the loop, the variable is defined equal to the PHI node.  If it
// shadows an existing variable, we have to restore it, so save it now.
Value *OldVal = NamedValues[VarName];
NamedValues[VarName] = Variable;

// Emit the body of the loop.  This, like any other expr, can change the
// current BB.  Note that we ignore the value computed by the body, but don't
// allow an error.
if (!Body->codegen())
  return nullptr;
```

现在代码开始变得更有趣了。我们的‘for’循环在符号表中引入了一个新变量。这意味着我们的符号表现在可以包含函数参数或循环变量。为了处理这个问题，在我们对循环体进行编码之前，我们添加循环变量作为其名称的当前值。请注意，外部作用域中可能存在同名的变量。很容易将此设置为错误(如果已有VarName条目，则发出错误并返回NULL)，但我们选择允许跟踪变量。为了正确处理这个问题，我们要记住在`OldVal`中可能隐藏的值(如果没有隐藏变量，则该值为NULL)。

一旦循环变量被设置到符号表中，代码递归地调用codegen。这允许主体使用循环变量：任何对它的引用都会自然地在符号表中找到它。

```c++
// Emit the step value.
Value *StepVal = nullptr;
if (Step) {
  StepVal = Step->codegen();
  if (!StepVal)
    return nullptr;
} else {
  // If not specified, use 1.0.
  StepVal = ConstantFP::get(TheContext, APFloat(1.0));
}

Value *NextVar = Builder.CreateFAdd(Variable, StepVal, "nextvar");
```

既然主体已经发出（emit），我们将通过添加Step值来计算迭代变量的下一个值，如果不存在，则使用1.0。‘`NextVar`’将是循环下一次迭代的循环变量的值。

```c++
// Compute the end condition.
Value *EndCond = End->codegen();
if (!EndCond)
  return nullptr;

// Convert condition to a bool by comparing non-equal to 0.0.
EndCond = Builder.CreateFCmpONE(
    EndCond, ConstantFP::get(TheContext, APFloat(0.0)), "loopcond");
```

最后，我们评估循环的退出值，以确定循环是否应该退出。这其实是IF/THEN/ELSE语句的条件求值的镜像。

```c++
// Create the "after loop" block and insert it.
BasicBlock *LoopEndBB = Builder.GetInsertBlock();
BasicBlock *AfterBB =
    BasicBlock::Create(TheContext, "afterloop", TheFunction);

// Insert the conditional branch into the end of LoopEndBB.
Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

// Any new code will be inserted in AfterBB.
Builder.SetInsertPoint(AfterBB);
```

循环主体的代码完成后，我们只需要完成它的控制流。此代码记住结束的block(对于Phi节点)，然后为循环出口创建block(“After Loop”)。根据退出条件的值，它创建一个条件分支，在再次执行循环和退出循环之间进行选择。将来的任何代码都会在“After Loop”block中发出，因此它会将插入位置设置为它。

```c++
// Add a new entry to the PHI node for the backedge.
Variable->addIncoming(NextVar, LoopEndBB);

// Restore the unshadowed variable.
if (OldVal)
  NamedValues[VarName] = OldVal;
else
  NamedValues.erase(VarName);

// for expr always returns 0.0.
return Constant::getNullValue(Type::getDoubleTy(TheContext));
}
```

最后的代码处理各种清理：现在我们有了“NextVar”值，我们可以将传入的值添加到循环PHI节点。之后，我们从符号表中删除循环变量，以便它不在for循环之后的作用域内。最后，for循环的代码生成总是返回0.0，这就是我们从`ForExprAST::codegen()`返回的内容。

至此，我们结束了本教程的“向Kaleidoscope添加控制流”一章。在本章中，我们添加了两个控制流构造，并使用它们来激发LLVM IR的一些重要方面，这些方面对于前端实现者来说是非常重要的。在我们传奇的下一章中，我们将变得更加疯狂，将[用户定义操作符](zh-LangImpl06.md)添加到我们可怜又无辜的语言中。

## 完整代码列表

下面是我们的运行示例的完整代码清单，并使用if/Then/Else和For表达式进行了增强。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```
[https://github.com/llvm/llvm-project/blob/main/llvm/examples/Kaleidoscope/Chapter5/toy.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/examples/Kaleidoscope/Chapter5/toy.cpp)

[下一步：扩展语言：自定义运算符](zh-LangImpl06.md)
