# 万花筒：扩展语言：控制流

：{.content local=“”}
**

## 第五章绪论

欢迎学习\“的第5章[使用实现语言
LLVM](index.html)\“教程。第1-4部分描述了
简单的万花筒语言并包含对生成的支持
LLVM IR，然后是优化和JIT编译器。不幸的是，由于
目前，万花筒几乎毫无用处：它没有其他控制流
而不是打电话回来。这意味着您不能有条件
代码中的分支，极大地限制了它的功能。在本期节目中
“构建编译器”，我们将对万花筒进行扩展，使其具有
IF/THEN/ELSE表达式加上一个简单的\‘for\’循环。

## 如果/然后/否则

扩展万花筒以支持IF/THEN/ELSE非常简单。
它基本上需要将对此\“新\”概念的支持添加到
lexer、parser、AST和LLVM代码发射器。这个示例很不错，因为
它展示了随着时间的推移，增量地\“成长\”一门语言是多么容易
当新的想法被发现时，扩展它。

在我们继续“如何”添加此扩展之前，让我们先谈谈
\“我们想要的\”基本的想法是我们想要能够写
这类事情：

    def fib(x)
      if x < 3 then
        1
      else
        fib(x-1)+fib(x-2);

在万花筒中，每个结构都是一个表达式：没有
结算单。因此，IF/THEN/ELSE表达式需要返回值
和其他人一样。因为我们使用的主要是函数形式，所以我们必须
它评估其条件，然后返回\‘THEN\’或\‘ELSE\’值
根据情况是如何解决的。这与C非常相似
\“？：\”表达式。

IF/THEN/ELSE表达式的语义是它计算
条件设置为布尔相等值：0.0被认为是假的，并且
其他一切都被认为是真实的。如果条件为真，则
如果条件为，则计算并返回第一个子表达式
False，则计算并返回第二个子表达式。因为
万花筒有副作用，这种行为对指甲很重要
放下。

既然我们知道了我们“想要”什么，让我们把它分解成它的
成分片。

### IF/THEN/ELSE的词法分析器扩展

词法分析器扩展很简单。首先，我们添加新的枚举值
对于相关令牌：

```c++
// control
tok_if = -6,
tok_then = -7,
tok_else = -8,
```

一旦我们有了它，我们就可以识别词法分析器中的新关键字。这是
非常简单的东西：

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

现在我们已经有了来自词法分析器的相关令牌，并且我们已经
要构建的AST节点，我们的解析逻辑相对简单。
首先，我们定义一个新的解析函数：

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

现在我们让它解析并构建AST，最后一部分是
添加LLVM代码生成支持。这是最有趣的部分
If/Then/Else示例，因为这是它开始的地方
引入新概念。上面的所有代码都是彻底的
在前面的章节中描述过。

为了激发我们想要生成的代码，让我们看一下简单的
举个例子。请考虑：

    extern foo();
    extern bar();
    def baz(x) if x then foo() else bar();

如果禁用优化，您(很快)将获得的代码
万花筒看起来像这样：

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

若要可视化控制流图，可以使用
llvm\‘[选项](https://llvm.org/cmds/opt.html)\’工具。如果你把这个
LLVM IR进入\“t.ll\”并运行
\“`llvm-as<t.ll|opt-analyze-view-cfg`\”，[将弹出一个窗口
up](../../ProgrammersManual.html#viewing-graphs-while-debugging-code)
你会看到这个图表：

！[示例配置](LangImpl05-cfg.png){.ign-center}

另一种方法是调用\“`F->viewCFG()`\”或
\“`F->viewCFGOnly()`\”(其中F是\“`函数*`\”)
将实际调用插入代码并重新编译或调用这些
在调试器中。LLVM有许多很好的功能，可以可视化各种
图表。

回到生成的代码，它相当简单：条目挡路
计算条件表达式(在我们的示例中为\“x\”)，并
使用\“`fcmp one`\”指令(\‘one\’)将结果与0.0进行比较
是\“有序且不相等\”)。基于该表达式的结果，
代码跳转到\“THEN\”或\“ELSE\”块，这两个块包含
真/假大小写的表达式。

THEN/ELSE块执行完毕后，它们都会返回分支
添加到\‘ifcont\’挡路以执行
如果/然后/否则。在这种情况下，剩下的唯一要做的就是返回到
函数的调用方。然后问题就变成了：代码是如何
知道要返回哪个表达式吗？

这个问题的答案涉及一个重要的SSA操作：
[φ]
operation](http://en.wikipedia.org/wiki/Static_single_assignment_form).
如果您不熟悉SSA，[维基百科
article](http://en.wikipedia.org/wiki/Static_single_assignment_form)是
一个很好的介绍，还有各种各样的其他介绍
在您最喜欢的搜索引擎上可用。简而言之，
PHI操作的\“执行\”需要\“记住\”哪个挡路
控制权来自于。Phi运算采用与以下各项相对应的值
输入控件挡路。在这种情况下，如果控制权来自
\“然后\”挡路，则获取\“calltmp\”的值。如果控制权来自
否则，它将获取\“calltmp1\”的值。

在这一点上，你可能会开始想，“哦，不！这意味着我的
简单而优雅的前端必须开始生成SSA表单
才能使用LLVM！\“。幸运的是，情况并非如此，我们强烈要求
建议*不*在您的
前端，除非有非常好的理由这样做。在……里面
实际上，代码中有两种浮动的值
为您可能需要的普通命令式编程语言编写的
PHI节点：

涉及用户变量的代码：`x=1；x=x+1；`
隐含在AST结构中的值，如
在本例中为Phi节点。

在本教程的[第7章](LangImpl07.html)中(\“mutable
变量\“)，我们将深入讨论#1。现在，请相信我
您不需要SSA构造来处理此情况。
可以选择使用我们将在#1中描述的技术，或者
如果方便，可以直接插入Phi节点。在这种情况下，它是
生成φ节点非常容易，所以我们选择直接生成。

好了，动机和概述到此为止，让我们生成代码吧！

### IF/THEN/ELSE的代码生成

为了生成代码，我们实现了`codegen`方法
对于`IfExprAST`：

```c++
Value *IfExprAST::codegen() {
  Value *CondV = Cond->codegen();
  if (!CondV)
    return nullptr;

  // Convert condition to a bool by comparing non-equal to 0.0.
  CondV = Builder.CreateFCmpONE(
      CondV, ConstantFP::get(TheContext, APFloat(0.0)), "ifcond");
```

这段代码简单明了，与我们之前看到的类似。我们会散发出
条件的表达式，然后将该值与零进行比较以获得
1位(布尔值)形式的真值。

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

此代码创建与IF/THEN/ELSE相关的基本块
语句，并直接对应于上面示例中的块。
第一行获取正在构建的当前函数对象。它
通过向生成器请求当前的BasicBlock，然后
该挡路用于其\“父\”(它当前嵌入的函数
进入)。

一旦有了它，它就会创建三个块。请注意，它通过了
\“TheFunction\”添加到\“THEN\”挡路的构造函数中。这会导致
自动将新挡路插入到
指定的函数。其他两个块已创建，但未创建
但仍被插入到函数中。

一旦创建了块，我们就可以发出条件分支，该分支
在他们之间做出选择。请注意，创建新块不会隐式
影响IRBuilder，因此它仍在插入到挡路中
情况开始恶化。另请注意，它正在创建到
\“然后\”挡路和\“其他\”挡路，即使\“其他\”挡路
还没有插入到函数中。这一切都没问题：它是
LLVM支持正向引用的标准方式。

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

插入条件分支后，我们移动构建器以启动
插入到\“然后\”挡路中。严格说来，这通电话
插入点位于指定挡路的末尾。然而，
由于\“THEN\”挡路为空，因此它也从插入
“挡路”的开局之年。：)

一旦设置了插入点，我们递归地编码生成\“THEN\”
来自AST的表达式。为了完成\“THEN\”挡路，我们创建一个
无条件将分支机构合并为挡路。一个有趣的(而且非常
重要)LLVM IR的一个方面是，它`要求所有基本块都被“终止”<functionstructure>`{.expreted-text
Role=“ref”}带有`控制流指令<Terminators>`{.解释文本角色=“ref”}，例如
返回或分支。这意味着所有控制流，*包括秋天在内
直通*必须在llvmIR中明确表示。如果你违反了这一点
规则，验证器将发出错误。

这里的最后一行相当微妙，但非常重要。最基本的
问题是，当我们在合并的挡路中创建Phi节点时，我们需要
设置挡路/值对，以指示PHI将如何工作。
重要的是，Phi节点希望每个前置任务都有一个条目
挡路在中央人民政府的地位。那么，为什么我们要买现在的挡路呢？
我们刚把它设为上面5行的ThenBB吗？问题是\“然后\”
表达式本身实际上可能会改变构建器所在的挡路
发送到IF，例如，它包含嵌套的\“IF/THEN/ELSE\”
表达式。因为递归调用`codegen()`可能会任意
改变现在挡路的观念，要求我们得到一个
将设置Phi节点的代码的最新值。

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

Else挡路的代码生成与代码生成基本相同
去吃\‘然后\’挡路。唯一显著的区别是第一个
行，它将\‘Else\’挡路添加到函数中。先前召回
已创建\‘Else\’挡路，但未将其添加到函数。现在
发出\‘Then\’和\‘Else\’块后，我们可以使用
合并代码：

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

这里的前两行现在很熟悉：第一行添加了\“merge\”
挡路连接到函数对象(它以前是浮动的，就像其他
挡路(上图)。第二个更改插入点，以便新的
创建的代码将进入\“合并\”挡路。一旦完成，我们
需要创建PHI节点并设置挡路/值对
菲。

最后，CodeGen函数返回φ节点作为计算值
IF/THEN/ELSE表达式。在上面的示例中，返回了
值将提供给顶级函数的代码，该代码将
创建返回指令。

总体而言，我们现在能够在中执行条件代码
万花筒。有了这个扩展，万花筒是一个相当完整的
可以计算各种数值函数的语言。下一个就是
我们将添加另一个熟悉的有用表达式
非函数式语言\.

## \‘for\’循环表达式

现在我们知道了如何将基本的控制流结构添加到
语言，我们有工具来添加更强大的东西。让我们来添加
一些更具攻击性的东西，一个\‘for\’的表达：

    extern putchard(char);
    def printstar(n)
      for i = 1, i < n, 1.0 in
        putchard(42);  # ascii 42 = '*'
    
    # print 100 '*' characters
    printstar(100);

该表达式定义了一个新变量(在本例中为\“i\”)，该变量
从起始值迭代，而条件(此条件中的\“i\<n\”
情况)为真，则递增一个可选的步长值(在此为\“1.0\”
案例)。如果省略步长值，则默认为1.0。而循环
为真，则它执行其正文表达式。因为我们没有
如果要返回更好的内容，我们将一如既往地定义循环
返回0.0。将来当我们有可变变量时，它会得到
更有用。

就像以前一样，让我们来谈谈我们需要对万花筒进行哪些改变
支持这一点。

### \‘for\’循环的词法分析器扩展

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

### \‘for\’循环的AST扩展

AST节点也同样简单。它基本上可以归结为捕获
节点中的变量名称和组成表达式。

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

### \‘for\’循环的解析器扩展

解析器代码也相当标准。这里唯一有趣的是
是对可选步长值的处理。解析器代码通过以下方式处理它
检查第二个逗号是否存在。如果不是，则设置步骤
在AST节点中将值设置为NULL：

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

### \‘for\’循环的LLVM IR

现在我们来看好的部分：我们要为此生成的LLVM IR
一件事。通过上面的简单示例，我们获得此LLVM IR(请注意
为清楚起见，生成此转储时禁用了优化)：

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

这个循环包含我们之前看到的所有相同的结构：一个φ节点，
几个表达式和一些基本块。让我们看看这件衣服合不合适
在一起。

### \‘for\’循环的代码生成

codegen的第一部分非常简单：我们只输出开始
循环值的表达式：

```c++
Value *ForExprAST::codegen() {
  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen();
  if (!StartVal)
    return nullptr;
```

解决了这个问题后，下一步是设置LLVM Basic
挡路为循环体的起点。在上面的情况下，整个循环
Body是一个挡路，但请记住Body代码本身可以包括
多个块(例如，如果它包含IF/THEN/ELSE或FOR/IN
表达式)。

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

此代码类似于我们在if/Then/Else中看到的代码。因为我们会的
需要它来创建Phi节点，我们记得失败的挡路
进入循环。一旦有了它，我们就创建了真正的挡路，它从
循环，并为过渡创建无条件分支
两个街区。

```c++
// Start insertion in LoopBB.
Builder.SetInsertPoint(LoopBB);

// Start the PHI node with an entry for Start.
PHINode *Variable = Builder.CreatePHI(Type::getDoubleTy(TheContext),
                                      2, VarName.c_str());
Variable->addIncoming(StartVal, PreheaderBB);
```

现在已经设置了循环的\“preheader\”，我们将切换到发射
循环体的代码。首先，我们移动插入点并
为循环感应变量创建PHI节点。因为我们已经
知道起始值的传入值，我们将其添加到φ
节点。请注意，phi最终将获得
Backedge，但是我们还不能设置它(因为它不存在！)

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

现在代码开始变得更有趣了。我们的\‘for\’循环引入了
符号表中的新变量。这意味着我们的符号表可以
现在包含函数参数或循环变量。为了处理这件事，
在对循环体进行编码生成之前，我们将循环变量添加为
其名称的当前值。请注意，可能存在一个
外部作用域中的同名变量。它会很容易做出来
这是一个错误(如果已经存在错误，则发出错误并返回NULL
条目)，但我们选择允许跟踪变量。在……里面
为了正确处理这个问题，我们要记住我们的价值
可能在`OldVal`中隐藏(如果没有，则为NULL
阴影变量)。

一旦将循环变量设置到符号表中，代码
递归地编码生成主体。这允许主体使用循环
变量：任何对它的引用都会自然地在符号中找到它
桌子。

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

既然主体已经发出，我们将计算迭代的下一个值
变量，如果不存在，则添加步长值或1.0。
\‘`NextVar`\’将是下一个循环变量的值
循环的迭代。

```c++
// Compute the end condition.
Value *EndCond = End->codegen();
if (!EndCond)
  return nullptr;

// Convert condition to a bool by comparing non-equal to 0.0.
EndCond = Builder.CreateFCmpONE(
    EndCond, ConstantFP::get(TheContext, APFloat(0.0)), "loopcond");
```

最后，我们评估循环的退出值，以确定是否
循环应该退出。这反映了
IF/THEN/ELSE语句。

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

循环主体的代码完成后，我们只需要完成
提高它的控制流。此代码记住结尾挡路(用于
φ节点)，然后为循环出口创建挡路(\“后循环\”)。
根据退出条件的值，它创建一个条件
分支，该分支选择是再次执行循环还是退出
循环。任何将来的代码都会在\“After Loop\”挡路中发出，因此它设置
它的插入位置。

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

最后的代码处理各种清理：现在我们有了
\“NextVar\”值，我们可以将传入的值添加到循环PHI节点。
之后，我们从符号表中删除循环变量，以便
它不在for循环之后的作用域内。
for循环总是返回0.0，因此这就是我们从中返回的内容
`ForExprAST：：codegen()`。

至此，我们得出了\“向万花筒添加控制流\”的结论
本教程的一章。在本章中，我们添加了两个控制流
构造，并用它们激发LLVM IR的几个方面
这些对于前端实现者来说非常重要。在接下来的时间里
在我们传奇的章节中，我们将变得更疯狂，并添加[User-Defined
操作员](LangImpl06.html)到我们可怜的无辜语言。

## 完整代码列表

以下是我们的运行示例的完整代码清单，增强了
IF/THEN/ELSE和FOR表达式。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

：{.writalinclude language=“c++”}
../examples/Kaleidoscope/Chapter5/toy.cpp
**：

[下一步：扩展语言：自定义运算符](LangImpl06.html)
