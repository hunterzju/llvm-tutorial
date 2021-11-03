# 万花筒：实现解析器和AST

：{.content local=“”}
**

## 第二章绪论

欢迎学习\“的第2章[使用实现语言
LLVM](index.html)\“教程。本章介绍如何使用
lexer，内置在[第1章](LangImpl01.html)中，构建完整的
用于我们的万花筒的[parser](http://en.wikipedia.org/wiki/Parsing)
语言。一旦我们有了解析器，我们将定义并构建[Abstract
语法Tree](http://en.wikipedia.org/wiki/Abstract_syntax_tree)(AST)。

我们将构建的解析器使用[递归下降]的组合
Parsing](http://en.wikipedia.org/wiki/Recursive_descent_parser)和
[操作员-优先
Parsing](http://en.wikipedia.org/wiki/Operator-precedence_parser)至
解析万花筒语言(后者用于二进制表达式和
前者适用于其他一切)。在我们开始解析之前，让我们先来看看
谈谈解析器的输出：抽象语法树。

## 抽象语法树(AST)

程序的AST捕获其行为的方式是
便于编译器的后期阶段(例如代码生成)
翻译一下。中的每个构造都需要一个对象。
语言，并且AST应该紧密地对语言进行建模。在……里面
万花筒，我们有表达式、原型和函数对象。
我们先从表达式开始：

```c++
/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() {}
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}
};
```

上面的代码显示了ExprAST基类和一个
我们用于数字文字的子类。需要注意的重要一点是
关于此代码的是，NumberExprAST类捕获数字
作为实例变量的文字值。这允许后续阶段
以了解存储的数值是什么。

现在我们只创建AST，所以没有有用的访问器
方法在他们身上。将虚拟方法添加到Pretty非常容易
例如，打印代码。下面是另一个表达式AST节点
我们将在万花筒的基本形式中使用的定义
语言：

```c++
/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(const std::string &Name) : Name(Name) {}
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
    : Op(op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
    : Callee(Callee), Args(std::move(Args)) {}
};
```

这一切(有意地)相当简单：变量捕获
变量名，二进制运算符捕获它们的操作码(例如，\‘+\’)，
并调用捕获函数名以及任何参数的列表
表情。我们的AST的一个优点是它捕获了
语言的特点，不谈语言的句法。
注意，没有讨论二元运算符的优先级，
词汇结构等。

对于我们的BASIC语言，这些是我们将要使用的所有表达式节点
定义。因为它没有条件控制流，所以它没有
图灵-完成；我们将在稍后的一期文章中解决这一问题。
我们接下来需要的是一种谈论函数接口的方法，以及
谈论函数本身的方式：

```c++
/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;

public:
  PrototypeAST(const std::string &name, std::vector<std::string> Args)
    : Name(name), Args(std::move(Args)) {}

  const std::string &getName() const { return Name; }
};

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
    : Proto(std::move(Proto)), Body(std::move(Body)) {}
};
```

在万花筒中，函数的类型仅使用其
争论。由于所有值都是双精度浮点数，因此
每个参数的类型不需要存储在任何地方。
具有攻击性和现实主义的语言，\“ExprAST\”类可能会
有一个类型字段。

有了这个脚手架，我们现在可以讨论解析表达式和
万花筒中的功能体。

## 解析器基础

现在我们有了一个要构建的AST，我们需要定义解析器代码
把它建起来。这里的想法是，我们希望解析类似于\“x+y\”的内容
(词法分析器将其作为三个令牌返回)到一个AST中，该AST可以
使用如下调用生成：

```c++
auto LHS = std::make_unique<VariableExprAST>("x");
auto RHS = std::make_unique<VariableExprAST>("y");
auto Result = std::make_unique<BinaryExprAST>('+', std::move(LHS),
                                              std::move(RHS));
```

为了做到这一点，我们将从定义一些基本的帮助器开始
例程：

```c++
/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() {
  return CurTok = gettok();
}
```

这在词法分析器周围实现了一个简单的令牌缓冲区。这使得我们可以
超前查看词法分析器返回的内容的一个标记。每个函数
在我们的解析器中，将假设CurTok是需要
被解析。

```c++
/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const char *Str) {
  fprintf(stderr, "LogError: %s\n", Str);
  return nullptr;
}
std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
  LogError(Str);
  return nullptr;
}
```

`LogError`例程是简单的帮助器例程，我们的解析器将
用于处理错误。我们的解析器中的错误恢复将不是
最好的，并且不是特别的用户友好，但它将足以满足我们的
教程。这些例程使处理例程中的错误变得更容易
它们具有各种返回类型：它们总是返回NULL。

有了这些基本的帮助器函数，我们就可以实现
我们的语法是：数字文字。

## 基本表达式解析

我们从数字文字开始，因为它们是最简单的
进程。对于我们语法中的每个结果，我们都会定义一个函数
它解析该产品。对于数字文字，我们有：

```c++
/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}
```

此例程非常简单：它预计在当前
Token为`tok_number`令牌。它采用当前数值，
创建`NumberExprAST‘节点，将词法分析器前进到下一个令牌，
最后又回来了。

这其中有一些有趣的方面。最重要的一个是
此例程会吃掉与
并返回带有下一个令牌的词法分析器缓冲区(即
不是语法作品的一部分)准备好了。这是一个相当不错的
递归下降解析器的标准方法。举个更好的例子，
圆括号运算符的定义如下：

```c++
/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("expected ')'");
  getNextToken(); // eat ).
  return V;
}
```

此函数说明了有关
解析器：

1\)它显示了我们如何使用LogError例程。调用时，此
函数要求当前令牌是\‘(\’令牌，但在
解析子表达式，可能没有\‘)\’
在等着呢。例如，如果用户键入\“(4 x\”而不是\“(4)\”，
解析器应该会发出错误。因为可能会发生错误，所以解析器
需要一种方法来指示它们已经发生：在我们的解析器中，我们返回
出现错误时为空。

2\)此函数的另一个有趣方面是它使用
通过调用`ParseExpression`进行递归(我们很快就会看到
`ParseExpression`可以调用`ParseParenExpr`)。这很强大，因为
它允许我们处理递归语法，并保留每个结果
非常简单。请注意，括号不会导致构造AST
节点本身。虽然我们可以这样做，但最重要的角色
括号的作用是指导解析器并提供分组。一旦
解析器构造AST，不需要括号。

下一个简单的结果是处理变量引用和
函数调用：

```c++
/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = IdentifierStr;

  getNextToken();  // eat identifier.

  if (CurTok != '(') // Simple variable ref.
    return std::make_unique<VariableExprAST>(IdName);

  // Call.
  getNextToken();  // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (1) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return std::make_unique<CallExprAST>(IdName, std::move(Args));
}
```

此例程遵循与其他例程相同的样式。(它预计
如果当前令牌是`tok_Identifier`令牌则调用)。它还
具有递归和错误处理功能。其中一个有趣的方面是
它使用*前瞻*来确定当前标识符是否为展台
单独的变量引用，或者如果它是函数调用表达式。它
通过检查标识符后面的标记是否为
\‘(\’令牌，构造`VariableExprAST`或`CallExprAST`
节点(视情况而定)。

现在我们已经准备好了所有简单的表达式解析逻辑，我们
可以定义帮助器函数以将其包装到一个入口点中。
出于某些原因，我们将这类表达式称为\“主\”表达式
这一点将变得更加清楚[稍后将在
tutorial](LangImpl06.html#user-defined-unary-operators).为了
解析任意主表达式，我们需要确定哪种类型的
表达式是：

```c++
/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
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
  }
}
```

现在您已经看到了该函数的定义，原因就更加明显了
我们可以假设CurTok在各种函数中的状态。这是用来
前瞻以确定正在检查哪种类型的表达式，以及
然后使用函数调用对其进行解析。

现在已经处理了基本表达式，我们需要处理二进制
表情。它们稍微复杂一些。

## 二进制表达式解析

二进制表达式很难解析，因为它们
经常是模棱两可的。例如，当给定字符串\“x+y\*z\”时，
解析器可以选择将其解析为\“(x+y)\*z\”或\“x+(y\*z)\”。
有了来自数学的共同定义，我们期待后面的解析，
因为\“\*\”(乘法)的*优先顺序*高于\“+\”
(附加)。

处理这一问题的方法有很多种，但一种优雅而有效的方法是
要使用[运算符-优先级
Parsing](http://en.wikipedia.org/wiki/Operator-precedence_parser).这
解析技术使用二元运算符的优先级来指导
递归。首先，我们需要一个优先顺序表：

```c++
/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
  if (!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0) return -1;
  return TokPrec;
}

int main() {
  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40;  // highest.
  ...
}
```

对于万花筒的基本形式，我们将只支持4个二进制
操作员(这显然可以由您来扩展，我们勇敢无畏的
读者)。`GetTokPrecedence`函数返回
当前令牌，如果该令牌不是二元运算符，则返回-1。有一张地图
使添加新运算符变得容易，并清楚地表明该算法
不依赖于所涉及的特定操作员，但这会很容易
足以消除地图并在
`GetTokPrecedence`函数。(或者只使用固定大小的数组)。

有了上面定义的帮助器，我们现在可以开始解析二进制文件了
表情。运算符优先解析的基本思想是中断
将具有可能不明确的二元运算符的表达式向下转换为
碎片。例如，考虑表达式\“a+b+(c+d)\*e\*f+g\”。
运算符优先解析将其视为主数据流
由二元运算符分隔的表达式。因此，它将首先解析
前导主表达式\“a\”，则它将看到对\[+，
B\]\[+，(c+d)\]\[\*，e\]\[\*，f\]和\[+，g\]。请注意，因为
圆括号是主表达式，二进制表达式解析器
根本不需要担心像(c+d)这样的嵌套子表达式。

首先，表达式是可能后跟的主表达式
\[binop，primary yexpr\]对的序列：

```c++
/// expression
///   ::= primary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression() {
  auto LHS = ParsePrimary();
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}
```

`ParseBinOpRHS`是解析以下项的配对序列的函数
我们。它采用优先级和指向部件表达式的指针
到目前为止已经被解析过了。请注意，\“x\”是完全有效的
表达式：因此，\“binoprhs\”允许为空，在这种情况下
它返回传递给它的表达式。在上面的示例中，
代码将\“a\”的表达式传递给`ParseBinOpRHS`，并且
当前令牌为\“+\”。

传入`ParseBinOpRHS`的优先级值表示*极小值
该函数被允许吃的运算符优先级*。例如,
如果当前对流为\[+，x\]并且传入`ParseBinOpRHS`
优先级为40，则不会消耗任何令牌(因为
\‘+\’的优先级仅为20)。考虑到这一点，`ParseBinOpRHS`
开头为：

```c++
/// binoprhs
///   ::= ('+' primary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS) {
  // If this is a binop, find its precedence.
  while (1) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;
```

此代码获取当前令牌的优先级，并检查是否
如果太低了。因为我们定义的无效令牌的优先级为
，则该检查隐含地知道当-1\f25-1\f25 Token-1\f6(令牌)
流耗尽了二元运算符。如果这次检查成功，我们就知道
该令牌是二元运算符，并且它将包含在此
表达式：

```c++
// Okay, we know this is a binop.
int BinOp = CurTok;
getNextToken();  // eat binop

// Parse the primary expression after the binary operator.
auto RHS = ParsePrimary();
if (!RHS)
  return nullptr;
```

因此，此代码吃掉(并记住)二元运算符，然后
分析后面的主表达式。这就构成了整个
对，对于运行的示例，第一个是\[+，b\]。

现在我们解析了表达式的左侧和一对
RHS序列，我们必须决定表达式关联的方式。
特别地，我们可以使用\“(a+b)binop unparsed\”或\“a+(b binop
未解析)\“。为了确定这一点，我们向前看\”binop\“以确定
它的优先级，并将其与BinOp的优先级(在中为\‘+\’)进行比较
本案)：

```c++
// If BinOp binds less tightly with RHS than the operator after RHS, let
// the pending operator take RHS as its LHS.
int NextPrec = GetTokPrecedence();
if (TokPrec < NextPrec) {
```

如果\“rhs\”右侧的binop的优先级较低或相等
设置为当前运算符的优先级，则我们知道
圆括号关联为\“(a+b)binop\.\”。在我们的示例中，
当前运算符是\“+\”，下一个运算符是\“+\”，我们知道
他们有相同的优先顺序。在本例中，我们将创建AST节点
对于\“a+b\”，然后继续解析：

```c++
... if body omitted ...
}

// Merge LHS/RHS.
LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS),
                                     std::move(RHS));
}  // loop around to the top of the while loop.
}
```

在上面的示例中，这将把\“a+b+\”转换为\“(a+b)\”并执行
循环的下一次迭代，当前令牌为\“+\”。这个
上面的代码将读取、记忆和解析\“(c+d)\”作为主要
表达式，它使当前对等于\[+，(c+d)\]。它会的
然后使用\“\*\”作为binop对上面的\‘if\’条件求值
初选的权利。在这种情况下，\“\*\”的优先级为
高于优先级\“+\”，因此将输入IF条件。

这里留下的关键问题是\“if条件如何解析
完全右手边\“？特别是要正确构建AST
在我们的示例中，它需要获取所有\“(c+d)\*e\*f\”作为RHS
表达式变量。执行此操作的代码出奇地简单(代码
从针对上下文复制的上述两个块中)：

```c++
// If BinOp binds less tightly with RHS than the operator after RHS, let
// the pending operator take RHS as its LHS.
int NextPrec = GetTokPrecedence();
if (TokPrec < NextPrec) {
  RHS = ParseBinOpRHS(TokPrec+1, std::move(RHS));
  if (!RHS)
    return nullptr;
}
// Merge LHS/RHS.
LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS),
                                       std::move(RHS));
}  // loop around to the top of the while loop.
}
```

在这一点上，我们知道我们的RHS的二元运算符
PRIMARY的优先级高于我们当前正在解析的binop。
因此，我们知道任何运算符都是
优先级高于\“+\”的应一起解析并返回为
\“RHS\”。为此，我们递归调用`ParseBinOpRHS`函数
指定\“TokPrec+1\”作为其执行以下操作所需的最低优先级
继续。在上面的示例中，这将导致它返回AST
节点\“(c+d)\*e\*f\”作为RHS，然后将其设置为
\‘+\’表达式。

最后，在While循环的下一次迭代中，\“+g\”部分是
已解析并添加到AST。使用这一小段代码(14
非平凡行)，我们正确地处理了完全通用的二进制表达式
以一种非常优雅的方式进行解析。这是一次旋风式的代码之旅，
这有点微妙。我建议带几个人浏览一下。
难的例子来看看它是如何工作的。

这就结束了表达式的处理。在这一点上，我们可以将
解析任意令牌流并从其构建表达式，
在不是表达式一部分的第一个令牌处停止。下一个就是
我们需要处理函数定义等。

## 解析睡觉

接下来缺少的是函数原型的处理。在……里面
万花筒，它们都用于\‘外部\’函数声明
以及函数体定义。执行此操作的代码为
直截了当，不是很有趣(一旦你活了下来
表达式)：

```c++
/// prototype
///   ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  if (CurTok != tok_identifier)
    return LogErrorP("Expected function name in prototype");

  std::string FnName = IdentifierStr;
  getNextToken();

  if (CurTok != '(')
    return LogErrorP("Expected '(' in prototype");

  // Read the list of argument names.
  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken();  // eat ')'.

  return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}
```

鉴于此，函数定义非常简单，只需一个原型加
实现正文的表达式：

```c++
/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken();  // eat def.
  auto Proto = ParsePrototype();
  if (!Proto) return nullptr;

  if (auto E = ParseExpression())
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}
```

此外，我们还支持\‘extern\’声明函数，如\‘sin\’和
\‘CoS\’以及支持用户函数的正向声明。
这些“外部”只是没有主体的原型：

```c++
/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken();  // eat extern.
  return ParsePrototype();
}
```

最后，我们还将允许用户键入任意的顶级内容
并在飞翔上对它们进行评估。我们将通过以下方式处理这件事
为它们定义匿名空(零参数)函数：

```c++
/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
    auto Proto = std::make_unique<PrototypeAST>("", std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}
```

现在我们已经有了所有的组件，让我们构建一个小驱动程序，它将
让我们真正*执行*我们已经构建的代码！

## 司机

此操作的驱动程序只需使用
顶级调度循环。这里没有太多有趣的东西，所以我就
只需包含顶级循环即可。参见[下文](#Full-code-Listing)了解
\“顶级解析\”部分中的完整代码。

```c++
/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (1) {
    fprintf(stderr, "ready> ");
    switch (CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_def:
      HandleDefinition();
      break;
    case tok_extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
  }
}
```

其中最有趣的部分是我们忽略了顶层
分号。你会问，为什么会这样？基本原因是如果您键入
\“4+5\”在命令行中，解析器不知道这是否
你要打字或不打字的结束语。例如，在下一行中您可以
可以键入\“def Foo\.\”，在这种情况下，4+5是顶级
表达式。或者，您也可以键入\“\*6\”，这将继续
这个表情。使用顶级分号可以键入\“4+5；\”，
解析器就会知道您已经完成了。

## 结论

在略低于400行注释代码的情况下(240行非注释代码，
非空代码)，我们完全定义了我们的最小语言，包括一个
词法分析器、解析器和AST构建器。完成此操作后，可执行文件将
验证万花筒代码并告诉我们它在语法上是否无效。
例如，下面是一个交互示例：

```bash
$ ./a.out
ready> def foo(x y) x+foo(y, 4.0);
Parsed a function definition.
ready> def foo(x y) x+y y;
Parsed a function definition.
Parsed a top-level expr
ready> def foo(x y) x+y );
Parsed a function definition.
Error: unknown token when expecting an expression
ready> extern sin(a);
ready> Parsed an extern
ready> ^D
$
```

这里有很大的扩展空间。您可以定义新的AST节点，
以多种方式扩展语言，等等。在[Next
LLVM](LangImpl03.html)，我们将介绍如何生成LLVM
来自AST的中间表示(IR)。

## 完整代码列表

下面是我们的运行示例的完整代码清单。因为这个
使用LLVM库，我们需要链接它们。为此，我们使用
[llvm-config](https://llvm.org/cmds/llvm-config.html)工具用于通知我们的
有关使用哪些选项的Makefile/命令行：

```bash
# Compile
clang++ -g -O3 toy.cpp `llvm-config --cxxflags`
# Run
./a.out
```

以下是代码：

：{.writalinclude language=“c++”}
../examples/Kaleidoscope/Chapter2/toy.cpp
**：

[下一步：实现LLVM IR代码生成](LangImpl03.html)
