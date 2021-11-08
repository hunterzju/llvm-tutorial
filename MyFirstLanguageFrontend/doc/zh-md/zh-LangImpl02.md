# Kaleidoscope：实现解析器和AST

## 第二章绪论

欢迎阅读“[使用LLVM实现语言](index.md)”教程的第2章。本章将向您展示如何使用[第1章](LangImpl01.md)中内置的词法分析器为我们的Kaleidoscope语言构建一个完整的[parser](http://en.wikipedia.org/wiki/Parsing)。一旦我们有了解析器，我们将定义并构建一个[抽象语法树](http://en.wikipedia.org/wiki/Abstract_syntax_tree)(AST)]。

我们将构建的解析器结合使用[递归下降Parsing](http://en.wikipedia.org/wiki/Recursive_descent_parser)]和[运算符优先Parsing](http://en.wikipedia.org/wiki/Operator-precedence_parser)]来解析Kaleidoscope语言(后者用于二进制表达式，前者用于其他所有内容)。在我们开始解析之前，让我们先谈谈解析器的输出：抽象语法树。

## 抽象语法树(AST)

程序的AST捕捉了程序行为，以便编译器后期阶段(例如代码生成)进行解释。基本上，我们希望语言中的每个构造（construct)都有一个对象，并且AST应该紧密地对语言进行建模。在Kaleidoscope中，我们有表达式、原型和函数对象。我们先从表达式开始：

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

上面的代码显示了ExprAST基类和一个用于数字文本的子类的定义。关于此代码需要注意的重要一点是，NumberExprAST类将文字的数值捕获为实例变量。这允许编译器的后续阶段知道存储的数值是什么。

现在我们只创建AST，所以没有创建有用的访问方法。例如，可以很容易地添加一个虚拟方法来漂亮地打印代码。下面是我们将在Kaleidoscope语言的基本形式中使用的其他表达式AST节点定义：

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

这一切(有意地)相当直观：变量捕获变量名，二元操作符捕获它们的操作码(例如，‘+’)，调用捕获函数名以及任何参数表达式的列表。我们的AST有一点很好，那就是它捕获了语言特性，而不涉及语言的语法。请注意，这里没有讨论二元运算符的优先级、词汇结构等。

对于我们的基础语言，这些都是我们将要定义的表达式节点。因为它没有条件控制流，所以它不是图灵完备的；我们将在后面的文章中修复这一点。接下来我们需要的两件事是一种表示函数接口的方式，以及一种表示函数本身的方式：

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

在Kaleidoscope中，函数的类型化只需对其参数进行计数。因为所有的值都是双精度浮点数，所以每个参数的类型不需要存储在任何地方。在更激进、更现实的语言中，“ExprAST”类可能会有一个类型字段。

有了这个脚手架，我们现在可以讨论在Kaleidoscope中解析表达式和函数体。

## 解析器基础

现在我们有一个AST要构建，我们需要定义解析器代码来构建它。这里的想法是，我们希望将类似“x+y”的内容(由词法分析器返回为三个令牌)解析为一个AST，该AST可以通过如下调用生成：

```c++
auto LHS = std::make_unique<VariableExprAST>("x");
auto RHS = std::make_unique<VariableExprAST>("y");
auto Result = std::make_unique<BinaryExprAST>('+', std::move(LHS),
                                              std::move(RHS));
```

为了做到这一点，我们将从定义一些基本的辅助例程开始：

```c++
/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() {
  return CurTok = gettok();
}
```

这在词法分析器周围实现了一个简单的令牌缓冲区。这允许我们提前查看词法分析器返回的内容。我们解析器中的每个函数都假定CurTok是需要解析的当前令牌。

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

`LogError`例程是简单的辅助例程，我们的解析器将使用它来处理错误。我们的解析器中的错误恢复不会是最好的，也不是特别用户友好的，但是对于我们的教程来说已经足够了。这些例程可以更容易地处理具有各种返回类型的例程中的错误：它们总是返回NULL。

有了这些基本的帮助器函数，我们就可以实现语法的第一部分：数字文本。

## 基本表达式解析

我们从数字文字开始，因为它们是最容易处理的。对于语法中的每个产生式，我们将定义一个函数来解析该产生式（production）。对于数字文字，我们有：

```c++
/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}
```

此例程非常简单：它预期在当前令牌为`tok_number`令牌时被调用。它接受当前的数字值，创建一个`NumberExprAST‘节点，将词法分析器前进到下一个令牌，最后返回。

这其中有一些有趣的方面。最重要的一点是，该例程会吃掉与源码相对应的所有标记，并返回词法分析器缓冲区，其中下一个标记(不是语法产生式的一部分)已准备就绪。对于递归下降解析器来说，这是一种相当标准的方式。举个更好的例子，圆括号运算符定义如下：

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

此函数说明了有关解析器的许多有趣的事情：

1)它显示了我们如何使用LogError例程。调用此函数时，该函数期望当前令牌是一个‘(’令牌，但在解析子表达式之后，可能没有‘)’在等待。例如，如果用户键入“(4x”而不是“(4)”)，解析器应该会发出错误。因为错误可能会发生，所以解析器需要一种方式来指示它们已经发生：在我们的解析器中，我们对错误返回NULL。

2)此函数的另一个有趣之处在于，它通过调用`ParseExpression`使用递归(我们很快就会看到`ParseExpression`可以调用`ParseParenExpr`)。这是非常强大的，因为它允许我们处理递归语法，并使每个产生式都非常简单。请注意，括号本身不会导致构造AST节点。虽然我们可以这样做，但是圆括号最重要的作用是引导解析器并提供分组。一旦解析器构造了AST，就不需要括号了。

下一个简单的例程用于处理变量引用和函数调用：

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

此例程遵循与其他例程相同的样式。(如果当前Token是`tok_Identifier`令牌，则预期会被调用)。它还具有递归和错误处理功能。其中一个有趣的方面是，它使用*前瞻（look ahead)*来确定当前标识符是独立变量引用还是函数调用表达式。它通过检查标识符之后的令牌是否是‘(’令牌来处理此问题，根据需要构造`VariableExprAST`或`CallExprAST`节点。

现在我们已经准备好了所有简单的表达式解析逻辑，我们可以定义一个辅助函数来将其包装到一个入口点中。我们将这类表达式称为“主（Primary）”表达式，原因在[后续第6章教程](LangImpl06.md#user-defined-unary-operators)将变得更加清楚.为了解析任意主表达式，我们需要确定它是哪种表达式：

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

现在您已经看到了该函数的定义，我们可以在各种函数中假定CurTok状态的原因就更加明显了。这使用前瞻来确定正在检查哪种类型的表达式，然后使用函数调用对其进行解析。

现在已经处理了基本表达式，我们需要处理二元表达式。它们稍微复杂一些。

## 二元表达式解析

二元表达式很难解析，因为它们通常是模棱两可的。例如，当给定字符串“x+y\*z”时，解析器可以选择将其解析为“(x+y)\*z”或“x+(y\*z)”。对于来自数学的通用定义，我们期待后面的解析，因为“\*”(乘法)的*优先顺序*高于“+”(加法)。

处理这一问题的方法有很多，但一种优雅而有效的方法是使用[操作符优先顺序解析](Operator-Prirecedence Parsing](http://en.wikipedia.org/wiki/Operator-precedence_parser).此解析技术使用二元运算符的优先级来指导递归。首先，我们需要一个优先顺序表：

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

对于Kaleidoscope的基本形式，我们将只支持4个二元运算符(这显然可以由您，我们勇敢无畏的读者来扩展)。`GetTokPrecedence`函数返回当前令牌的优先级，如果令牌不是二元运算符，则返回-1。有一个map可以方便地添加新的运算符，并清楚地表明算法不依赖于涉及的特定运算符，并且消除map并在`GetTokPrecedence`函数中进行比较也足够容易(或者只使用固定大小的数组)。

有了上面定义的辅助函数，我们现在可以开始解析二元表达式了。运算符优先解析的基本思想是将具有潜在歧义二元运算符的表达式分解为多个片段。例如，考虑表达式“a+b+(c+d)\*e\*f+g”。运算符优先解析将其视为由二元运算符分隔的主表达式流。因此，它将首先解析前导主表达式“a”，然后将看到对[+，b][+，(c+d)][\*，e][\*，f]和[+，g]。注意，因为括号是主表达式，所以二元表达式解析器根本不需要担心像(c+d)这样的嵌套子表达式。

首先，表达式可能是后面跟了一系列[binop，primary yexpr]对的主表达式：

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

`ParseBinOpRHS`是为我们解析成对序列的函数。它具有优先级和指向到目前为止已解析的部分的表达式的指针。请注意，“x”是一个完全有效的表达式：因此，允许“binoprhs”为空，在这种情况下，它返回传递给它的表达式。在上面的示例中，代码将“a”的表达式传递给`ParseBinOpRHS`，当前令牌为“+”。

传入`ParseBinOpRHS`的优先级值表示函数可以吃的*最小算子优先级*。例如，如果当前对流为[+，x]，且`ParseBinOpRHS`的优先级为40，则不会消耗任何Token(因为‘+’的优先级仅为20)。考虑到这一点，`ParseBinOpRHS`以下述代码开始：

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

此代码获取当前令牌的优先级，并检查是否太低。因为我们定义了优先级为-1的无效令牌，所以此检查隐含地知道当令牌流用完二元运算符时，对流结束。如果检查成功，我们就知道该令牌是二元运算符，并且它将包含在以下表达式中：

```c++
// Okay, we know this is a binop.
int BinOp = CurTok;
getNextToken();  // eat binop

// Parse the primary expression after the binary operator.
auto RHS = ParsePrimary();
if (!RHS)
  return nullptr;
```

因此，此代码吃掉(并记住)二元运算符，然后解析后面的主表达式。这将构建整个对，对于运行的示例，第一个对是[+，b]。

现在我们已经解析了表达式的左侧和一对RHS序列，我们必须确定表达式关联的方式。特别地，我们可以使用“(a+b)binop unparsed”或“a+(B Binop Unparsed)”。为了确定这一点，我们向前看“binop”以确定其优先级，并将其与BinOp的优先级(在本例中为‘+’)进行比较：

```c++
// If BinOp binds less tightly with RHS than the operator after RHS, let
// the pending operator take RHS as its LHS.
int NextPrec = GetTokPrecedence();
if (TokPrec < NextPrec) {
```

如果“rhs”右侧的binop的优先级低于或等于当前操作符的优先级，那么我们知道圆括号关联为“(a+b)binop.”。在我们的示例中，当前操作符是“+”，下一个操作符是“+”，我们知道它们具有相同的优先级。在本例中，我们将为“a+b”创建AST节点，然后继续解析：

```c++
... if body omitted ...
}

// Merge LHS/RHS.
LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS),
                                     std::move(RHS));
}  // loop around to the top of the while loop.
}
```

在上面的示例中，这将把“a+b+”转换为“(a+b)”，并执行循环的下一次迭代，当前令牌为“+”。上面的代码将吃掉、记住并解析“(c+d)”作为主要表达式，这使得当前对等于[+，(c+d)]。然后，它将计算上面的‘if’条件，并将“\*”作为主数据库右侧的binop。在这种情况下，优先级“\*”高于优先级“+”，因此将输入IF条件。

这里留下的关键问题是“if条件如何完全解析右侧”？特别是，要为我们的示例正确构建AST，它需要获取所有“(c+d)\*e\*f”作为RHS表达式变量。执行此操作的代码出奇地简单(以上两个块中的代码在上下文中重复)：

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

此时，我们知道PRIMARY的RHS的二元运算符比我们当前正在解析的binop具有更高的优先级。因此，我们知道运算符的优先级都高于“+”的任何对序列都应该一起解析并返回为“RHS”。为此，我们递归调用`ParseBinOpRHS`函数，将“TokPrec+1”指定为继续执行所需的最低优先级。在上面的示例中，这将导致它返回“(c+d)\*e\*f”的AST节点作为RHS，然后将其设置为‘+’表达式的RHS。

最后，在While循环的下一次迭代中，将解析“+g”片段并将其添加到AST。通过这一小段代码(14行)，我们以非常优雅的方式正确地处理了完全通用的二进制表达式解析。这是这段代码的快速浏览，有点微妙。我推荐用几个难理解的例子来看看它是如何工作的。

这就结束了表达式的处理。此时，我们可以将解析器指向任意令牌流，并从它构建表达式，在不属于表达式的第一个令牌处停止。接下来，我们需要处理函数定义等。

## 解析剩余部分

接下来缺少的是函数原型的处理。在Kaleidoscope中，它们既用于‘extern’函数声明，也用于函数体定义。执行此操作的代码简单明了，并且不是很有趣(一旦您从表达式中幸存下来)：

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

有了上述代码，解析函数定义非常简单，只需一个原型加上一个表达式来实现：

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

此外，我们还支持‘extern’声明函数，如‘sin’和‘cos’，并支持用户函数的正向声明。这些“extern”只是没有主体的原型：

```c++
/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken();  // eat extern.
  return ParsePrototype();
}
```

最后，我们还将允许用户键入任意顶层表达式并动态（译者注：原文为[on the fly](https://en.wikipedia.org/wiki/On_the_fly#Computer_usage)）对其求值。我们将通过为其定义匿名空(零参数)函数来处理此问题：

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

现在我们已经有了所有的部分，让我们构建一个小驱动程序，它将让我们真正*执行*我们已经构建的代码！

## 驱动

驱动程序只需使用顶层分派循环调用所有解析段。这里没有太多有趣的地方，所以我将只包含顶层循环。请参阅[下面](#完整代码列表)以获取“顶层解析”部分的完整代码。

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

其中最有趣的部分是我们忽略了顶层分号。你会问，为什么会这样？基本原因是，如果您在命令行键入“4+5”，解析器不知道您要键入的内容是否结束。例如，您可以在下一行键入“def Foo.”，在这种情况下，4+5是顶层表达式的末尾。或者，您也可以键入“\*6”，这将继续表达式。拥有顶层分号解析允许您键入“4+5；”，解析器可以理解您的行为。

## 结论

用不到400行注释代码(240行非注释、非空白代码)，我们完全定义了我们的最小语言，包括词法分析器、解析器和AST构建器。完成此操作后，可执行文件将验证Kaleidoscope代码，并告诉我们它在语法上是否无效。例如，下面是一个交互示例：

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

这里有很大的扩展空间。您可以定义新的AST节点，以多种方式扩展语言等。在[下一篇](LangImpl03.md)中，我们将介绍如何从AST生成LLVM中间表示(IR)。

## 完整代码列表

下面是我们的运行示例的完整代码清单。因为它使用LLVM库，所以我们需要链接它们。为此，我们使用[llvm-config](https://llvm.org/cmds/llvm-config.html)工具通知生成文件/命令行要使用哪些选项：

```bash
# Compile
clang++ -g -O3 toy.cpp `llvm-config --cxxflags`
# Run
./a.out
```

以下是代码：
[https://github.com/llvm/llvm-project/blob/main/llvm/examples/Kaleidoscope/Chapter2/toy.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/examples/Kaleidoscope/Chapter2/toy.cpp)

[下一步：实现LLVM IR代码生成](LangImpl03.md)

## 后记：心得体会
1. 抽象语法树（AST）是对语言建模的结果，这里AST分为表达式，原型（protoType）和函数三大类；
2. 语法解析的过程就是将Token构建为抽象语法树的过程；
3. 解析过程采用递归下降解析和运算符优先解析。