# Kaleidoscope：扩展语言：用户定义运算符

## 第六章绪论

欢迎阅读\“[使用LLVM实现语言](index.html)\”教程的第6章。在本教程的这一点上，我们现在已经有了一种功能齐全的语言，它相当简单，但也很有用。然而，它仍然有一个很大的问题。我们的语言没有很多有用的运算符(比如除法、逻辑否定，甚至除了小于之外的任何比较)。

本教程的这一章将离题介绍如何将用户定义的运算符添加到简单而漂亮的Kaleidoscope语言中。这种离题在某些方面给了我们一种简单而丑陋的语言，但同时也给了我们一种强有力的语言。创造自己的语言的一大好处就是你可以决定什么是好的，什么是坏的。在本教程中，我们将假设将其用作展示一些有趣的解析技术的一种方式是可以的。

在本教程的最后，我们将介绍一个示例的Kaleidoscope应用程序，该应用程序[渲染Mandelbrot集](#踢轮胎)。这给出了一个使用Kaleidoscope及其功能集可以构建的示例。

## 用户定义运算符：理念

我们将添加到Kaleidoscope中的\“运算符重载\”比在C++等语言中的\“运算符重载\”更通用。在C++中，您只允许重新定义现有操作符：您不能以编程方式更改语法、引入新操作符、更改优先级别等。在本章中，我们将向Kaleidoscope添加此功能，这将允许用户对所支持的操作符集合进行四舍五入。

在这样的教程中介绍用户定义的运算符的目的是展示使用手写解析器的功能和灵活性。到目前为止，我们已经实现的解析器对语法的大部分使用递归下降，对表达式使用运算符优先解析。详见[第2章](LangImpl02.html)。通过使用运算符优先解析，很容易允许程序员在语法中引入新的运算符：随着JIT的运行，语法是动态可扩展的。

我们要添加的两个特定功能是可编程的一元运算符(目前，Kaleidoscope根本没有一元运算符)以及二元运算符。例如：
```
    # Logical unary not.
    def unary!(v)
      if v then
        0
      else
        1;

    # Define > with the same precedence as <.
    def binary> 10 (LHS RHS)
      RHS < LHS;

    # Binary "logical or", (note that it does not "short circuit")
    def binary| 5 (LHS RHS)
      if LHS then
        1
      else if RHS then
        1
      else
        0;

    # Define = with slightly lower precedence than relationals.
    def binary= 9 (LHS RHS)
      !(LHS < RHS | LHS > RHS);
```

许多语言都渴望能够用语言本身实现它们的标准运行时库。在Kaleidoscope中，我们可以在库中实现语言的重要部分！

我们将把这些功能的实现分为两部分：实现对用户定义的二元运算符的支持和添加一元运算符。

## 用户定义的二元运算符

在我们当前的框架中，添加对用户定义的二元运算符的支持非常简单。我们将首先添加对一元/二进制关键字的支持：

```c++
enum Token {
  ...
  // operators
  tok_binary = -11,
  tok_unary = -12
};
...
static int gettok() {
...
    if (IdentifierStr == "for")
      return tok_for;
    if (IdentifierStr == "in")
      return tok_in;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    return tok_identifier;
```

这只是添加了对一元和二进制关键字的词法分析器支持，就像我们在[以前的chapters](LangImpl05.html#lexer-extensions-for-if-then-else).]中所做的那样我们当前AST的一个优点是，我们通过使用二元运算符的ASCII代码作为操作码来表示完全泛化的二元运算符。对于我们的扩展操作符，我们将使用相同的表示，因此我们不需要任何新的AST或解析器支持。

另一方面，我们必须能够在函数定义的\“def Binary\\5\”部分中表示这些新运算符的定义。到目前为止，在我们的语法中，函数定义的\“name\”被解析为\“Prototype\”产品，并解析到`PrototypeAST‘AST节点。要将新的用户定义运算符表示为原型，我们必须扩展`PrototypeAST`AST节点，如下所示：

```c++
/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its argument names as well as if it is an operator.
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;
  bool IsOperator;
  unsigned Precedence;  // Precedence if a binary op.

public:
  PrototypeAST(const std::string &name, std::vector<std::string> Args,
               bool IsOperator = false, unsigned Prec = 0)
  : Name(name), Args(std::move(Args)), IsOperator(IsOperator),
    Precedence(Prec) {}

  Function *codegen();
  const std::string &getName() const { return Name; }

  bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
  bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name[Name.size() - 1];
  }

  unsigned getBinaryPrecedence() const { return Precedence; }
};
```

基本上，除了知道原型的名称之外，我们现在还跟踪它是否是运算符，如果是，则跟踪运算符的优先级别。优先级仅用于二元运算符(正如您将在下面看到的，它不适用于一元运算符)。现在我们有了表示用户定义运算符的原型的方法，我们需要对其进行解析：

```c++
/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  std::string FnName;

  unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Expected function name in prototype");
  case tok_identifier:
    FnName = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Expected binary operator");
    FnName = "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (NumVal < 1 || NumVal > 100)
        return LogErrorP("Invalid precedence: must be 1..100");
      BinaryPrecedence = (unsigned)NumVal;
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return LogErrorP("Expected '(' in prototype");

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken();  // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Invalid number of operands for operator");

  return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames), Kind != 0,
                                         BinaryPrecedence);
}
```

这些都是相当简单的解析代码，我们在过去已经看到了很多类似的代码。上述代码的一个有趣部分是为二元运算符设置`FnName`的几行代码。这将为新定义的\“@\”运算符构建\“BINARY@\”之类的名称。然后，它利用LLVM符号表中的符号名称被允许包含任何字符的事实，包括嵌入的NUL字符。

接下来要添加的有趣内容是对这些二元运算符的代码生成支持。根据我们当前的结构，这是为现有二元运算符节点添加一个默认情况的简单示例：

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
    break;
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "binary operator not found!");

  Value *Ops[2] = { L, R };
  return Builder.CreateCall(F, Ops, "binop");
}
```

正如您在上面看到的，新代码实际上非常简单。它只是在符号表中查找适当的运算符，并生成对它的函数调用。由于用户定义的运算符只是构建为普通函数(因为\“Prototype\”归根结底是一个具有正确名称的函数)，所以一切都井然有序。

我们遗漏的最后一段代码是一些顶级的魔术：

```c++
Function *FunctionAST::codegen() {
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function *TheFunction = getFunction(P.getName());
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext, "entry", TheFunction);
  ...
```

基本上，在对函数进行代码生成之前，如果它是用户定义的运算符，我们会将其注册到优先顺序表中。这允许我们已有的二元运算符解析逻辑来处理它。由于我们正在开发一个完全通用的运算符优先解析器，这就是我们需要做的全部工作，以\“扩展语法\”。

现在我们有了有用的用户定义的二元运算符。这在很大程度上建立在我们之前为其他运营商构建的框架之上。添加一元运算符更具挑战性，因为我们还没有任何框架-让我们看看需要什么。

## 用户定义的一元运算符

因为我们目前不支持Kaleidoscope语言中的一元运算符，所以我们需要添加所有内容来支持它们。上面，我们在词法分析器中添加了对\‘unary\’关键字的简单支持。除此之外，我们还需要一个AST节点：

```c++
/// UnaryExprAST - Expression class for a unary operator.
class UnaryExprAST : public ExprAST {
  char Opcode;
  std::unique_ptr<ExprAST> Operand;

public:
  UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
    : Opcode(Opcode), Operand(std::move(Operand)) {}

  Value *codegen() override;
};
```

到目前为止，这个AST节点非常简单和明显。它直接镜像二元运算符AST节点，只是它只有一个子节点。因此，我们需要添加解析逻辑。解析一元运算符非常简单：我们将添加一个新函数来执行此操作：

```c++
/// unary
///   ::= primary
///   ::= '!' unary
static std::unique_ptr<ExprAST> ParseUnary() {
  // If the current token is not an operator, it must be a primary expr.
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
    return ParsePrimary();

  // If this is a unary operator, read it.
  int Opc = CurTok;
  getNextToken();
  if (auto Operand = ParseUnary())
    return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
  return nullptr;
}
```

我们在这里添加的语法相当简单。如果在解析主运算符时看到一元运算符，我们会将该运算符作为前缀，并将其余部分作为另一个一元运算符进行解析。这允许我们处理多个一元运算符(例如，\“！！x\”)。请注意，一元操作符不能像二元操作符那样具有模棱两可的解析，因此不需要优先级信息。

这个函数的问题在于，我们需要从某个地方调用ParseUnary。为此，我们将之前的ParsePrimary调用方更改为调用ParseUnary：

```c++
/// binoprhs
///   ::= ('+' unary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS) {
  ...
    // Parse the unary expression after the binary operator.
    auto RHS = ParseUnary();
    if (!RHS)
      return nullptr;
  ...
}
/// expression
///   ::= unary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression() {
  auto LHS = ParseUnary();
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}
```

通过这两个简单的更改，我们现在可以解析一元运算符并为它们构建AST。接下来，我们需要添加对原型的解析器支持，以解析一元运算符原型。我们使用以下内容扩展上面的二元运算符代码：

```c++
/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  std::string FnName;

  unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Expected function name in prototype");
  case tok_identifier:
    FnName = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Expected unary operator");
    FnName = "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    ...
```

与二元运算符一样，我们使用包含运算符字符的名称命名一元运算符。这在代码生成时对我们有帮助。说到这里，我们需要添加的最后一点是对一元运算符的代码生成支持。它看起来是这样的：

```c++
Value *UnaryExprAST::codegen() {
  Value *OperandV = Operand->codegen();
  if (!OperandV)
    return nullptr;

  Function *F = getFunction(std::string("unary") + Opcode);
  if (!F)
    return LogErrorV("Unknown unary operator");

  return Builder.CreateCall(F, OperandV, "unop");
}
```

此代码类似于二元运算符的代码，但比二元运算符的代码更简单。它更简单，主要是因为它不需要处理任何预定义的运算符。

## 踢轮胎

这有点令人难以置信，但通过我们在上一章中介绍的几个简单扩展，我们已经发展出一种真正意义上的语言。有了这些，我们可以做很多有趣的事情，包括I/O、数学和许多其他事情。例如，我们现在可以添加一个很好的排序操作符(printd定义为打印指定的值和换行符)：
```
    ready> extern printd(x);
    Read extern:
    declare double @printd(double)

    ready> def binary : 1 (x y) 0;  # Low-precedence operator that ignores operands.
    ...
    ready> printd(123) : printd(456) : printd(789);
    123.000000
    456.000000
    789.000000
    Evaluated to 0.000000
```

我们还可以定义一组其他的\“原始\”操作，例如：
```
    # Logical unary not.
    def unary!(v)
      if v then
        0
      else
        1;

    # Unary negate.
    def unary-(v)
      0-v;

    # Define > with the same precedence as <.
    def binary> 10 (LHS RHS)
      RHS < LHS;

    # Binary logical or, which does not short circuit.
    def binary| 5 (LHS RHS)
      if LHS then
        1
      else if RHS then
        1
      else
        0;

    # Binary logical and, which does not short circuit.
    def binary& 6 (LHS RHS)
      if !LHS then
        0
      else
        !!RHS;

    # Define = with slightly lower precedence than relationals.
    def binary = 9 (LHS RHS)
      !(LHS < RHS | LHS > RHS);

    # Define ':' for sequencing: as a low-precedence operator that ignores operands
    # and just returns the RHS.
    def binary : 1 (x y) y;
```

给定前面的IF/THEN/ELSE支持，我们还可以为I/O定义有趣的函数。例如，下面打印出一个字符，其\“Density\”反映传入的值：该值越低，该字符就越密集：

    ready> extern putchard(char);
    ...
    ready> def printdensity(d)
      if d > 8 then
        putchard(32)  # ' '
      else if d > 4 then
        putchard(46)  # '.'
      else if d > 2 then
        putchard(43)  # '+'
      else
        putchard(42); # '*'
    ...
    ready> printdensity(1): printdensity(2): printdensity(3):
           printdensity(4): printdensity(5): printdensity(9):
           putchard(10);
    **++.
    Evaluated to 0.000000

基于这些简单的原语操作，我们可以开始定义更有趣的东西。例如，下面是一个小函数，它确定复杂平面中的某个函数发散所需的迭代次数：
```
    # Determine whether the specific location diverges.
    # Solve for z = z^2 + c in the complex plane.
    def mandelconverger(real imag iters creal cimag)
      if iters > 255 | (real*real + imag*imag > 4) then
        iters
      else
        mandelconverger(real*real - imag*imag + creal,
                        2*real*imag + cimag,
                        iters+1, creal, cimag);

    # Return the number of iterations required for the iteration to escape
    def mandelconverge(real imag)
      mandelconverger(real, imag, 0, real, imag);
```

这个\“`Z=z2+c`\”函数是一个美丽的小生物，它是计算[Mandelbrot Set](http://en.wikipedia.org/wiki/Mandelbrot_set).]的基础我们的‘mandelConverge`函数返回复杂轨道逃逸所需的迭代次数，饱和为255。这本身并不是一个非常有用的函数，但是如果您在二维平面上绘制它的值，您可以看到Mandelbrot集。鉴于我们在这里仅限于使用putchard，我们令人惊叹的图形输出也是有限的，但我们可以使用上面的密度绘图仪拼凑出一些东西：
```
    # Compute and plot the mandelbrot set with the specified 2 dimensional range
    # info.
    def mandelhelp(xmin xmax xstep   ymin ymax ystep)
      for y = ymin, y < ymax, ystep in (
        (for x = xmin, x < xmax, xstep in
           printdensity(mandelconverge(x,y)))
        : putchard(10)
      )

    # mandel - This is a convenient helper function for plotting the mandelbrot set
    # from the specified position with the specified Magnification.
    def mandel(realstart imagstart realmag imagmag)
      mandelhelp(realstart, realstart+realmag*78, realmag,
                 imagstart, imagstart+imagmag*40, imagmag);
```

考虑到这一点，我们可以试着画出曼德尔布洛特布景！让我们试试看：
```
    ready> mandel(-2.3, -1.3, 0.05, 0.07);
    *******************************+++++++++++*************************************
    *************************+++++++++++++++++++++++*******************************
    **********************+++++++++++++++++++++++++++++****************************
    *******************+++++++++++++++++++++.. ...++++++++*************************
    *****************++++++++++++++++++++++.... ...+++++++++***********************
    ***************+++++++++++++++++++++++.....   ...+++++++++*********************
    **************+++++++++++++++++++++++....     ....+++++++++********************
    *************++++++++++++++++++++++......      .....++++++++*******************
    ************+++++++++++++++++++++.......       .......+++++++******************
    ***********+++++++++++++++++++....                ... .+++++++*****************
    **********+++++++++++++++++.......                     .+++++++****************
    *********++++++++++++++...........                    ...+++++++***************
    ********++++++++++++............                      ...++++++++**************
    ********++++++++++... ..........                        .++++++++**************
    *******+++++++++.....                                   .+++++++++*************
    *******++++++++......                                  ..+++++++++*************
    *******++++++.......                                   ..+++++++++*************
    *******+++++......                                     ..+++++++++*************
    *******.... ....                                      ...+++++++++*************
    *******.... .                                         ...+++++++++*************
    *******+++++......                                    ...+++++++++*************
    *******++++++.......                                   ..+++++++++*************
    *******++++++++......                                   .+++++++++*************
    *******+++++++++.....                                  ..+++++++++*************
    ********++++++++++... ..........                        .++++++++**************
    ********++++++++++++............                      ...++++++++**************
    *********++++++++++++++..........                     ...+++++++***************
    **********++++++++++++++++........                     .+++++++****************
    **********++++++++++++++++++++....                ... ..+++++++****************
    ***********++++++++++++++++++++++.......       .......++++++++*****************
    ************+++++++++++++++++++++++......      ......++++++++******************
    **************+++++++++++++++++++++++....      ....++++++++********************
    ***************+++++++++++++++++++++++.....   ...+++++++++*********************
    *****************++++++++++++++++++++++....  ...++++++++***********************
    *******************+++++++++++++++++++++......++++++++*************************
    *********************++++++++++++++++++++++.++++++++***************************
    *************************+++++++++++++++++++++++*******************************
    ******************************+++++++++++++************************************
    *******************************************************************************
    *******************************************************************************
    *******************************************************************************
    Evaluated to 0.000000
    ready> mandel(-2, -1, 0.02, 0.04);
    **************************+++++++++++++++++++++++++++++++++++++++++++++++++++++
    ***********************++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    *********************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++.
    *******************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++...
    *****************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++.....
    ***************++++++++++++++++++++++++++++++++++++++++++++++++++++++++........
    **************++++++++++++++++++++++++++++++++++++++++++++++++++++++...........
    ************+++++++++++++++++++++++++++++++++++++++++++++++++++++..............
    ***********++++++++++++++++++++++++++++++++++++++++++++++++++........        .
    **********++++++++++++++++++++++++++++++++++++++++++++++.............
    ********+++++++++++++++++++++++++++++++++++++++++++..................
    *******+++++++++++++++++++++++++++++++++++++++.......................
    ******+++++++++++++++++++++++++++++++++++...........................
    *****++++++++++++++++++++++++++++++++............................
    *****++++++++++++++++++++++++++++...............................
    ****++++++++++++++++++++++++++......   .........................
    ***++++++++++++++++++++++++.........     ......    ...........
    ***++++++++++++++++++++++............
    **+++++++++++++++++++++..............
    **+++++++++++++++++++................
    *++++++++++++++++++.................
    *++++++++++++++++............ ...
    *++++++++++++++..............
    *+++....++++................
    *..........  ...........
    *
    *..........  ...........
    *+++....++++................
    *++++++++++++++..............
    *++++++++++++++++............ ...
    *++++++++++++++++++.................
    **+++++++++++++++++++................
    **+++++++++++++++++++++..............
    ***++++++++++++++++++++++............
    ***++++++++++++++++++++++++.........     ......    ...........
    ****++++++++++++++++++++++++++......   .........................
    *****++++++++++++++++++++++++++++...............................
    *****++++++++++++++++++++++++++++++++............................
    ******+++++++++++++++++++++++++++++++++++...........................
    *******+++++++++++++++++++++++++++++++++++++++.......................
    ********+++++++++++++++++++++++++++++++++++++++++++..................
    Evaluated to 0.000000
    ready> mandel(-0.9, -1.4, 0.02, 0.03);
    *******************************************************************************
    *******************************************************************************
    *******************************************************************************
    **********+++++++++++++++++++++************************************************
    *+++++++++++++++++++++++++++++++++++++++***************************************
    +++++++++++++++++++++++++++++++++++++++++++++**********************************
    ++++++++++++++++++++++++++++++++++++++++++++++++++*****************************
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++*************************
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++**********************
    +++++++++++++++++++++++++++++++++.........++++++++++++++++++*******************
    +++++++++++++++++++++++++++++++....   ......+++++++++++++++++++****************
    +++++++++++++++++++++++++++++.......  ........+++++++++++++++++++**************
    ++++++++++++++++++++++++++++........   ........++++++++++++++++++++************
    +++++++++++++++++++++++++++.........     ..  ...+++++++++++++++++++++**********
    ++++++++++++++++++++++++++...........        ....++++++++++++++++++++++********
    ++++++++++++++++++++++++.............       .......++++++++++++++++++++++******
    +++++++++++++++++++++++.............        ........+++++++++++++++++++++++****
    ++++++++++++++++++++++...........           ..........++++++++++++++++++++++***
    ++++++++++++++++++++...........                .........++++++++++++++++++++++*
    ++++++++++++++++++............                  ...........++++++++++++++++++++
    ++++++++++++++++...............                 .............++++++++++++++++++
    ++++++++++++++.................                 ...............++++++++++++++++
    ++++++++++++..................                  .................++++++++++++++
    +++++++++..................                      .................+++++++++++++
    ++++++........        .                               .........  ..++++++++++++
    ++............                                         ......    ....++++++++++
    ..............                                                    ...++++++++++
    ..............                                                    ....+++++++++
    ..............                                                    .....++++++++
    .............                                                    ......++++++++
    ...........                                                     .......++++++++
    .........                                                       ........+++++++
    .........                                                       ........+++++++
    .........                                                           ....+++++++
    ........                                                             ...+++++++
    .......                                                              ...+++++++
                                                                        ....+++++++
                                                                       .....+++++++
                                                                        ....+++++++
                                                                        ....+++++++
                                                                        ....+++++++
    Evaluated to 0.000000
    ready> ^D
```

在这一点上，您可能开始意识到Kaleidoscope是一种真实而强大的语言。它可能不是自相似的：)，但它可以用来绘制具有自相似的东西！

至此，我们结束了本教程的\“添加用户定义运算符\”一章。我们已经成功地扩展了我们的语言，添加了在库中扩展语言的能力，并且我们已经展示了如何使用这一功能在Kaleidoscope中构建简单但有趣的最终用户应用程序。在这一点上，Kaleidoscope可以构建各种功能齐全的应用程序，并且可以调用有副作用的函数，但是它不能实际定义和改变变量本身。

值得注意的是，变量突变是一些语言的一个重要特性，如何在不向前端添加\“SSA构造\”阶段的情况下[添加对可变变量的支持](LangImpl07.html)一点也不明显。在下一章中，我们将介绍如何在前端不构建SSA的情况下添加变量突变。

## 完整代码列表

下面是我们的运行示例的完整代码清单，增强了对用户定义运算符的支持。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

在某些平台上，链接时需要指定-rdynamic或-wl，\--export-dynamic。这确保将主可执行文件中定义的符号导出到动态链接器，以便在运行时可用于符号解析。如果将支持代码编译到共享库中，则不需要执行此操作，尽管这样做会在Windows上导致问题。

以下是代码：

[下一步：扩展语言：可变变量/SSA构造](LangImpl07.html)
