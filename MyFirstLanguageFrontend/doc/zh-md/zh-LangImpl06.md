# 万花筒：扩展语言：用户定义运算符

：{.content local=“”}
**

## 第六章绪论

欢迎学习\“的第6章[使用实现语言
LLVM](index.html)\“教程。
一种功能齐全的语言，它相当简单，但也很有用。
然而，它仍然有一个很大的问题。我们的语言没有
有许多有用的运算符(如除法、逻辑否定，甚至任何
除了小于)之外，还进行了比较。

本教程的本章对添加
简单美观的万花筒用户自定义运算符
语言。这段题外话现在给了我们一种简单而丑陋的语言
在某些方面，但同时也是一种强有力的方式。世界上最伟大的
关于创造你自己的语言的事情是你可以决定什么
是好是坏。在本教程中，我们将假设可以使用
这是展示一些有趣的解析技术的一种方式。

在本教程的最后，我们将介绍一个示例万花筒
[渲染Mandelbrot集](#踢轮胎)的应用程序。这
给出了一个使用万花筒可以构建的示例及其功能
准备好了。

## 用户定义运算符：理念

我们将向万花筒添加的\“运算符重载\”比
它比C++之类的语言更通用。在C++中，您只允许
重新定义现有运算符：不能以编程方式更改
语法、引入新运算符、更改优先级别等。
一章中，我们将在万花筒中添加此功能，它将使
用户对支持的运算符集合进行四舍五入。

在这样的教程中介绍用户定义的运算符的意义在于
是为了展示使用手写解析器的功能和灵活性。
到目前为止，我们一直在实现的解析器使用递归下降
对于语法的大多数部分和运算符优先级分析
表情。详见[第2章](LangImpl02.html)。通过使用
运算符优先解析，很容易让程序员
在语法中引入新的运算符：语法是动态的
JIT运行时可扩展。

我们要添加的两个特定功能是可编程的一元运算符
(目前，万花筒根本没有一元运算符)以及
二元运算符。这方面的一个示例是：

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

许多语言都渴望能够实现它们的标准运行时
库中的语言本身。在万花筒里，我们可以实现
库中语言的重要部分！

我们将这些功能的实现分为两个部分：
实现对用户定义的二元运算符的支持并添加一元
操作员。

## 用户定义的二元运算符

添加对用户定义的二元运算符的支持非常简单
我们目前的框架。我们将首先添加对一元/二进制的支持
关键词：

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

这只是添加了对一元和二进制关键字的词法分析器支持，就像我们
在[以前]中完成
chapters](LangImpl05.html#lexer-extensions-for-if-then-else).一个不错的
关于我们当前的AST，是我们用来表示二元运算符的
通过使用他们的ASCII码作为操作码来实现完全泛化。为了我们的
扩展运算符，我们将使用相同的表示，所以我们不
需要任何新的AST或解析器支持。

另一方面，我们必须能够表示
这些新运算符位于函数的\“def Binary\\5\”部分
定义。到目前为止，在我们的语法中，函数的\“名称\”
将定义解析为\“Prototype\”产品，并将其解析为
`PrototypeAST`AST节点。将新的用户定义运算符表示为
Prototype，我们必须像这样扩展`PrototypeAST`AST节点：

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

基本上，除了知道原型的名称外，我们现在还
跟踪它是否是运算符，如果是，优先级是什么
操作员所处的级别。优先级仅用于二进制
运算符(正如您将在下面看到的，它不适用于一元
运算符)。现在我们有了一种方法来表示
自定义运算符，需要解析：

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

这些都是相当简单的解析代码，我们已经
在过去看到了很多类似的代码。有一点很有趣，那就是
上面的代码是为二进制文件设置`FnName`的几行代码
操作员。这将为新定义的\“@\”生成类似\“Binary@\”的名称
接线员。然后，它利用这样一个事实，即符号在
LLVM符号表允许包含任何字符，包括
嵌入NUL字符。

下一个要添加有趣事情是对这些二进制文件的代码生成支持
操作员。根据我们当前的结构，这只是简单地添加了一个
现有二元运算符节点的默认情况：

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

正如您在上面看到的，新代码实际上非常简单。它只是
在符号表中查找适当的运算符，并
生成对它的函数调用。由于用户定义运算符仅
作为普通函数构建(因为\“原型\”归结为
函数使用正确的名称)，一切都井然有序。

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

基本上，在对函数进行代码生成之前，如果该函数是用户定义的
运算符，我们将其注册到优先级表中。这允许二进制文件
我们已经有了运算符解析逻辑来处理它。因为我们
正在开发一个完全通用的运算符优先解析器，仅此而已
我们需要做的是\“扩展语法\”。

现在我们有了有用的用户定义的二元运算符。这在很大程度上建立在
我们之前为其他运营商构建的框架。正在添加一元
操作员更具挑战性，因为我们没有
它的框架还没有-让我们看看它需要什么。

## 用户定义的一元运算符

因为我们目前不支持万花筒中的一元运算符
语言，我们需要添加所有内容来支持它们。上面，我们添加了
对词法分析器的\‘unary\’关键字的简单支持。除了……之外
因此，我们需要一个AST节点：

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

到目前为止，这个AST节点非常简单和明显。它直接反映了
二元运算符AST节点，但它只有一个子级。有了这个，
我们需要添加解析逻辑。解析一元运算符很不错
简单：我们将添加一个新函数来完成此操作：

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

我们在这里添加的语法相当简单。如果我们看到一元
运算符分析主运算符时，我们将运算符作为
将剩余部分作为另一个一元运算符添加前缀并进行解析。这
允许我们处理多个一元运算符(例如\“！！x\”)。请注意，
一元运算符不能像二元运算符那样具有模棱两可的解析，
因此不需要优先级信息。

此函数的问题在于，我们需要从
在某个地方。为此，我们将以前的ParsePrimary调用方更改为
改为调用ParseUnary：

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

通过这两个简单的更改，我们现在可以解析一元运算符
为他们建造天桥。接下来，我们需要添加对以下内容的解析器支持
原型，用于解析一元运算符原型。我们扩展了二进制
上面的运算符代码为：

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

与二元运算符一样，我们使用以下名称命名一元运算符
包括操作员字符。这有助于我们生成代码
时间到了。说到这里，我们需要添加的最后一点是对代码生成的支持
一元运算符。它看起来是这样的：

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

此代码类似于二进制代码，但比二进制代码更简单
操作员。它更简单，主要是因为它不需要处理
任何预定义的运算符。

## 踢轮胎

这有点令人难以置信，但经过几个简单的扩展，我们已经
在最后几章中，我们已经发展出一种真正的语言。使用
这样，我们可以做很多有趣的事情，包括I/O、数学和
一堆其他的东西。例如，我们现在可以添加一个很好的排序
运算符(printd定义为打印出指定值和
Newline)：

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

我们还可以定义一组其他的\“原始\”操作，例如：

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

考虑到前面的IF/THEN/ELSE支持，我们还可以定义感兴趣的
用于I/O的函数。例如，下面的代码打印出一个字符
其\“Density\”反映传入的值：值越低，
角色更加密集：

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

基于这些简单的原语操作，我们可以开始定义更多
有趣的事情。例如，这里有一个小函数，它
中某个函数所需的迭代次数。
要发散的复杂平面：

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

这个\“`Z=z2+c`\”函数是一个美丽的小生物，
[Mandelbrot]的计算基础
Set](http://en.wikipedia.org/wiki/Mandelbrot_set).我们的“MandelConverge”
函数返回复数所需的迭代次数
绕圈逃逸，饱和到255度。这不是一个非常有用的函数
本身，但如果将其值绘制在二维平面上，则
可以看到曼德尔布洛特的套装。假设我们只能使用Putchard
在这里，我们令人惊叹的图形输出是有限的，但我们可以在一起
使用上面的密度绘图仪的内容：

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

考虑到这一点，我们可以试着画出曼德尔布洛特布景！让我们试试看：

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

在这一点上，您可能开始意识到万花筒是一个
真实而有力的语言。它可能不是自相似的：)，但它可以是
用来策划事情的真相！

至此，我们结束了的\“添加用户定义运算符\”一章
教程。我们已经成功地扩展了我们的语言，添加了
在库中扩展语言的能力，我们已经展示了如何
这可用于构建简单但有趣的最终用户应用程序
在万花筒里。在这一点上，万花筒可以构建各种
功能正常的应用程序，并且可以使用
副作用，但它实际上不能定义和变异变量
它本身。

值得注意的是，变量突变是某些语言的重要特征，
而且完全不清楚如何[添加对可变的支持
变量](LangImpl07.html)，而无需添加\“SSA
构建\“阶段到您的前端。在下一章中，我们将
描述如何在不构建SSA的情况下添加变量突变
前端。

## 完整代码列表

以下是我们的运行示例的完整代码清单，增强了
对用户定义运算符的支持。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

在某些平台上，您需要指定-rdynamic或
-WL，\-导出-链接时动态。这样可以确保在中定义的符号
主可执行文件被导出到动态链接器，因此也是如此
可用于运行时的符号解析。如果您要执行此操作，则不需要执行此操作
将您的支持代码编译到共享库中，尽管这样做
将在Windows上导致问题。

以下是代码：

：{.writalinclude language=“c++”}
../examples/Kaleidoscope/Chapter6/toy.cpp
**：

[下一步：扩展语言：可变变量/SSA
建设](LangImpl07.html)
