# Kaleidoscope：Kaleidoscope介绍与词法分析

## Kaleidoscope语言

本教程使用一种名为“[Kaleidoscope](http://en.wikipedia.org/wiki/Kaleidoscope)”的玩具语言进行说明(含义是“漂亮、标准和直观”)。Kaleidoscope是一种过程性语言，允许您定义函数、使用条件、数学等。在本教程中，我们将扩展Kaleidoscope以支持IF/THEN/ELSE结构、for循环、用户定义的运算符、具有简单命令行界面的JIT编译、调试信息等。

我们希望保持简单，因此Kaleidoscope中唯一的数据类型是64位浮点类型(在C语言中也称为“双精度”)。因此，所有值都是隐式双精度的，并且该语言不需要类型声明。这为该语言提供了一种非常好并且简单的语法。例如，下面的简单示例计算[斐波那契数列：](http://en.wikipedia.org/wiki/Fibonacci_number)

```
    # Compute the x'th fibonacci number.
    def fib(x)
      if x < 3 then
        1
      else
        fib(x-1)+fib(x-2)

    # This expression will compute the 40th number.
    fib(40)
```

我们还允许Kaleidoscope调用标准库函数-LLVM JIT使这一点变得非常容易。这意味着您可以在使用函数之前使用‘extern’关键字来定义该函数(这对于相互递归的函数也很有用)。例如：

```
    extern sin(arg);
    extern cos(arg);
    extern atan2(arg1 arg2);

    atan2(sin(.4), cos(42))
```

第6章包括了一个更有趣的例子，在那里我们编写了一个小的Kaleidoscope应用程序，它以不同的放大倍数[显示一个Mandelbrot集](LangImpl06.md#Kick-the-the Tires)。

让我们深入研究一下这种语言的实现！

## 词法分析器

当谈到实现一种语言时，首先需要的是处理文本文件并识别其内容的能力。执行此操作的传统方法是使用词法分析器(也称为“[lexer](http://en.wikipedia.org/wiki/Lexical_analysis)”)将输入分解为“令牌（token）”。词法分析器返回的每个令牌都包括一个令牌码和一些可能的元数据(例如，数字的数字值)。首先，我们定义了可能性：

```c++
// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,
};

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;             // Filled in if tok_number
```

我们的词法分析器返回的每个令牌要么是一个令牌枚举值，要么是一个‘未知’字符，如‘+’，该字符将作为其ASCII值返回。如果当前令牌是标识符，则`IdentifierStr`全局变量保存标识符的名称。如果当前标记是数字文字(如1.0)，则`NumVal`保存其值。为简单起见，我们使用全局变量，但这不是真正语言实现的最佳选择：)。

词法分析器的实际实现是一个名为`gettok`的函数。调用`gettok`函数从标准输入返回下一个令牌。它的定义开始于：

```c++
/// gettok - Return the next token from standard input.
static int gettok() {
  static int LastChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = getchar();
```

`gettok`的工作原理是调用C语言中`getchar()`函数从标准输入中一次读取一个字符。它在识别它们时会将其吃掉，并将最后读取但未处理的字符存储在LastChar中。它必须做的第一件事是忽略标记之间的空格。这是通过上面的循环实现的。

`gettok`需要做的下一件事是识别标识符和特定的关键字，如“def”。Kaleidoscope用这个简单的循环来做这件事：

```c++
if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
  IdentifierStr = LastChar;
  while (isalnum((LastChar = getchar())))
    IdentifierStr += LastChar;

  if (IdentifierStr == "def")
    return tok_def;
  if (IdentifierStr == "extern")
    return tok_extern;
  return tok_identifier;
}
```

请注意，此代码在每次词法分析标识符时设置‘`IdentifierStr`’全局。此外，因为语言关键字由相同的循环匹配，所以我们在这里内联处理它们。数值相似：

```c++
if (isdigit(LastChar) || LastChar == '.') {   // Number: [0-9.]+
  std::string NumStr;
  do {
    NumStr += LastChar;
    LastChar = getchar();
  } while (isdigit(LastChar) || LastChar == '.');

  NumVal = strtod(NumStr.c_str(), 0);
  return tok_number;
}
```

这些都是用于处理输入的非常简单的代码。从输入读取数值时，我们使用C`strtod`函数将其转换为存储在`NumVal`中的数值。请注意，这没有进行充分的错误检查：它将错误地读取“1.23.45.67”并将其视为您键入的“1.23”。请随意扩展它！接下来，我们将处理注释：

```c++
if (LastChar == '#') {
  // Comment until end of line.
  do
    LastChar = getchar();
  while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

  if (LastChar != EOF)
    return gettok();
}
```

我们通过跳到行尾来处理注释，然后返回下一个标记。最后，如果输入与上述其中一种情况不匹配，则它要么是操作字符，如‘+’，要么是文件的末尾。这些使用以下代码进行处理：

```c++
// Check for end of file.  Don't eat the EOF.
if (LastChar == EOF)
  return tok_eof;

// Otherwise, just return the character as its ascii value.
int ThisChar = LastChar;
LastChar = getchar();
return ThisChar;
}
```

这样，我们就有了基本Kaleidoscope语言的完整词法分析器(本教程的[下一章](LangImpl02.md)中提供了词法分析器的[完整代码清单](LangImpl02.md#Full-code-Listing))。接下来，我们将[构建一个简单的解析器，使用它来构建抽象语法树](LangImpl02.md)。当我们完成后，我们将包含一个驱动程序，以便您可以同时使用词法分析器和解析器。

[下一步：实现解析器和AST](LangImpl02.md)

## 后记：心得体会
本教程仅用到了最基础的c++来完成词法分析部分的工作，相比于用flex做词法分析，是非常简洁的入门教程。进阶阅读可以了解下正则表达式以及flex。