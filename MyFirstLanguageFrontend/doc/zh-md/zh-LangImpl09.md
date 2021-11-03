# 万花筒：添加调试信息

：{.content local=“”}
**

## 第九章引言

欢迎学习\“的第9章[使用实现语言
llvm](index.html)\“教程。在第1章到第8章中，我们构建了
带函数和变量的像样的小编程语言。什么
但是，如果出现问题，您如何调试您的程序呢？

源代码级别调试使用格式化数据来帮助调试器
将机器的二进制和状态转换回源代码
程序员写的。在LLVM中，我们通常使用一种称为
[矮人](http://dwarfstd.org).DWARF是一种紧凑编码，
表示类型、源位置和变量位置。

本章的简短总结是，我们将通过各种
您必须添加到编程语言中以支持调试信息的内容，
还有你怎么把它翻译成侏儒。

警告：目前我们不能通过JIT进行调试，所以我们需要编译
我们的节目精细化到一些小而独立的东西。作为这件事的一部分
我们将对语言的运行以及如何运行进行一些修改
编译程序。这意味着我们将有一个源文件，其中包含
用万花筒编写的简单程序，而不是交互式JIT。
它确实涉及一个限制，即我们只能有一个\“顶级\”
命令，以减少必要的更改次数。

下面是我们将要编译的示例程序：

```python
def fib(x)
  if x < 3 then
    1
  else
    fib(x-1)+fib(x-2);

fib(10)
```

## 为什么这是一个很难解决的问题？

调试信息是一个很难解决的问题，原因有几个-主要是
以优化代码为中心。首先，优化使守源。
地点更加困难。在LLVM IR中，我们保留原始源
指令上每个IR级别指令的位置。优化
通道应该保存新创建的指令的源位置，
但是合并后的指令只能保留一个位置-这可以
导致在单步执行优化程序时跳来跳去。第二，
优化可以以优化出的方式移动变量，
在内存中与其他变量共享，或难以跟踪。对于
本教程的目的是避免优化(正如您将
请参见使用下面的补丁程序集之一)。

## 提前编译模式

要仅突出显示将调试信息添加到源的各个方面，请执行以下操作
语言，无需担心JIT的复杂性
调试我们将对万花筒做一些更改以支持
将前端发出的IR编译成简单的单机版
可以执行、调试和查看结果的程序。

首先，我们创建包含顶层的匿名函数
语句为我们的\“Main\”：

```udiff
-    auto Proto = std::make_unique<PrototypeAST>("", std::vector<std::string>());
+    auto Proto = std::make_unique<PrototypeAST>("main", std::vector<std::string>());
```

只是简单地给它起了个名字。

然后，我们将删除任何存在的命令行代码：

```udiff
@@ -1129,7 +1129,6 @@ static void HandleTopLevelExpression() {
 /// top ::= definition | external | expression | ';'
 static void MainLoop() {
   while (1) {
-    fprintf(stderr, "ready> ");
     switch (CurTok) {
     case tok_eof:
       return;
@@ -1184,7 +1183,6 @@ int main() {
   BinopPrecedence['*'] = 40; // highest.

   // Prime the first token.
-  fprintf(stderr, "ready> ");
   getNextToken();
```

最后，我们将禁用所有优化过程，
JIT，所以在我们完成解析后唯一发生的事情就是
生成的代码是LLVM IR转到标准错误：

```udiff
@@ -1108,17 +1108,8 @@ static void HandleExtern() {
 static void HandleTopLevelExpression() {
   // Evaluate a top-level expression into an anonymous function.
   if (auto FnAST = ParseTopLevelExpr()) {
-    if (auto *FnIR = FnAST->codegen()) {
-      // We're just doing this to make sure it executes.
-      TheExecutionEngine->finalizeObject();
-      // JIT the function, returning a function pointer.
-      void *FPtr = TheExecutionEngine->getPointerToFunction(FnIR);
-
-      // Cast it to the right type (takes no arguments, returns a double) so we
-      // can call it as a native function.
-      double (*FP)() = (double (*)())(intptr_t)FPtr;
-      // Ignore the return value for this.
-      (void)FP;
+    if (!F->codegen()) {
+      fprintf(stderr, "Error generating code for top level expr");
     }
   } else {
     // Skip token for error recovery.
@@ -1439,11 +1459,11 @@ int main() {
   // target lays out data structures.
   TheModule->setDataLayout(TheExecutionEngine->getDataLayout());
   OurFPM.add(new DataLayoutPass());
+#if 0
   OurFPM.add(createBasicAliasAnalysisPass());
   // Promote allocas to registers.
   OurFPM.add(createPromoteMemoryToRegisterPass());
@@ -1218,7 +1210,7 @@ int main() {
   OurFPM.add(createGVNPass());
   // Simplify the control flow graph (deleting unreachable blocks, etc).
   OurFPM.add(createCFGSimplificationPass());
-
+  #endif
   OurFPM.doInitialization();

   // Set the global so the code gen can use this.
```

这组相对较小的更改使我们能够
将我们的万花筒语言编译成可执行程序
通过此命令行：

```bash
Kaleidoscope-Ch9 < fib.ks | & clang -x ir -
```

这将在当前工作目录中提供a.out/a.exe。

## 编译单位

DWARF中一段代码的顶级容器是编译器
单位。它包含个人的类型和功能数据
翻译单元(读取：源代码文件一份)。所以我们首先要做的就是
需要做的是为我们的fier.ks文件构建一个。

## 矮星发射设置

与`IRBuilder`类类似，我们有一个
[DIBuilder](https://llvm.org/doxygen/classllvm_1_1DIBuilder.html)类
这有助于构建LLVM IR文件的调试元数据。它
与`IRBuilder`和LLVM IR类似，但比`IRBuilder`和LLVM IR更好
名字。使用它确实需要您对Dwarf更加熟悉
术语比使用`IRBuilder`和`Instruction`所需的更多
名称，但如果您通读
[Metadata Format](https://llvm.org/docs/SourceLevelDebugging.html)It]
应该说得更清楚一点。我们将使用这个类来构造
我们所有的红外线级别描述。它的构造需要一个模块，所以
我们需要在构建模块后不久构建它。我们已经
将其保留为全局静电变量以使其更易于使用。

接下来，我们将创建一个小容器来缓存我们的一些
频繁的数据。第一个将是我们的编译单元，但我们还将编写
为我们的一种类型编写一些代码，因为我们不必担心
多类型表达式：

```c++
static DIBuilder *DBuilder;

struct DebugInfo {
  DICompileUnit *TheCU;
  DIType *DblTy;

  DIType *getDoubleTy();
} KSDbgInfo;

DIType *DebugInfo::getDoubleTy() {
  if (DblTy)
    return DblTy;

  DblTy = DBuilder->createBasicType("double", 64, dwarf::DW_ATE_float);
  return DblTy;
}
```

然后在稍后的“main`”中，当我们构建我们的模块时：

```c++
DBuilder = new DIBuilder(*TheModule);

KSDbgInfo.TheCU = DBuilder->createCompileUnit(
    dwarf::DW_LANG_C, DBuilder->createFile("fib.ks", "."),
    "Kaleidoscope Compiler", 0, "", 0);
```

这里有几件事需要注意。首先，在我们制作的同时
一种叫做万花筒的语言的编译单元我们使用的是这种语言
C常量。这是因为调试器不一定
了解一种语言的调用约定或默认ABI
无法识别，我们在LLVM代码生成中遵循C ABI
所以这是最接近准确的东西。这确保了我们可以
从调试器调用函数并执行它们。第二，
您将在对`createCompileUnit`的调用中看到\“fib.ks\”。这是一个
默认硬编码值，因为我们使用外壳重定向将我们的
源码到万花筒编译器中。在通常的前端，你会有
一个输入文件名，它就会出现在那里。

通过DIBuilder发出调试信息的最后一件事是
我们需要\“确定\”调试信息。部分原因是
DIBuilder的底层API，但请确保在接近
主干末端：

```c++
DBuilder->finalize();
```

在你倾倒模块之前。

## 功能

现在我们有了‘Compile Unit’和源位置，我们可以添加
调试信息的函数定义。所以在`PrototypeAST：：codegen()`中
我们添加几行代码来描述子程序的上下文，在
本例中的\“File\”和函数的实际定义
它本身。

所以上下文是这样的：

```c++
DIFile *Unit = DBuilder->createFile(KSDbgInfo.TheCU.getFilename(),
                                    KSDbgInfo.TheCU.getDirectory());
```

给我们一个DIFile，并向我们上面创建的‘Compile Unit’询问
我们当前所在的目录和文件名。那么，现在，我们使用
某些源位置为0(因为我们的AST当前没有源
位置信息)，并构造我们的函数定义：

```c++
DIScope *FContext = Unit;
unsigned LineNo = 0;
unsigned ScopeLine = 0;
DISubprogram *SP = DBuilder->createFunction(
    FContext, P.getName(), StringRef(), Unit, LineNo,
    CreateFunctionType(TheFunction->arg_size(), Unit),
    false /* internal linkage */, true /* definition */, ScopeLine,
    DINode::FlagPrototyped, false);
TheFunction->setSubprogram(SP);
```

现在我们有了一个DISubProgram，它包含对我们所有
函数的元数据。

## 源位置

调试信息最重要的是来源准确
位置--这使您可以将源代码映射回原来的位置。我们有一个
问题是，万花筒实际上没有任何震源位置
信息在词法分析器或解析器中，所以我们需要添加它。

```c++
struct SourceLocation {
  int Line;
  int Col;
};
static SourceLocation CurLoc;
static SourceLocation LexLoc = {1, 0};

static int advance() {
  int LastChar = getchar();

  if (LastChar == '\n' || LastChar == '\r') {
    LexLoc.Line++;
    LexLoc.Col = 0;
  } else
    LexLoc.Col++;
  return LastChar;
}
```

在这组代码中，我们添加了一些关于如何跟踪的功能
\“源文件\”的行和列的。当我们征用每一个令牌时，我们
将当前的\“lexical location\”设置为分类行，然后
用于标记开头的列。我们通过覆盖所有
前面使用我们的新`Advance()`调用`getchar()`，该新`Advance()`保持
跟踪信息，然后我们已将所有AST添加到
对源位置进行分类：

```c++
class ExprAST {
  SourceLocation Loc;

  public:
    ExprAST(SourceLocation Loc = CurLoc) : Loc(Loc) {}
    virtual ~ExprAST() {}
    virtual Value* codegen() = 0;
    int getLine() const { return Loc.Line; }
    int getCol() const { return Loc.Col; }
    virtual raw_ostream &dump(raw_ostream &out, int ind) {
      return out << ':' << getLine() << ':' << getCol() << '\n';
    }
```

我们在创建新表达式时会传递这些信息：

```c++
LHS = std::make_unique<BinaryExprAST>(BinLoc, BinOp, std::move(LHS),
                                       std::move(RHS));
```

为我们提供每个表达式和变量的位置。

确保每条指令都有正确的源位置
信息，每当我们在一个新的来源时，我们都必须告诉“建造者”
地点。为此，我们使用一个小帮助器函数：

```c++
void DebugInfo::emitLocation(ExprAST *AST) {
  DIScope *Scope;
  if (LexicalBlocks.empty())
    Scope = TheCU;
  else
    Scope = LexicalBlocks.back();
  Builder.SetCurrentDebugLocation(
      DILocation::get(Scope->getContext(), AST->getLine(), AST->getCol(), Scope));
}
```

这既告诉了主要的‘IRBuilder’我们所在的位置，也告诉了我们的作用域
我们进入了。作用域可以是编译单元级别，也可以是
最近的封闭词法挡路，喜欢当前函数。来代表
这将创建一个作用域堆栈：

```c++
std::vector<DIScope *> LexicalBlocks;
```

并在启动时将作用域(函数)推到堆栈的顶部
为每个函数生成代码：

```c++
KSDbgInfo.LexicalBlocks.push_back(SP);
```

此外，我们可能不会忘记将作用域从作用域堆栈中弹出
函数的代码生成结束：

```c++
// Pop off the lexical block for the function since we added it
// unconditionally.
KSDbgInfo.LexicalBlocks.pop_back();
```

然后，我们确保每次开始生成时都会发出该位置
新AST对象的代码：

```c++
KSDbgInfo.emitLocation(this);
```

## 变量

现在我们有了函数，我们需要能够打印出
我们有范围内的变量。让我们将我们的函数参数设置为
我们可以获得适当的回溯，并查看我们的函数是如何被调用的。
它的代码不是很多，我们通常在创建时处理它
`FunctionAST：：codegen`中的参数分配。

```c++
// Record the function arguments in the NamedValues map.
NamedValues.clear();
unsigned ArgIdx = 0;
for (auto &Arg : TheFunction->args()) {
  // Create an alloca for this variable.
  AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());

  // Create a debug descriptor for the variable.
  DILocalVariable *D = DBuilder->createParameterVariable(
      SP, Arg.getName(), ++ArgIdx, Unit, LineNo, KSDbgInfo.getDoubleTy(),
      true);

  DBuilder->insertDeclare(Alloca, D, DBuilder->createExpression(),
                          DILocation::get(SP->getContext(), LineNo, 0, SP),
                          Builder.GetInsertBlock());

  // Store the initial value into the alloca.
  Builder.CreateStore(&Arg, Alloca);

  // Add arguments to variable symbol table.
  NamedValues[Arg.getName()] = Alloca;
}
```

在这里，我们首先创建变量，为其提供作用域(`SP`)，
名称、源位置、类型，并且由于它是一个参数，因此该参数
索引。接下来，我们创建一个`lvm.dbg.ECLARRE`调用来指示IR
我们在alloca中有一个变量的级别(它给出了一个开始
变量的位置)，并为
声明上作用域的开始。

在这一点上需要注意的一件有趣的事情是各种调试器
根据代码和调试信息的生成方式进行假设
对他们来说都是过去的事。在这种情况下，我们需要做一点修改
要避免为函数序言生成行信息，以便
调试器知道在设置
断点。所以在`FunctionAST：：CodeGen`中，我们再增加几行：

```c++
// Unset the location for the prologue emission (leading instructions with no
// location in a function are considered part of the prologue and the debugger
// will run past them when breaking on a function)
KSDbgInfo.emitLocation(nullptr);
```

，然后在我们实际开始为其生成代码时发出一个新位置
函数体：

```c++
KSDbgInfo.emitLocation(Body.get());
```

有了这些，我们就有了足够的调试信息来设置断点
函数、打印参数变量和调用函数。还不算太差
只需几行简单的代码！

## 完整代码列表

以下是我们的运行示例的完整代码清单，增强了
调试信息。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

：{.writalinclude language=“c++”}
../examples/Kaleidoscope/Chapter9/toy.cpp
**：

[下一步：结论和其他有用的LLVM花絮](LangImpl10.html)
