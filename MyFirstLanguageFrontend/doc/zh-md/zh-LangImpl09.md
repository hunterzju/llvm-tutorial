# Kaleidoscope：添加调试信息

## 第九章引言

欢迎阅读“[使用LLVM实现语言](zh-index.md)”教程的第9章。在第1章到第8章中，我们已经用函数和变量构建了一种不错的小型编程语言。但是，如果出现问题怎么办，您如何调试您的程序呢？

源代码级别调试使用格式化数据来帮助调试器将二进制代码和计算机状态转换回程序员编写的源代码。在LLVM中，我们通常使用称为[DWARF](http://dwarfstd.org)格式。DWARF是一种表示类型、源代码位置和变量位置的紧凑编码。

本章的简短总结是，我们将介绍为支持调试信息而必须添加到编程语言中的各种内容，以及如何将其转换为DWARF。

> 警告：目前我们不能通过JIT进行调试，因此我们需要将我们的程序编译成一些小而独立的东西。作为这项工作的一部分，我们将对语言的运行和程序的编译方式进行一些修改。这意味着我们将有一个源文件，其中包含一个用Kaleidoscope而不是交互式JIT编写的简单程序。它确实涉及一个限制，即我们一次只能有一个“顶层”命令，以减少必要的更改次数。

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

由于几个不同的原因，调试信息是一个棘手的问题-主要集中在优化的代码上。首先，优化使得保持源代码位置更加困难。在LLVM IR中，我们在指令上保留每个IR级别指令的原始源位置。优化passes应该保留新创建的指令的源位置，但合并的指令只保留一个位置-这可能会导致在单步执行优化程序时原地跳转。其次，优化可以通过优化、与其他变量共享内存或难以跟踪的方式移动变量。出于本教程的目的，我们将避免优化(正如您将在接下来的补丁程序中看到的那样)。

## 提前编译模式

为了只强调将调试信息添加到源语言的各个方面，而不需要担心JIT调试的复杂性，我们将对Kaleidoscope进行一些更改，以支持将前端发出的IR编译成可以执行、调试和查看结果的简单独立程序。

首先，我们将包含顶层语句的匿名函数设置为“main”：

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

最后，我们将禁用所有优化过程和JIT，以便在我们完成解析和生成代码后唯一发生的事情是LLVM IR转到标准错误流输出：

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

这组相对较小的更改使我们可以通过以下命令行将我们的一段Kaleidoscope语言编译成可执行程序：

```bash
Kaleidoscope-Ch9 < fib.ks | & clang -x ir -
```

这将在当前工作目录中提供a.out/a.exe。

## 编译单元

DWARF中代码段的顶层容器是编译单元。它包含单个翻译单元的类型和功能数据(读取：一个源代码文件)。因此，我们需要做的第一件事是为fier.ks文件构建一个编译单元。

## DWARF发射设置

与`IRBuilder`类类似，我们有一个[DIBuilder](https://llvm.org/doxygen/classllvm_1_1DIBuilder.html)类，它帮助构建LLVMIR文件的调试元数据。与`IRBuilder`和LLVM IR 1：1对应，但名称更好听。使用它确实需要您比熟悉`IRBuilder`和`Instruction`名称时更熟悉Dwarf术语，但是如果您通读[Metadata Format](https://llvm.org/docs/SourceLevelDebugging.html)]上的通用文档，应该会更清楚一些。我们将使用这个类来构造我们所有的IR级别描述。它的构造需要一个模块，所以我们需要在构造模块后不久构造它。为了使它更易于使用，我们将其保留为全局静态变量。

接下来，我们将创建一个小容器来缓存一些频繁使用的数据。第一个容器将是我们的编译单元，但是我们也将为我们的每种类型编写一些代码，因为我们不必担心多个类型的表达式：

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

这里有几件事需要注意。首先，当我们为名为Kaleidoscope的语言生成编译单元时，我们使用了C语言中的常量，这是因为调试器不一定理解它无法识别的语言的调用约定或缺省ABI，并且我们在LLVM代码生成中遵循C ABI，所以它是最接近准确的。这确保了我们可以实际从调试器调用函数并执行它们。其次，您将在对`createCompileUnit`的调用中看到“fib.ks”。这是默认的硬编码值，因为我们使用shell重定向将源代码放入Kaleidoscope编译器。在通常的前端，您会有一个输入文件名，它会放在那里。

通过DIBuilder发出调试信息的最后一件事是，我们需要“确定”调试信息。原因是DIBuilder的底层API的一部分，但请确保在Main的末尾,导出模块之前执行此操作：

```c++
DBuilder->finalize();
```

## 函数

现在我们有了`Compile Unit`和源位置，我们可以将函数定义添加到调试信息中。因此，在`PrototypeAST::codegen()`中，我们添加了几行代码来描述子程序的上下文，在本例中为“File”，以及函数本身的实际定义。

所以上下文是这样的：

```c++
DIFile *Unit = DBuilder->createFile(KSDbgInfo.TheCU.getFilename(),
                                    KSDbgInfo.TheCU.getDirectory());
```

给我们一个DIFile，并向我们上面创建的`Compile Unit`询问我们当前所在的目录和文件名。现在，我们使用一些值为0的源位置(因为我们的AST当前没有源位置信息)，并构造我们的函数定义：

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

现在我们有了一个DISubProgram，它包含对函数的所有元数据的引用。

## 源位置

调试信息最重要的是准确的源代码位置-这使得您可以将源代码映射回原来的位置。但是我们有一个问题，Kaleidoscope在词法分析器或解析器中确实没有任何源位置信息，所以我们需要添加它。

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

在这组代码中，我们添加了一些关于如何跟踪“源文件”的行和列的功能。当我们对每个令牌进行lex时，我们将当前的“lexical location”设置为令牌开头的分类行和列。为此，我们使用跟踪信息的新的`Advance()`覆盖了之前对`getchar()`的所有调用，然后我们向所有AST类添加了一个源位置：

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

为了确保每条指令都能获得正确的源位置信息，每当我们在一个新的源位置时，我们都必须告诉`Builder`。为此，我们使用了一个小的辅助函数：

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

这既告诉主`IRBuilder‘我们所在的位置，也告诉我们所在的作用域。作用域可以是编译单元级别的，也可以是最接近的封闭词法block，比如当前函数。为了表示这一点，我们创建了一个作用域堆栈：

```c++
std::vector<DIScope *> LexicalBlocks;
```

并在开始为每个函数生成代码时将作用域(函数)推到堆栈的顶部：

```c++
KSDbgInfo.LexicalBlocks.push_back(SP);
```

此外，我们不能忘记在函数的代码生成结束时将作用域从作用域堆栈中弹出：

```c++
// Pop off the lexical block for the function since we added it
// unconditionally.
KSDbgInfo.LexicalBlocks.pop_back();
```

然后，我们确保在每次开始为新AST对象生成代码时发出位置：

```c++
KSDbgInfo.emitLocation(this);
```

## 变量

现在我们有了函数，我们需要能够打印出范围内的变量。让我们设置我们的函数参数，这样我们就可以进行适当的回溯，看看我们的函数是如何被调用的。这不是很多代码，我们通常在`FunctionAST::codegen`中创建参数allocas时处理它。

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

在这里，我们首先创建变量，为其提供作用域(`SP`)、名称、源位置、类型，并且由于它是参数，因此还提供参数索引。接下来，我们创建一个`lvm.dbg.declare`调用，以在IR级别指示我们在alloca中有一个变量(并且它给出变量的起始位置)，并在声明上设置作用域开始的源位置。

在这一点上需要注意的一件有趣的事情是，各种调试器都有基于过去如何为它们生成代码和调试信息的假设。在这种情况下，我们需要做一些修改，以避免为函数序言生成行信息，以便调试器知道在设置断点时跳过这些指令。所以在`FunctionAST::CodeGen`中，我们再增加几行：

```c++
// Unset the location for the prologue emission (leading instructions with no
// location in a function are considered part of the prologue and the debugger
// will run past them when breaking on a function)
KSDbgInfo.emitLocation(nullptr);
```

然后在我们实际开始为函数体生成代码时发出一个新位置：

```c++
KSDbgInfo.emitLocation(Body.get());
```

这样，我们就有了足够的调试信息，可以在函数中设置断点、打印参数变量和调用函数。对于仅仅几行简单的代码来说还不错！

## 完整代码列表

下面是我们运行示例的完整代码清单，并使用调试信息进行了增强。要构建此示例，请使用：

```bash
# Compile
clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
# Run
./toy
```

以下是代码：

[下一步：结论和其他有用的LLVM花絮](zh-LangImpl10.html)
