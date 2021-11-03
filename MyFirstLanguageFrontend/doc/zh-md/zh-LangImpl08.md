# 万花筒：编译成目标代码

：{.content local=“”}
**

## 第八章引言

欢迎学习\“的第8章[使用实现语言
LLVM](index.html)\“教程。本章介绍如何编译我们的
语言向下延伸到目标文件。

## 选择目标

LLVM具有对交叉编译的本机支持。您可以编译到
架构，或者同样容易地为
其他架构。在本教程中，我们将针对当前
机器。

要指定您想要面向的体系结构，我们使用一个字符串
叫做\“目标三元组\”。它的形式是
`-`(参见[交叉编译
docs](https://clang.llvm.org/docs/CrossCompilation.html#target-triple)).

举个例子，我们可以看到clang认为我们目前的目标是什么。
三重：

    $ clang --version | grep Target
    Target: x86_64-unknown-linux-gnu

运行此命令可能会在您的计算机上显示一些不同的内容
可能使用与我不同的架构或操作系统。

幸运的是，我们不需要硬编码目标三元组来瞄准
当前机器。LLVM提供`sys：：getDefaultTargetTriple`，该`sys：：getDefaultTargetTriple`
返回当前计算机的目标三元组。

```c++
auto TargetTriple = sys::getDefaultTargetTriple();
```

LLVM不需要我们链接所有目标功能。对于
例如，如果我们只使用JIT，则不需要程序集
打印机。同样，如果我们只针对某些架构，我们
只能链接那些架构的功能。

在本例中，我们将初始化用于发射对象的所有目标
密码。

```c++
InitializeAllTargetInfos();
InitializeAllTargets();
InitializeAllTargetMCs();
InitializeAllAsmParsers();
InitializeAllAsmPrinters();
```

我们现在可以使用我们的目标三元组来获得一个`Target`：

```c++
std::string Error;
auto Target = TargetRegistry::lookupTarget(TargetTriple, Error);

// Print an error and exit if we couldn't find the requested target.
// This generally occurs if we've forgotten to initialise the
// TargetRegistry or we have a bogus target triple.
if (!Target) {
  errs() << Error;
  return 1;
}
```

## 目标计算机

我们还需要一台‘TargetMachine’。这个类提供了完整的
我们的目标计算机的计算机描述。如果我们要
瞄准特定功能(如SSE)或特定CPU(如
英特尔的桑迪莱克)，我们现在就这么做。

要了解LLVM了解哪些功能和CPU，可以使用`llc`。
例如，让我们看看x86：

    $ llvm-as < /dev/null | llc -march=x86 -mattr=help
    Available CPUs for this target:
    
      amdfam10      - Select the amdfam10 processor.
      athlon        - Select the athlon processor.
      athlon-4      - Select the athlon-4 processor.
      ...
    
    Available features for this target:
    
      16bit-mode            - 16-bit mode (i8086).
      32bit-mode            - 32-bit mode (80386).
      3dnow                 - Enable 3DNow! instructions.
      3dnowa                - Enable 3DNow! Athlon instructions.
      ...

在我们的示例中，我们将使用通用CPU，而不使用任何额外的
功能、选项或重新定位模式。

```c++
auto CPU = "generic";
auto Features = "";

TargetOptions opt;
auto RM = Optional<Reloc::Model>();
auto TargetMachine = Target->createTargetMachine(TargetTriple, CPU, Features, opt, RM);
```

## 配置模块

我们现在已经准备好配置我们的模块，以指定目标和数据
布局。这不是严格要求，但[前端性能
Guide](../front end/PerformanceTips.html)建议您这样做。优化
从了解目标和数据布局中获益。

```c++
TheModule->setDataLayout(TargetMachine->createDataLayout());
TheModule->setTargetTriple(TargetTriple);   
```

## 发送对象代码

我们已经准备好发出目标代码！让我们定义我们想要写入的位置
我们的文件发送到：

```c++
auto Filename = "output.o";
std::error_code EC;
raw_fd_ostream dest(Filename, EC, sys::fs::OF_None);

if (EC) {
  errs() << "Could not open file: " << EC.message();
  return 1;
}
```

最后，我们定义一个发出对象代码的过程，然后运行该过程：

```c++
legacy::PassManager pass;
auto FileType = CGFT_ObjectFile;

if (TargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
  errs() << "TargetMachine can't emit a file of this type";
  return 1;
}

pass.run(*TheModule);
dest.flush();
```

## 把这一切放在一起

它能用吗？让我们试一试。我们需要编译我们的代码，但是
请注意，`llvm-config`的参数与前面的
章节。

    $ clang++ -g -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs all` -o toy

让我们运行它，并定义一个简单的`verage`函数。
你的任务完成了。

    $ ./toy
    ready> def average(x y) (x + y) * 0.5;
    ^D
    Wrote output.o

我们有一个目标文件！为了测试它，让我们编写一个简单的程序并
把它和我们的输出联系起来。以下是源代码：

```c++
#include <iostream>

extern "C" {
    double average(double, double);
}

int main() {
    std::cout << "average of 3.0 and 4.0: " << average(3.0, 4.0) << std::endl;
}
```

我们将程序链接到output.o并检查结果
预期：

    $ clang++ main.cpp output.o -o main
    $ ./main
    average of 3.0 and 4.0: 3.5

## 完整代码列表

：{.writalinclude language=“c++”}
../examples/Kaleidoscope/Chapter8/toy.cpp
**：

[下一步：添加调试信息](LangImpl09.html)
