# Kaleidoscope：编译成目标代码

## 第八章引言

欢迎阅读“[使用LLVM实现语言](index.html)”教程的第8章。本章介绍如何将我们的语言编译成目标文件。

## 选择目标

LLVM具有对交叉编译的原生支持。您可以编译到当前计算机的体系结构，也可以同样轻松地编译到其他体系结构。在本教程中，我们将以当前计算机为目标。

为了指定您想要面向的体系结构，我们使用一个名为“目标三元组”的字符串。它的形式为`<arch><sub>-<vendor>-<sys>-<abi>`(请参阅[交叉编译docs](https://clang.llvm.org/docs/CrossCompilation.html#target-triple)).

举个例子，我们可以看到Clang认为我们目前的目标三元组：
```
    $ clang --version | grep Target
    Target: x86_64-unknown-linux-gnu
```

运行此命令可能会在您的计算机上显示一些不同的内容，因为您可能正在使用与我不同的架构或操作系统。

幸运的是，我们不需要硬编码目标三元组来瞄准当前机器，LLVM提供了`sys::getDefaultTargetTriple`，它返回当前机器的目标三元组。

```c++
auto TargetTriple = sys::getDefaultTargetTriple();
```

LLVM不要求我们链接所有的目标功能。例如，如果我们只使用JIT，我们就不需要装配printers。同样，如果我们只针对某些架构，我们只能链接那些架构的功能。

在本例中，我们将初始化发出object code的所有targets。

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

我们还需要一台‘TargetMachine’。这个类提供了我们目标机器的完整机器描述。如果我们想要针对特定的功能(如SSE)或特定的CPU(如Intel的Sandylake)，我们现在就可以这么做。

要了解LLVM支持哪些功能和CPU，可以使用`llc`。例如，让我们看看x86：
```
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
```

在我们的示例中，我们将使用通用CPU，没有任何附加功能、选项或重新定位模型。

```c++
auto CPU = "generic";
auto Features = "";

TargetOptions opt;
auto RM = Optional<Reloc::Model>();
auto TargetMachine = Target->createTargetMachine(TargetTriple, CPU, Features, opt, RM);
```

## 配置模块

我们现在已经准备好配置我们的模块，以指定目标和数据布局。这并不是严格需要的，但[前端性能指南](../front end/PerformanceTips.html)建议您这样做。了解目标和数据布局对优化有好处。

```c++
TheModule->setDataLayout(TargetMachine->createDataLayout());
TheModule->setTargetTriple(TargetTriple);   
```

## 发送对象代码

我们已准备好发出目标代码！让我们定义我们要将文件写入的位置：

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

它能用吗？让我们试一试，我们需要编译代码，但是请注意，`llvm-config`的参数与前几章不同。
```
    $ clang++ -g -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs all` -o toy
```

让我们运行它，并定义一个简单的`verage`函数。完成后按Ctrl-D组合键。
```
    $ ./toy
    ready> def average(x y) (x + y) * 0.5;
    ^D
    Wrote output.o
```

我们有一个目标文件！为了测试它，让我们编写一个简单的程序，并将其与我们的输出相链接。源代码如下：

```c++
#include <iostream>

extern "C" {
    double average(double, double);
}

int main() {
    std::cout << "average of 3.0 and 4.0: " << average(3.0, 4.0) << std::endl;
}
```

我们将程序链接到output.o并检查结果是否符合我们的预期：
```
    $ clang++ main.cpp output.o -o main
    $ ./main
    average of 3.0 and 4.0: 3.5
```

## 完整代码列表

[下一步：添加调试信息](zh-LangImpl09.md)
