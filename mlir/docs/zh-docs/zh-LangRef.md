# MLIR语言参考

MLIR(多级IR)是一种编译器中间表示，与传统的三地址SSA表示(如[LLVMIR](http://llvm.org/docs/LangRef.html)或[SIL](https://github.com/apple/swift/blob/master/docs/SIL.rst))相似，但它引入了多面体循环优化的概念作为一级概念。这种混合设计经过优化，可以表示、分析和转换高级数据流图以及为高性能数据并行系统生成的特定于目标的代码。除了其表示能力之外，它的单一连续设计还提供了一个框架，可以将数据流图降低到高性能的特定于目标的代码。

本文档定义并描述了MLIR中的关键概念，旨在作为一份干性参考文档-[Rationale Documentation](Rationale/Rationale.md)、[Glossary](../Get_Started/Glossary.md)和其他内容位于其他地方。

MLIR被设计成以三种不同的形式使用：适于调试的人类可读的文本形式、适于编程转换和分析的内存中形式以及适于存储和传输的紧凑的序列化形式。不同的形式都描述了相同的语义内容。本文档描述了人类可读的文本形式。

## 高层结构

MLIR基本上基于称为*Operation*的节点和称为*value*的边的图数据结构。每个值都恰好是一个Operation或block参数的结果，并且有一个由[type system](#type system)定义的*Value Type*。[Operations](#Operations)包含在[Blocks](#Blocks)中，Blocks包含在[Regions](#Regions)中。Operations也在其包含的block中排序，块在其包含的Regions中排序，尽管此顺序在给定的[Regions种类](Interfaces.md#regionkindinterface)中可能在语义上有意义，也可能没有意义。Operations还可以包含Regions，从而能够表示分层结构。

Operation可以表示许多不同的概念，从较高级别的概念(如函数定义、函数调用、缓冲区分配、缓冲区视图或切片以及进程创建)到较低级别的概念(如独立于目标的算术、特定于目标的指令、配置寄存器和逻辑门)。这些不同的概念由MLIR中的不同Operation表示，并且MLIR中可用的Operation集可以任意扩展。

MLIR还使用熟悉的编译器[通道](Passes.md)概念，为Operation上的转换提供了一个可扩展的框架。在任意一组Operation上启用任意一组PASS会导致重大的伸缩挑战，因为每个转换都必须潜在地考虑任何Operation的语义。MLIR允许使用[特征](Traits.md)和[接口](Interfaces.md)抽象地描述Operation语义，从而使转换能够更通用地OperationOperation，从而解决了这一复杂性。特征通常描述对有效IR的验证约束，使得复杂的不变量能够被捕获和检查。(请参阅[Op vs Operation](docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations))

MLIR的一个明显应用是表示SSAIR，如LLVMcore IR，使用适当的Operation类型选择来定义[模块](#[SSA-based](https://en.wikipedia.org/wiki/Static_single_assignment_form))、[函数](#函数)、分支、分配和验证约束，以确保SSA支配属性。MLIR包括一种“标准”方言，它正好定义了这样的结构。然而，MLIR的目的是足够通用，以表示其他类似编译器的数据结构，如语言前端中的抽象语法树、特定于目标的后端中生成的指令或高级合成工具中的电路。

以下是MLIR模块的示例：

```mlir
// Compute A*B using an implementation of multiply kernel and print the
// result using a TensorFlow op. The dimensions of A and B are partially
// known. The shapes are assumed to match.
func @mul(%A: tensor<100x?xf32>, %B: tensor<?x50xf32>) -> (tensor<100x50xf32>) {
  // Compute the inner dimension of %A using the dim operation.
  %n = dim %A, 1 : tensor<100x?xf32>

  // Allocate addressable "buffers" and copy tensors %A and %B into them.
  %A_m = alloc(%n) : memref<100x?xf32>
  tensor_store %A to %A_m : memref<100x?xf32>

  %B_m = alloc(%n) : memref<?x50xf32>
  tensor_store %B to %B_m : memref<?x50xf32>

  // Call function @multiply passing memrefs as arguments,
  // and getting returned the result of the multiplication.
  %C_m = call @multiply(%A_m, %B_m)
          : (memref<100x?xf32>, memref<?x50xf32>) -> (memref<100x50xf32>)

  dealloc %A_m : memref<100x?xf32>
  dealloc %B_m : memref<?x50xf32>

  // Load the buffer data into a higher level "tensor" value.
  %C = tensor_load %C_m : memref<100x50xf32>
  dealloc %C_m : memref<100x50xf32>

  // Call TensorFlow built-in function to print the result tensor.
  "tf.Print"(%C){message: "mul result"}
                  : (tensor<100x50xf32) -> (tensor<100x50xf32>)

  return %C : tensor<100x50xf32>
}

// A function that multiplies two memrefs and returns the result.
func @multiply(%A: memref<100x?xf32>, %B: memref<?x50xf32>)
          -> (memref<100x50xf32>)  {
  // Compute the inner dimension of %A.
  %n = dim %A, 1 : memref<100x?xf32>

  // Allocate memory for the multiplication result.
  %C = alloc() : memref<100x50xf32>

  // Multiplication loop nest.
  affine.for %i = 0 to 100 {
     affine.for %j = 0 to 50 {
        store 0 to %C[%i, %j] : memref<100x50xf32>
        affine.for %k = 0 to %n {
           %a_v  = load %A[%i, %k] : memref<100x?xf32>
           %b_v  = load %B[%k, %j] : memref<?x50xf32>
           %prod = mulf %a_v, %b_v : f32
           %c_v  = load %C[%i, %j] : memref<100x50xf32>
           %sum  = addf %c_v, %prod : f32
           store %sum, %C[%i, %j] : memref<100x50xf32>
        }
     }
  }
  return %C : memref<100x50xf32>
}
```

## 表示方法

MLIR具有简单而明确的语法，允许它可靠地往返于文本形式。这对于编译器的开发非常重要-例如，对于理解代码被转换时的状态和编写测试用例。

本文档描述了使用[扩展巴克斯-诺尔形式(EBNF)](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form)的语法

这是本文档中使用的EBNF语法。

```
alternation ::= expr0 | expr1 | expr2  // Either expr0 or expr1 or expr2.
sequence    ::= expr0 expr1 expr2      // Sequence of expr0 expr1 expr2.
repetition0 ::= expr*  // 0 or more occurrences.
repetition1 ::= expr+  // 1 or more occurrences.
optionality ::= expr?  // 0 or 1 occurrence.
grouping    ::= (expr) // Everything inside parens is grouped together.
literal     ::= `abcd` // Matches the literal `abcd`.
```

代码示例。

```mlir
// This is an example use of the grammar above:
// This matches things like: ba, bana, boma, banana, banoma, bomana...
example ::= `b` (`an` | `om`)* `a`
```

### 常见语法

本文档中使用了以下核心语法：

```
// TODO: Clarify the split between lexing (tokens) and parsing (grammar).
digit     ::= [0-9]
hex_digit ::= [0-9a-fA-F]
letter    ::= [a-zA-Z]
id-punct  ::= [$._-]

integer-literal ::= decimal-literal | hexadecimal-literal
decimal-literal ::= digit+
hexadecimal-literal ::= `0x` hex_digit+
float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
string-literal  ::= `"` [^"\n\f\v\r]* `"`   TODO: define escaping rules
```

此处未列出，但MLIR确实支持注释。它们使用标准的BCPL语法，以`//`开始，一直到行尾。

### 标识符和关键字

语法：

```
// Identifiers
bare-id ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list ::= bare-id (`,` bare-id)*
value-id ::= `%` suffix-id
suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

symbol-ref-id ::= `@` (suffix-id | string-literal)
value-id-list ::= value-id (`,` value-id)*

// Uses of value, e.g. in an operand list to an operation.
value-use ::= value-id
value-use-list ::= value-use (`,` value-use)*
```

标识符命名实体，如值、类型和函数，并由MLIR代码的编写者选择。标识符可以是描述性的(例如`%Batch_size`、`@matmul`)，也可以是自动生成时的非描述性的(例如`%23`、`@func42`)。值的标识符名称可以在MLIR文本文件中使用，但不会作为IR的一部分保存-打印机将为它们提供匿名名称，如`%42`。

MLIR通过在标识符前面加上符号(如`%`、`#`、`@`、`^`、`！`)来保证标识符不会与关键字冲突。在某些明确的上下文中(例如仿射表达式)，为简洁起见，标识符没有前缀。可以将新关键字添加到MLIR的未来版本中，而不会有与现有标识符冲突的危险。

对于定义值标识符的(嵌套)Regions，值标识符仅[在作用域中](#值作用域)，不能在该Regions之外进行访问或引用。映射函数中的参数标识符在映射体的作用域内。特定Operation可以进一步限制哪些标识符在其Regions的范围内。例如，具有[SSA控制流语义](#CONTROL-FLOW-AND-SSAFG-REGIONS)的Regions中的值的范围根据[SSA dominance](https://en.wikipedia.org/wiki/Dominator_(graph_theory)).]的标准定义来约束另一个例子是[IsolatedFromAbove特征](Traits.md#solatedfrom上述)，它限制直接访问包含Regions中定义的值。

函数标识符和映射标识符与[SymbolsAndSymbolTables](SymbolsAndSymbolTables)相关联，并具有依赖于符号属性的作用域规则。

## 方言

方言是参与和扩展MLIR生态系统的机制。它们允许定义新的[Operation](#Operation)，以及[属性](#属性)和[类型](#type-system)。每种方言都被赋予了唯一的“命名空间”，该命名空间位于每个定义的属性/Operation/类型的前缀。例如，[Affine方言](Dialects/Affine.md)定义了命名空间：`affine`。

MLIR允许在一个模块中共存多种方言，甚至包括主树之外的方言。方言是由某些通道产生和使用的。MLIR提供了一个[框架](DialectConversion.md)来在不同方言之间和内部进行转换。

MLIR支持的一些方言：

* [仿射方言](方言/Affine.md)
* [GPU方言](方言/GPU.md)
* [LVM方言](方言/LVM.md)
* [SPIR-V方言](方言/SPIR-V.md)
* [标准方言](方言/Standard.md)
* [矢量方言](方言/Vector.md)

### 目标特定Operation

方言提供了一种模块化的方式，目标可以通过MLIR直接公开特定于目标的Operation。例如，一些目标会经历LLVM。LLVM对于某些与目标无关的Operation(例如，带有溢出检查的加法)有一组丰富的内部功能，并为它支持的目标提供对特定于目标的Operation的访问(例如，向量置换Operation)。MLIR中的LLVM内部特性通过以“llvm”开头的Operation表示。名字。

示例：

```mlir
// LLVM: %x = call {i16, i1} @llvm.sadd.with.overflow.i16(i16 %a, i16 %b)
%x:2 = "llvm.sadd.with.overflow.i16"(%a, %b) : (i16, i16) -> (i16, i1)
```

这些Operation仅在将LLVM作为后端目标时才起作用(例如，针对CPU和
GPU)，并且需要与这些内部的LLVM定义保持一致。

## Operation

语法：

```
operation         ::= op-result-list? (generic-operation | custom-operation)
                      trailing-location?
generic-operation ::= string-literal `(` value-use-list? `)`  successor-list?
                      (`(` region-list `)`)? attribute-dict? `:` function-type
custom-operation  ::= bare-id custom-operation-format
op-result-list    ::= op-result (`,` op-result)* `=`
op-result         ::= value-id (`:` integer-literal)
successor-list    ::= successor (`,` successor)*
successor         ::= caret-id (`:` bb-arg-list)?
region-list       ::= region (`,` region)*
trailing-location ::= (`loc` `(` location `)`)?
```

MLIR引入了一个称为*Operation*的统一概念，以支持描述许多不同级别的抽象和计算。MLIR中的Operation是完全可扩展的(没有固定的Operation列表)，并且具有特定于应用程序的语义。例如，MLIR支持[目标无关的Operation](operations](Dialects/Standard.md#memory-operations)，/Affine.md)和[目标特定的机器Operation](#目标特定的Operation)。

Operation的内部表示很简单：Operation由唯一的字符串(如`dim`、`tf.Conv2d`、`x86.repmovsb`、`ppc.eieio`等)标识，可以返回零个或多个结果，接受零个或多个Operation数，可以有零个或多个属性，可以有零个或多个后继者，可以有零个或多个封闭的[Regions](#Regions)。通用打印形式从字面上包括所有这些元素，并使用函数类型来指示结果和Operation数的类型。

示例：

```mlir
// An operation that produces two results.
// The results of %result can be accessed via the <name> `#` <opNo> syntax.
%result:2 = "foo_div"() : () -> (f32, i32)

// Pretty form that defines a unique name for each result.
%foo, %bar = "foo_div"() : () -> (f32, i32)

// Invoke a TensorFlow function called tf.scramble with two inputs
// and an attribute "fruit".
%2 = "tf.scramble"(%result#0, %bar) {fruit: "banana"} : (f32, i32) -> f32
```

除了上面的基本语法之外，方言还可以注册已知的Operation。这允许这些方言支持*自定义装配表单*进行解析和打印Operation。在下面列出的Operation集中，我们显示了这两种形式。

### 终结器Operation

这些是一类特殊的Operation，例如[branches](Dialects/Standard.md#terminator-operations).，用于*必须*终止block这些Operation还可能具有后继列表([块](块数)及其参数)。

示例：

```mlir
// Branch to ^bb1 or ^bb2 depending on the condition %cond.
// Pass value %v to ^bb2, but not to ^bb1.
"cond_br"(%cond)[^bb1, ^bb2(%v : index)] : (i1) -> ()
```

### 模块

```
module ::= `module` symbol-ref-id? (`attributes` attribute-dict)? region
```

MLIR模块代表顶级容器Operation。它包含一个[SSACFGRegions](#control-flow-and-ssafg-region)，其中包含一个block，可以包含任何Operation。此Regions内的Operation不能隐式捕获模块外部定义的值，即模块为[IsolatedFromAbove](Traits.md#isolatedfromabove).模块有一个可选的[Symbol Name](SymbolsAndSymbolTables.md)，可用于在Operation中引用它们。

### 功能

MLIR函数是名称包含单个[SSACFGRegions](#control-flow-and-ssafg-Regions)的Operation。此Regions内的Operation不能隐式捕获函数外部定义的值，即函数为[IsolatedFromAbove](Traits.md#isolatedfromabove).所有外部引用都必须使用建立符号连接的函数参数或属性(例如，通过字符串属性(如[SymbolRefAttr](#symbol-reference-attribute))：)按名称引用的符号

```
function ::= `func` function-signature function-attributes? function-body?

function-signature ::= symbol-ref-id `(` argument-list `)`
                       (`->` function-result-list)?

argument-list ::= (named-argument (`,` named-argument)*) | /*empty*/
argument-list ::= (type attribute-dict? (`,` type attribute-dict?)*) | /*empty*/
named-argument ::= value-id `:` type attribute-dict?

function-result-list ::= function-result-list-parens
                       | non-function-type
function-result-list-parens ::= `(` `)`
                              | `(` function-result-list-no-parens `)`
function-result-list-no-parens ::= function-result (`,` function-result)*
function-result ::= type attribute-dict?

function-attributes ::= `attributes` attribute-dict
function-body ::= region
```

外部函数声明(在引用其他模块中声明的函数时使用)没有正文。虽然MLIR文本形式为函数参数提供了很好的内联语法，但它们在内部表示为该Regions第一个block的“block参数”。

只能在函数参数、结果或函数本身的属性字典中指定方言属性名称。

示例：

```mlir
// External function definitions.
func @abort()
func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// A function that returns its argument twice:
func @count(%x: i64) -> (i64, i64)
  attributes {fruit: "banana"} {
  return %x, %x: i64, i64
}

// A function with an argument attribute
func @example_fn_arg(%x: i32 {swift.self = unit})

// A function with a result attribute
func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

// A function with an attribute
func @example_fn_attr() attributes {dialectName.attrName = false}
```

## 街区

语法：

```
block           ::= block-label operation+
block-label     ::= block-id block-arg-list? `:`
block-id        ::= caret-id
caret-id        ::= `^` suffix-id
value-id-and-type ::= value-id `:` type

// Non-empty list of names and types.
value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

block-arg-list ::= `(` value-id-and-type-list? `)`
```

*街区*是Operation的有序列表，以单个[结束符Operation](#结束符-Operation)结束。在[SSACFGRegions](#CONTROL-FLOW-AND-SSAFG-REGIONS)中，每个block代表一个编译器[基础block](https://en.wikipedia.org/wiki/Basic_block)，block内部指令按顺序执行，结束符Operation实现基本块之间的控制流分支。

MLIR中的块接受block参数列表，以类似函数的方式表示。block参数绑定到由单个Operation的语义指定的值。Regions的入口block参数也是该Regions的参数，绑定到这些参数的值由包含Operation的语义决定。其他块的block参数由终止符Operation的语义确定，例如分支，它以block为后继。在具有[控制流](#control-flow-and-ssafg-region)的Regions中，MLIR利用此结构隐式表示依赖于控制流的值的通过，而没有传统SSA表示中PHI节点的复杂细微差别。请注意，与控制流无关的值可以直接引用，不需要通过block参数传递。

下面是一个显示分支、返回和block参数的简单示例函数：

```mlir
func @simple(i64, i1) -> i64 {
^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
  cond_br %cond, ^bb1, ^bb2

^bb1:
  br ^bb3(%a: i64)    // Branch passes %a as the argument

^bb2:
  %b = addi %a, %a : i64
  br ^bb3(%b: i64)    // Branch passes %b as the argument

// ^bb3 receives an argument, named %c, from predecessors
// and passes it on to bb4 along with %a. %a is referenced
// directly from its defining operation and is not passed through
// an argument of ^bb3.
^bb3(%c: i64):
  br ^bb4(%c, %a : i64, i64)

^bb4(%d : i64, %e : i64):
  %0 = addi %d, %e : i64
  return %0 : i64   // Return is also a terminator.
}
```

**上下文：**与传统的“PHI节点是Operation”SSAIR(如LLVM)相比，“block参数”表示从IR中消除了许多特殊情况。例如，ssa的[Parallel Copy semantics](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.524.5461&rep=rep1&type=pdf)]一目了然，函数参数也不再是特例：它们成为条目block[[More rationale](Rationale/Rationale.md#block-arguments-vs-phi-nodes)].]的参数块也是一个不能由Operation表示的基本概念，因为在Operation中定义的值不能在Operation之外访问。

## Regions

### 定义

Regions是MLIR[块](块数)的有序列表。Regions内的语义不是由IR强加的。相反，包含Operation定义它包含的Regions的语义。MLIR目前定义了两种Regions：[SSACFGRegions](#control-flow-and-sacfg-Regions)和[图形Regions](#Graphs-Regions)，前者描述块之间的控制流，后者不需要block之间的控制流。使用[RegionKindInterface](Interfaces.md#regionkindinterfaces).描述Operation中的Regions类型

Regions没有名称或地址，只有Regions中包含的块有名称或地址。Regions必须包含在Operation中，并且没有类型或属性。该地区的第一个block是一种特殊的block，被称为“入口block”。加入block的论点也是该地区本身的论点。条目block不能被列为任何其他block的继任者。Regions的语法如下：

```
region ::= `{` block* `}`
```

函数体是Regions的一个示例：它由块的CFG组成，并且具有其他类型的Regions可能没有的附加语义限制。例如，在函数体中，block终止符必须分支到不同的block，或者从一个函数返回，其中`rereturn‘参数的类型必须与函数签名的结果类型匹配。同样，函数参数必须与Regions参数的类型和计数匹配。一般而言，Regions运算可以任意定义这些对应关系。

### 值范围界定

Regions提供程序的分层封装：不可能引用(即分支到)与引用的源不在同一Regions的block，即终止符Operation。类似地，Regions为值可见性提供了一个自然的作用域：Regions中定义的值不会转义到封闭Regions(如果有的话)。默认情况下，只要封闭Operation的Operation数引用这些值是合法的，Regions内的Operation就可以引用在Regions外定义的值，但这可以使用特征(如[OpTrait：：IsolatedFromAbove](Traits.md#isolatedfromabove)，或自定义验证器)进行限制。

示例：

```mlir
  "any_op"(%a) ({ // if %a is in-scope in the containing region...
	 // then %a is in-scope here too.
    %new_value = "another_op"(%a) : (i64) -> (i64)
  }) : (i64) -> (i64)
```

MLIR定义了一个跨层次Operation的广义“层次优势”概念，并定义了值是否在“作用域”内以及是否可以由特定Operation使用。同一Regions中的另一个Operation是否可以使用某个值，由Regions的类型来定义。当且仅当父级可以使用某个Regions中定义的值时，该Regions中定义的值才能由在同一Regions中有父级的Operation使用。由Regions的参数定义的值始终可以由深入包含在该Regions中的任何Operation使用。在Regions中定义的值永远不能在Regions之外使用。

### 控制流和SSACFGRegions

在MLIR中，Regions的控制流语义由[RegionKind：：SSACFG](Interfaces.md#regionkindinterfaces).表示非正式地，这些Regions支持Regions中的Operation“顺序执行”的语义。在Operation执行之前，其Operation数具有明确定义的值。Operation执行后，Operation数具有相同的值，结果也具有明确定义的值。在一个Operation执行之后，block中的下一个Operation会一直执行，直到该Operation是block结尾处的终结符Operation，在这种情况下，会执行其他一些Operation。确定要执行的下一条指令是“传递控制流”。

通常，当控制流传递到Operation时，MLIR不会限制控制流何时进入或退出该Operation中包含的Regions。但是，当控制流进入一个地域时，总是从该地域的第一个block开始，称为*条目*block。结束每个block的终止符Operation通过显式指定block的后继块来表示控制流。控制流只能传递给指定的后继块之一，就像在`分支‘Operation中一样，或者像在`rereturn’Operation中一样返回到包含Operation。没有后继的终结器Operation只能将控制传递回包含Operation。在这些限制范围内，终止符Operation的特定语义由所涉及的特定方言Operation决定。未作为终止符Operation的后续项列出的块(block条目除外)被定义为不可访问，并且可以在不影响包含Operation的语义的情况下移除。

虽然控制流始终通过入口block进入一个Regions，但是控制流可以通过任何带有合适终止符的block退出一个Regions。标准方言利用这一功能来定义单入口多出口Regions的Operation，可能流经该Regions中的不同区块，并通过任何block以“返回”Operation退出。此行为类似于大多数编程语言中的函数体。此外，控制流也可能不会到达block或Regions的末尾，例如，如果函数调用没有返回。

示例：

```mlir
func @accelerator_compute(i64, i1) -> i64 { // An SSACFG region
^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
  cond_br %cond, ^bb1, ^bb2

^bb1:
  // This def for %value does not dominate ^bb2
  %value = "op.convert"(%a) : (i64) -> i64
  br ^bb3(%a: i64)    // Branch passes %a as the argument

^bb2:
  accelerator.launch() { // An SSACFG region
    ^bb0:
      // Region of code nested under "accelerator.launch", it can reference %a but
      // not %value.
      %new_value = "accelerator.do_something"(%a) : (i64) -> ()
  }
  // %new_value cannot be referenced outside of the region

^bb3:
  ...
}
```

#### 具有多个Regions的Operation

包含多个Regions的Operation还完全确定这些Regions的语义。具体地说，当控制流被传递到Operation时，它可以将控制流传输到任何包含的Regions。当控制流退出Regions并返回到包含Operation时，包含Operation可以将控制流传递到同一Operation中的任何Regions。Operation还可以同时将控制流传递到多个包含的Regions。Operation还可以将控制流传递到在其他Operation中指定的Regions，特别是那些定义了给定Operation在调用Operation中使用的值或符号的Regions。该控制通道通常独立于控制流通过包含Regions的基本块的通道。

#### 闭合

Regions允许定义创建闭包的Operation，例如通过将Regions的主体“装箱”到它们生成的值中。它仍然由Operation来定义其语义。请注意，如果Operation触发Regions的异步执行，则由Operation调用者负责等待Regions执行，以确保任何直接使用的值保持活动状态。

### 图形区

在多层语义检索中，Regions内的图状语义由[RegionKind：：Graph](Interfaces.md#regionkindinterfaces).表示图Regions适用于没有控制流的并发语义，或者适用于对通用有向图数据结构进行建模。图形Regions适用于表示耦合值之间的循环关系，其中这些关系没有基本顺序。例如，图形Regions中的Operation可以表示具有表示数据流的值的独立控制线程。与MLIR一样，Regions的特定语义完全由其包含Operation决定。图形Regions只能包含单个基本block(条目block)。

**基本原理：**目前图域任意限定为单个基础block，但没有具体的语义原因。添加此限制是为了更容易地稳定PASS基础设施和处理图形Regions的常用PASS，以便正确处理反馈循环。如果未来出现需要的用例，可能会允许多个blockRegions。

在图形Regions中，MLIROperation自然表示节点，而每个MLIR值表示连接单个源节点和多个目的节点的多边。在该Regions中定义为Operation结果的所有值都在该Regions的范围内，并且可以由该Regions中的任何其他Operation访问。在图形Regions中，block内的Operation顺序和Regions中的块的顺序在语义上没有意义，并且非终止符Operation可以自由地重新排序，例如通过规范化。其他类型的图，例如具有多个源节点和多个目的节点的图，也可以通过将图的边表示为MLIROperation来表示。

请注意，循环可以出现在图形Regions中的单个block内，也可以出现在基本块之间。

```mlir
"test.graph_region"() ({ // A Graph region
  %1 = "op1"(%1, %3) : (i32, i32) -> (i32)  // OK: %1, %3 allowed here
  %2 = "test.ssacfg_region"() ({
	 %5 = "op2"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> (i32) // OK: %1, %2, %3, %4 all defined in the containing region
  }) : () -> (i32)
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)  // OK: %4 allowed here
  %4 = "op3"(%1) : (i32) -> (i32)
}) : () -> ()
```

### 论据和结果

一个地区的第一个block的论据被视为该地区的论据。这些参数的来源由父Operation的语义定义。它们可能对应于Operation本身使用的一些值。

Regions生成值列表(可能为空)。Operation语义定义了Regions结果和Operation结果之间的关系。

## 类型系统

MLIR中的每个值都有一个由下面的类型系统定义的类型。有许多基元类型(如整数)，也有用于张量和内存缓冲区的聚合类型。MLIR[内置类型](#builtin-type)不包括结构、数组或字典。

MLIR具有开放的类型系统(即没有固定的类型列表)，并且类型可以具有特定于应用程序的语义。例如，MLIR支持一组[方言类型](#方言类型)。

```
type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// This is a common way to refer to a value with a specified type.
ssa-use-and-type ::= ssa-use `:` type

// Non-empty list of names and types.
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*
```

### 类型别名

```
type-alias-def ::= '!' alias-name '=' 'type' type
type-alias ::= '!' alias-name
```

MLIR支持为类型定义命名别名。类型别名是可以用来代替它定义的类型的标识符。这些别名*必须*应在使用之前定义。别名不能包含‘.’，因为这些名称是为[方言类型](#Dialect-Types)保留的。

示例：

```mlir
!avx_m128 = type vector<4 x f32>

// Using the original type.
"foo"(%x) : vector<4 x f32> -> ()

// Using the type alias.
"foo"(%x) : !avx_m128 -> ()
```

### 方言类型

与Operation类似，方言可以定义类型系统的自定义扩展。

```
dialect-namespace ::= bare-id

opaque-dialect-item ::= dialect-namespace '<' string-literal '>'

pretty-dialect-item ::= dialect-namespace '.' pretty-dialect-item-lead-ident
                                              pretty-dialect-item-body?

pretty-dialect-item-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'
pretty-dialect-item-body ::= '<' pretty-dialect-item-contents+ '>'
pretty-dialect-item-contents ::= pretty-dialect-item-body
                              | '(' pretty-dialect-item-contents+ ')'
                              | '[' pretty-dialect-item-contents+ ']'
                              | '{' pretty-dialect-item-contents+ '}'
                              | '[^[<({>\])}\0]+'

dialect-type ::= '!' opaque-dialect-item
dialect-type ::= '!' pretty-dialect-item
```

可以用详细的形式指定方言类型，例如：

```mlir
// LLVM type that wraps around llvm IR types.
!llvm<"i32*">

// Tensor flow string type.
!tf.string

// Complex type
!foo<"something<abcd>">

// Even more complex type
!foo<"something<a%%123^^^>>>">
```

足够简单的方言类型可以使用Pretty格式，这是一种较轻的语法，相当于上面的形式：

```mlir
// Tensor flow string type.
!tf.string

// Complex type
!foo.something<abcd>
```

需要足够复杂的方言类型才能使用详细形式来实现一般性。例如，上面显示的更复杂的类型在较轻的语法中不会有效：`！foo.thing<a%%123^^>>`，因为它包含在较轻的语法中不允许使用的字符，以及不平衡的`<>`字符。

请参阅[here](Tutorials/DefiningAttributesAndTypes.md)了解如何定义方言类型。

### 内置类型

内置类型是以内置方言定义的[方言类型](#Dialect-Types)的核心集合，因此可供MLIR的所有用户使用。

```
builtin-type ::=      complex-type
                    | float-type
                    | function-type
                    | index-type
                    | integer-type
                    | memref-type
                    | none-type
                    | tensor-type
                    | tuple-type
                    | vector-type
```

#### 复杂类型

语法：

```
complex-type ::= `complex` `<` type `>`
```

`Complex‘类型的值表示具有参数化元素类型的复数，该元素类型由该元素类型的实值和虚值组成。元素必须是浮点或整数标量类型。

示例：

```mlir
complex<f32>
complex<i32>
```

#### 浮点类型

语法：

```
// Floating point.
float-type ::= `f16` | `bf16` | `f32` | `f64` | `f80` | `f128`
```

MLIR支持如上所述广泛使用的特定宽度的浮点类型。

#### 函数类型

语法：

```
// MLIR functions can return multiple values.
function-result-type ::= type-list-parens
                       | non-function-type

function-type ::= type-list-parens `->` function-result-type
```

MLIR支持一级函数：例如，[`constant`operation](Dialects/Standard.md#stdconstant-constantop)产生函数的地址作为值。该值可以传递到函数或从函数返回，使用[block参数](#块)跨控制流边界合并，并使用[`operation](Dialects/Standard.md#call-indirect-operation)._indirect`块]调用

函数类型还用于指示[Operation](#Operation)的参数和结果。

#### 索引类型

语法：

```
// Target word-sized integer.
index-type ::= `index`
```

`index`类型是一个无符号整数，其大小等于目标([rationale](Rationale/Rationale.md#integer-signedness-semantics))的自然机器字，并由MLIR中的仿射构造使用。与固定大小的整数不同，它不能用作向量([rationale](Rationale/Rationale.md#index-type-disallowed-in-vector-types)).的元素

**基本原理：**具体平台位宽的整数表示大小、维度和下标比较实用。

#### 整数类型

语法：

```
// Sized integers like i1, i4, i8, i16, i32.
signed-integer-type ::= `si` [1-9][0-9]*
unsigned-integer-type ::= `ui` [1-9][0-9]*
signless-integer-type ::= `i` [1-9][0-9]*
integer-type ::= signed-integer-type |
                 unsigned-integer-type |
                 signless-integer-type
```

MLIR支持任意精度整数类型。整数类型具有指定的宽度，并且可以具有符号语义。

**基本原理：**低精度整数(如`i2`、`i4`等)适用于低精度推理芯片，任意精度整数适用于硬件综合(13位乘法器比16位乘法器便宜/小很多)。

TODO：需要决定量化整数的表示形式
([初始thoughts](Rationale/Rationale.md#quantized-integer-operations)).

#### Memref类型

语法：

```
memref-type ::= ranked-memref-type | unranked-memref-type

ranked-memref-type ::= `memref` `<` dimension-list-ranked type
                      (`,` layout-specification)? (`,` memory-space)? `>`

unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`

stride-list ::= `[` (dimension (`,` dimension)*)? `]`
strided-layout ::= `offset:` dimension `,` `strides: ` stride-list
semi-affine-map-composition ::= (semi-affine-map `,` )* semi-affine-map
layout-specification ::= semi-affine-map-composition | strided-layout
memory-space ::= integer-literal /* | TODO: address-space-id */
```

`memref`类型是对内存Regions的引用(类似于缓冲区指针，但功能更强大)。可以分配、别名和释放memref指向的缓冲区。memref可用于从其引用的存储区读取数据和向其引用的存储区写入数据。Memref类型使用与张量类型相同的形状说明符。请注意，`memref<f32>`、`memref<0 x f32>`、`memref<1 x 0 x f32>`和`memref<0 x 1 x f32>`都是不同的类型。

允许`memref`具有未知的等级(例如`memref<*xf32>`)。未排名的memref的目的是允许外部库函数接收任何排名的memref参数，而无需根据排名对函数进行版本控制。此类型的其他使用是不允许的，或者将具有未定义的行为。

##### 未排序Memref的Codegen

除了上述情况外，强烈不鼓励在编码组中使用未排序的memref。Codegen负责为高性能、未排序的memref生成循环嵌套和专门指令，而Memref则负责隐藏秩，从而隐藏迭代数据所需的封闭循环的数量。然而，如果需要代码生成未排名的Memref，一种可能的方法是基于动态排名将其强制转换为静电排名类型。另一种可能的路径是发出以线性索引为条件的单个WHILE循环，并将线性索引去线性化到包含(未排序的)索引的动态阵列。虽然这是可能的，但预计在代码生成期间执行此Operation不是一个好主意，因为预计翻译成本会高得令人望而却步，而且在此级别进行优化也不值得。如果表现力是主要关注点，那么不管性能如何，将未排名的memrefs传递到外部C++库并实现与排名无关的逻辑应该会简单得多。

未排序的记忆引用可能会在未来提供表现力增益，并帮助弥合与未排序张量之间的差距。将不会期望未排名的MemRef暴露于编码生成，但是可以查询未排名的Memref的排名(为此将需要特殊的Operation)，并执行切换并强制转换成排名的Memref，作为编码生成的先决条件。

示例：

```mlir
// With static ranks, we need a function for each possible argument type
%A = alloc() : memref<16x32xf32>
%B = alloc() : memref<16x32x64xf32>
call @helper_2D(%A) : (memref<16x32xf32>)->()
call @helper_3D(%B) : (memref<16x32x64xf32>)->()

// With unknown rank, the functions can be unified under one unranked type
%A = alloc() : memref<16x32xf32>
%B = alloc() : memref<16x32x64xf32>
// Remove rank info
%A_u = memref_cast %A : memref<16x32xf32> -> memref<*xf32>
%B_u = memref_cast %B : memref<16x32x64xf32> -> memref<*xf32>
// call same function with dynamic ranks
call @helper(%A_u) : (memref<*xf32>)->()
call @helper(%B_u) : (memref<*xf32>)->()
```

布局规范的核心语法和表示是[半仿射映射](方言/Affine.md#半仿射映射)。此外，还支持语法糖，使某些布局规范更易于阅读。目前，“memref`”支持解析自动转换为半仿射映射的跨步形式。

memref的内存空间由特定于目标的整数索引指定。如果未指定内存空间，则使用默认内存空间(0)。默认空间是特定于目标的，但始终位于索引0。

TODO：MLIR最终将拥有允许象征性使用内存层次结构名称(例如，L3、L2、L1等)的目标方言但我们还没有详细说明那个机制的细节。在此之前，本文档冒充用‘Bare-id’来指代这些内存是有效的。

memref值的名义动态值包括分配的缓冲区地址，以及形状、布局映射和索引映射引用的符号。

Memref静电类型的示例

```mlir
// Identity index/layout map
#identity = affine_map<(d0, d1) -> (d0, d1)>

// Column major layout.
#col_major = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

// A 2-d tiled layout with tiles of size 128 x 256.
#tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>

// A tiled data layout with non-constant tile sizes.
#tiled_dynamic = affine_map<(d0, d1)[s0, s1] -> (d0 floordiv s0, d1 floordiv s1,
                             d0 mod s0, d1 mod s1)>

// A layout that yields a padding on two at either end of the minor dimension.
#padded = affine_map<(d0, d1) -> (d0, (d1 + 2) floordiv 2, (d1 + 2) mod 2)>


// The dimension list "16x32" defines the following 2D index space:
//
//   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
//
memref<16x32xf32, #identity>

// The dimension list "16x4x?" defines the following 3D index space:
//
//   { (i, j, k) : 0 <= i < 16, 0 <= j < 4, 0 <= k < N }
//
// where N is a symbol which represents the runtime value of the size of
// the third dimension.
//
// %N here binds to the size of the third dimension.
%A = alloc(%N) : memref<16x4x?xf32, #col_major>

// A 2-d dynamic shaped memref that also has a dynamically sized tiled layout.
// The memref index space is of size %M x %N, while %B1 and %B2 bind to the
// symbols s0, s1 respectively of the layout map #tiled_dynamic. Data tiles of
// size %B1 x %B2 in the logical space will be stored contiguously in memory.
// The allocation size will be (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2
// f32 elements.
%T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>

// A memref that has a two-element padding at either end. The allocation size
// will fit 16 * 64 float elements of data.
%P = alloc() : memref<16x64xf32, #padded>

// Affine map with symbol 's0' used as offset for the first dimension.
#imapS = affine_map<(d0, d1) [s0] -> (d0 + s0, d1)>
// Allocate memref and bind the following symbols:
// '%n' is bound to the dynamic second dimension of the memref type.
// '%o' is bound to the symbol 's0' in the affine map of the memref type.
%n = ...
%o = ...
%A = alloc (%n)[%o] : <16x?xf32, #imapS>
```

##### 索引空间

memref维列表定义了一个索引空间，在该索引空间内可以索引memref以访问数据。

##### 索引

使用进入由memref的维列表定义的多维索引空间的多维索引，通过memref类型访问数据。

示例

```mlir
// Allocates a memref with 2D index space:
//   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
%A = alloc() : memref<16x32xf32, #imapA>

// Loads data from memref '%A' using a 2D index: (%i, %j)
%v = load %A[%i, %j] : memref<16x32xf32, #imapA>
```

##### 索引贴图

索引映射是将多维索引从一个索引空间转换到另一个索引空间的一对一[半仿射映射](方言/仿射.md#半仿射映射)。例如，下图显示了一个索引映射，它使用符号`S0`和`S1`作为偏移量，将二维索引从2x2索引空间映射到3x3索引空间。

！索引Map example(包括/包括/IMG/Index-mapp.svg)

索引映射的域维和范围维的数量可以不同，但必须与映射在其上Operation的输入和输出索引空间的维度数量相匹配。指数空间总是非负的、整数的。此外，索引映射必须指定其映射到的每个范围维的大小。索引映射符号必须按顺序列出，首先是动态标注大小的符号，然后是其他所需的符号。

##### 布局图

布局图是[半仿射图](方言/Affine.md#半仿射图)，其通过将输入维度映射到它们从最大(变化最慢)到最次要(变化最快)的顺序来编码逻辑到物理索引空间的映射。因此，标识布局映射对应于以行为主的布局。标识布局映射不会影响MemRef类型标识，并且在构造时会被丢弃。即有显式标识映射的类型为`memref<？x？xf32，(i，j)->(i，j)>`与没有布局映射的类型`memref<？x？xf32>`严格相同。

布局图示例：

```mlir
// MxN matrix stored in row major layout in memory:
#layout_map_row_major = (i, j) -> (i, j)

// MxN matrix stored in column major layout in memory:
#layout_map_col_major = (i, j) -> (j, i)

// MxN matrix stored in a 2-d blocked/tiled layout with 64x64 tiles.
#layout_tiled = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
```

##### 仿射贴图合成

memref指定半仿射贴图合成作为其类型的一部分。半仿射地图合成是以零个或多个索引地图开始，以布局地图结束的半仿射地图的合成。合成必须是一致的：一个地图范围的维数必须与合成中下一个地图的域的维数相匹配。

memref类型中指定的半仿射映射组合从用于索引加载/存储Operation中的memref的访问映射到其他索引空间(即逻辑到物理索引映射)。每个[半仿射映射](方言/Affine.md)及其组成都要求是一对一的。

半仿射映射组合可用于相关性分析、存储器访问模式分析以及诸如矢量化、副本省略和就地更新之类的性能优化。如果没有为memref指定仿射映射组合，则假定为标识仿射映射。

##### 跨步MemRef

memref可以将步幅指定为其类型的一部分。跨距规范是一个整数值列表，可以是静电或`？`(动态大小写)。跨度以元素数编码(线性)存储器中沿特定维度的连续条目之间的距离。跨距规范是使用半仿射映射的等效跨距memref表示的语法糖。例如，`memref<42x16xf32，Offset：33，Strides：[1，64]>`指定`42`由`16``f32`元素组成的非连续内存Regions，使得：

封闭内存区的最小大小必须为`33+42*1+16*64=1066`个元素；
访问元素`(i，j)`的地址计算计算`33+i+64*j`
沿外维的两个连续元素之间的距离为‘1’个元素，沿外维的两个连续元素之间的距离为‘64’个元素。

这对应于存储器Regions的列主视图，并且在内部表示为类型`memref<42x16xf32，(i，j)->(33+i+64*j)>`。

跨度的指定不能有别名：给定n维跨度Memref，索引`(i1，.，in)`和`(j1，.，jn)`可以不引用相同的存储器地址，除非`i1==j1，.，in==jn`。

跨度内存引用表示对预分配数据的视图抽象。它们是用特种部队建造的，还没有引入。跨步内存引用是具有通用半仿射映射的内存引用的特殊子类，在降低到LLVM时对应于规范化的内存引用描述符。

#### 无类型

语法：

```
none-type ::= `none`
```

`none`类型是单位类型，即恰好具有一个可能值的类型，其中它的值没有定义的动态表示。

#### 张量型

语法：

```
tensor-type ::= `tensor` `<` dimension-list type `>`

dimension-list ::= dimension-list-ranked | (`*` `x`)
dimension-list-ranked ::= (dimension `x`)*
dimension ::= `?` | decimal-literal
```

具有张量类型的值表示聚合的N维数据值，并且具有已知的元素类型。它可以具有未知的排名(用`*`表示)，也可以具有带有维度列表的固定排名。每个维度可以是静电的非负十进制常量，也可以动态确定(`？`表示)。

MLIR张量类型的运行时表示是有意抽象的-您不能控制布局或获取指向数据的指针。对于低级缓冲区访问，MLIR的类型为[`memref`type](#memref-type)。这个抽象的运行时表示既保存张量数据值，也保存关于张量(潜在的动态)形状的信息。[`dim`operation](Dialects/Standard.md#dim-operation)从张量型的值返回维度的大小。

注意：张量类型声明中不允许使用十六进制整数文字，避免念力出现在`0xf32`和`0 x f32`之间。张量允许为零大小，并按其他大小处理，例如`张量<0 x 1 x I32>`和`张量<1 x 0 x I32>`是不同的类型。由于在某些其他类型中不允许零大小，因此在将张量降低到向量之前，应该将这些张量优化掉。

示例：

```mlir
// Tensor with unknown rank.
tensor<* x f32>

// Known rank but unknown dimensions.
tensor<? x ? x ? x ? x f32>

// Partially known dimensions.
tensor<? x ? x 13 x ? x f32>

// Full static shape.
tensor<17 x 4 x 13 x 4 x f32>

// Tensor with rank zero. Represents a scalar.
tensor<f32>

// Zero-element dimensions are allowed.
tensor<0 x 42 x f32>

// Zero-element tensor of f32 type (hexadecimal literals not allowed here).
tensor<0xf32>
```

#### 塔普尔·泰普

语法：

```
tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
```

`tuple`类型的值代表固定大小的元素集合，其中每个元素可以是不同的类型。

**tuple`类型：**虽然该类型是类型系统中的第一类，但([rationale](Rationale/Rationale.md#tuple-types)).没有提供对`基本原理`类型进行Operation的标准Operation

示例：

```mlir
// Empty tuple.
tuple<>

// Single element
tuple<f32>

// Many elements.
tuple<i32, f32, tensor<i1>, i5>
```

#### 矢量类型

语法：

```
vector-type ::= `vector` `<` static-dimension-list vector-element-type `>`
vector-element-type ::= float-type | integer-type

static-dimension-list ::= (decimal-literal `x`)+
```

向量类型表示SIMD样式向量，由特定于目标的Operation集(如AVX)使用。虽然最常见的用途是1D向量(例如Vector<16xF32>)，但我们也支持在支持它们的目标上的多维寄存器(如TPU)。

矢量形状必须是正十进制整数。

注意：向量类型声明中不允许使用十六进制整数字面，`Vector<0x42xi32>`被解释为形状为`(0，42)`的二维向量，不允许为零形状，因此`向量<0x42xi32>`无效。

## 属性

语法：

```
attribute-dict ::= `{` `}`
                 | `{` attribute-entry (`,` attribute-entry)* `}`
attribute-entry ::= dialect-attribute-entry | dependent-attribute-entry
dialect-attribute-entry ::= dialect-namespace `.` bare-id `=` attribute-value
dependent-attribute-entry ::= dependent-attribute-name `=` attribute-value
dependent-attribute-name ::= ((letter|[_]) (letter|digit|[_$])*)
                           | string-literal
```

属性是一种机制，用于在永远不允许变量的地方指定Operation的常量数据-例如，[`dim‘operation](Dialects/Standard.md#stddim-dimop)，的索引或卷积的步长。它们由名称和具体属性值组成。预期属性集、它们的结构和它们的解释都依赖于它们所依附的上下文。

主要有两类属性：从属属性和方言属性。从属属性从它们所附加的内容派生出它们的结构和含义；例如，`dim`Operation上的`index`属性的含义由`dim`Operation定义。另一方面，方言属性从特定的方言派生出它们的上下文和意义。方言属性的示例可以是指示自变量是self/context参数的`swft.self`函数自变量属性。此属性的上下文由`swft`方言定义，而不是由函数参数定义。

属性值由以下形式表示：

```
attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
```

### 属性价值别名

```
attribute-alias ::= '#' alias-name '=' attribute-value
attribute-alias ::= '#' alias-name
```

MLIR支持为属性值定义命名别名。属性别名是可以用来代替它定义的属性的标识符。这些别名*必须*应在使用之前定义。别名不能包含‘.’，因为这些名称是为[方言属性](#Dialect-Attribute-Values)保留的。

示例：

```mlir
#map = affine_map<(d0) -> (d0 + 10)>

// Using the original attribute.
%b = affine.apply affine_map<(d0) -> (d0 + 10)> (%a)

// Using the attribute alias.
%b = affine.apply #map(%a)
```

### 方言属性值

与Operation类似，方言可以定义自定义属性值。这些值的语法结构与自定义方言类型值相同，不同之处在于方言属性值用前导‘#’区分，而方言类型用前导‘！’区分。

```
dialect-attribute-value ::= '#' opaque-dialect-item
dialect-attribute-value ::= '#' pretty-dialect-item
```

方言属性值可以详细形式指定，例如：

```mlir
// Complex attribute value.
#foo<"something<abcd>">

// Even more complex attribute value.
#foo<"something<a%%123^^^>>>">
```

足够简单的方言属性值可以使用Pretty格式，这是一种较轻的语法，相当于上面的形式：

```mlir
// Complex attribute
#foo.something<abcd>
```

需要足够复杂的方言属性值才能使用详细形式来实现一般性。例如，上面显示的更复杂的类型在较轻的语法中是无效的：`#foo.thing<a%%123^^>>`，因为它包含在较轻的语法中不允许的字符，以及不平衡的`<>`字符。

有关如何定义方言属性值的信息，请参阅[here](Tutorials/DefiningAttributesAndTypes.md)。

### 内置属性值

内置属性是以内置方言定义的一组核心[方言属性](#Dialect-Attribute-Values)，因此可供MLIR的所有用户使用。

```
builtin-attribute ::=    affine-map-attribute
                       | array-attribute
                       | bool-attribute
                       | dictionary-attribute
                       | elements-attribute
                       | float-attribute
                       | integer-attribute
                       | integer-set-attribute
                       | string-attribute
                       | symbol-ref-attribute
                       | type-attribute
                       | unit-attribute
```

#### AffineMap属性

语法：

```
affine-map-attribute ::= `affine_map` `<` affine-map `>`
```

仿射贴图属性是表示仿射贴图对象的属性。

#### 数组属性

语法：

```
array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
```

数组属性是表示属性值集合的属性。

#### 博兰属性

语法：

```
bool-attribute ::= bool-literal
```

布尔属性是表示一位布尔值(TRUE或FALSE)的文字属性。

#### 字典属性

语法：

```
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
```

字典属性是表示命名属性值的排序集合的属性。元素按名称排序，并且每个名称在集合中必须是唯一的。

#### 元素属性

语法：

```
elements-attribute ::= dense-elements-attribute
                     | opaque-elements-attribute
                     | sparse-elements-attribute
```

元素属性是表示常量[Vector](#Vector-type)或[张量](#tensor-type)值的文字属性。

##### dense元素属性

语法：

```
dense-elements-attribute ::= `dense` `<` attribute-value `>` `:`
                             ( tensor-type | vector-type )
```

密集元素属性是常量向量或张量值的存储已被密集打包的元素属性。该属性支持存储整数或浮点元素，具有整数/索引/浮点元素类型。它还支持使用自定义方言字符串元素类型存储字符串元素。

##### 奥帕克元素属性

语法：

```
opaque-elements-attribute ::= `opaque` `<` dialect-namespace  `,`
                              hex-string-literal `>` `:`
                              ( tensor-type | vector-type )
```

不透明元素属性是值内容不透明的元素属性。该元素属性存储的常量的表示形式只能由创建它的方言理解，因此是可解码的。

注意：解析的字符串文字必须是十六进制形式。

##### 储蓄项属性

语法：

```
sparse-elements-attribute ::= `sparse` `<` attribute-value `,` attribute-value
                              `>` `:` ( tensor-type | vector-type )
```

稀疏元素属性是表示稀疏向量或张量对象的元素属性。这就是极少数元素为非零的地方。

该属性使用COO(坐标列表)编码来表示Elements属性的稀疏元素。索引通过形状为[N，ndims]的64位整数元素的二维张量存储，该张量指定包含非零值的稀疏张量中元素的索引。元素值通过形状为[N]的一维张量存储，该张量为索引提供相应的值。

示例：

```mlir
  sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>

// This represents the following tensor:
///  [[1, 0, 0, 0],
///   [0, 0, 5, 0],
///   [0, 0, 0, 0]]
```

#### float属性

语法：

```
float-attribute ::= (float-literal (`:` float-type)?)
                  | (hexadecimal-literal `:` float-type)
```

FLOAT属性是一个文字属性，表示指定的[FLOAT TYPE](#FLOAT-POINT-TYPE)的浮点值。它可以用十六进制形式表示，其中十六进制值被解释为底层二进制表示的位。此表单对于表示无穷大和NaN浮点值非常有用。为避免念力具有整数属性，十六进制文字*必须*后跟浮点类型以定义浮点属性。

示例：

```
42.0         // float attribute defaults to f64 type
42.0 : f32   // float attribute of f32 type
0x7C00 : f16 // positive infinity
0x7CFF : f16 // NaN (one of possible values)
42 : f32     // Error: expected integer type
```

#### 整体性属性

语法：

```
integer-attribute ::= integer-literal ( `:` (index-type | integer-type) )?
```

整数属性是表示指定整数或索引类型的整数值的文字属性。此属性的默认类型(如果未指定)为64位整数。

##### 整合集属性

语法：

```
integer-set-attribute ::= `affine_set` `<` integer-set `>`
```

整数集属性是表示整数集对象的属性。

#### 串属性

语法：

```
string-attribute ::= string-literal (`:` type)?
```

字符串属性是表示字符串文字值的属性。

#### 引用属性图标

语法：

```
symbol-ref-attribute ::= symbol-ref-id (`::` symbol-ref-id)*
```

符号引用属性是一个文字属性，表示对嵌套在具有`OpTrait：：SymbolTable`特性的Operation中的Operation的命名引用。因此，此引用由包含`OpTrait：：SymbolTable`特性的最近父Operation赋予意义。它可以可选地包含一组嵌套引用，这些嵌套引用进一步解析为嵌套在不同符号表中的符号。

该属性只能由[数组属性](#array-attribute)和[字典属性](#tionary-attribute)(包括顶级Operation属性字典)在内部保存，即不能有位置、扩展属性等其他属性类型。

**基本原理：**识别对全局数据的访问是实现高效多线程编译的关键。将全局数据访问限制为通过符号进行，并限制可以合法持有符号引用的位置，简化了有关这些数据访问的推理。

详见[`SymbolsAndSymbolTables`](SymbolsAndSymbolTables.md)。

#### Type属性

语法：

```
type-attribute ::= type
```

类型属性是表示[类型对象](#type-system)的属性。

#### Unit属性

```
unit-attribute ::= `unit`
```

单位属性是表示`unit`类型的值的属性。`unit`类型只允许一个值组成一个单一集合。此属性值用于表示仅因其存在而有意义的属性。

这种属性的一个示例可以是`swft.self`属性。此属性指示函数参数是self/context参数。它可以表示为[Boolean Attribute](#boolean-attribute)(true或false)，但是值false实际上不会带来任何值。参数可以是self/context，也可以不是。

```mlir
// A unit attribute defined with the `unit` value specifier.
func @verbose_form(i1) attributes {dialectName.unitAttr = unit}

// A unit attribute can also be defined without the value specifier.
func @simple_form(i1) attributes {dialectName.unitAttr}
```
