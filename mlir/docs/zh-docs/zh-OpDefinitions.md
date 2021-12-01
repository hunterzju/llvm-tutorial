# 定义专科行动(ODS)

MLIR除了专门化`mlir：：op`C++模板外，还支持以表驱动的方式定义操作和数据类型。这是通过[TableGen](https://llvm.org/docs/TableGen/index.html)，实现的，它既是一种通用语言，也是维护领域特定信息记录的工具。关于操作的事实被简明地指定到TableGen记录中，该记录将在编译器构建时扩展为等效的`mlir：：op`C++模板专门化。

本手册详细说明了以这种表驱动方式定义操作的所有可用机制。它的目标是成为规范，而不是教程。后者请参考[MLIR图形重写快速入门教程](tutorials/QuickstartRewrites.md)。

除了详细介绍每种机制外，本手册还尝试捕获最佳实践。它们以引用的项目符号的形式呈现。

## 动机

MLIR允许使用可插拔的方言，并且方言包含操作列表等。这种开放和可扩展的生态系统导致了“串式”类型的IR问题，例如，在优化和分析过程中重复的字符串比较、不直观的访问器方法(例如，具有更一般的返回类型的泛型/容易出错的`getOperand(3)`vs自文档化的`getStride()`)、没有默认参数的冗长和泛型构造函数、冗长的文本IR转储，等等。此外，操作验证为：

最好的情况：中央字符串到验证函数的映射，
中间情况：跨代码库重复验证，或者
最坏情况：没有验证功能。

修复方法是支持以表驱动的方式定义操作。然后，对于每种方言，我们可以有一个中心位置，其中包含您需要了解的关于每个OP的所有内容，包括它的约束、自定义组装形式等。此描述还用于生成助手函数和类，以允许构建、验证、解析、打印、分析等。

## 优势

与C++模板相比，这种表驱动方法有几个优点，包括但不限于：

* **单一的真理来源**：我们致力于将一项操作的所有事实都编码到记录中，让读者不需要在代码片段之间跳跃就能完全理解一项操作。
* **删除样板文件**：我们可以从记录中自动生成操作数/属性/结果getter方法、操作构建方法、操作验证方法，以及更多实用工具。这大大减少了定义新OP所需的样板。
* **促进自动生成**：这些操作信息记录的用途绝不局限于OP定义本身。我们可以使用它们来驱动许多其他组件的自动生成，比如计算图序列化。

## 表基因语法

我们使用TableGen作为指定操作信息的语言。TableGen本身只提供了写入记录的语法；在TableGen文件(通常带有文件名后缀`.td`)中允许的语法和构造可以在[here](https://llvm.org/docs/TableGen/ProgRef.html).中找到

* TableGen`class`类似于C++类，可以模板化和子类化。
* TableGen`def`类似于C++Object，可以通过专门化TableGen`class`(如`def MyDef：MyClass<.>；`)声明，也可以完全独立声明(如`def MyDef；`)。它不能进一步模板化或子类化。
* TableGen`dag`是元素的有向无环图的专用类型。`dag`有一个运算符和零个或多个参数。其语法为`(运算符arg0，arg1，argN)`。运算符可以是任何TableGen`def`；参数可以是任何内容，包括`dag`本身。我们可以将名称附加到操作符和参数，如`(MYOP：$OP_NAME MyArg：$Arg_NAME)`。

请参考[Language reference](https://llvm.org/docs/TableGen/ProgRef.html)]了解TableGen支持的所有类型和表达式。

## 行动定义

MLIR定义了几个常见的构造来帮助定义操作，并通过特殊的[TableGen backend](https://llvm.org/docs/TableGen/BackEnds.html#introduction)：[`OpDefinitionsGen`](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp).]提供它们的语义这些构造在[`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td).中定义主要有以下几个方面

* `Op`类：是定义操作的主要构造。有关操作的所有事实都是在专门化该类时在以下构造的帮助下指定的。
* `Dialect`类：属于一个逻辑组的操作使用相同的方言。`Dialect`类包含方言级别信息。
* `OpTrait`类层次结构：用于指定操作的特殊属性和约束，包括操作是否有副作用，或者其输出是否与输入具有相同的形状。
* `ins`/`outs`标记：这是`OpDefinitionsGen`后端内置的两个特殊标记。它们分别领导操作数/属性和结果的定义。
* `TypeConstraint`类层次结构：用于指定对操作数或结果的约束。一个值得注意的子类层次结构是`Type`，它代表常见C++类型的约束。
* `AttrConstraint`类层次结构：用于指定属性上的约束。一个值得注意的子类层次结构是`Attr`，它代表值属于普通类型的属性的约束。

操作的定义是将`Op`类专门化，将其需要的所有字段都具体化。例如，`tf.AvgPool`定义为

```tablegen
def TF_AvgPoolOp : TF_Op<"AvgPool", [NoSideEffect]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    TF_FpTensor:$value,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvertDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}
```

在下面，我们将描述所有需要的字段。支持字段的完整列表请参考`Op`类的定义。

### 操作名称

操作名称是MLIR中操作的唯一标识符，例如TensorFlow方言中加法操作的`tf.Add`。这相当于汇编语言中的助记符。它用于以文本格式进行解析和打印。它还用于图形重写中的模式匹配。

全操作名由方言名和OP名组成，前者通过方言提供，后者作为第二个模板参数提供给`Op`类。

### 操作文档

这包括一行`摘要`和较长的人类可读的`描述`。它们将用于驱动方言文档的自动生成。它们需要在操作的定义体中提供：

```tablegen
let summary = "...";

let description = [{
...
}];
```

`Description`应使用Markdown语法编写。

建议将文档放在开头，因为这有助于理解操作。

> * 将文档放在操作定义的开头
> * 总结应该简明扼要。它应该是不带尾随标点符号的一行。在描述中加入扩展的解释。


### 操作参数

有两种参数：操作数和属性。操作数是由其他操作生成的运行时值；而属性是编译时已知的常量值，包括两个类别：

自然属性：这些属性影响操作的行为(例如，用于卷积的填充)；
派生属性：这些属性不是定义操作所必需的，而是从操作信息派生而来。例如，文字的输出形状。这主要用于方便的界面生成或与其他框架/翻译的交互。

所有派生属性都应可物化为属性。也就是说，即使它们没有物化，也应该可以存储为属性。

操作数和属性都在`dag`类型的`arguments`中指定，以`ins`为首：

```tablegen
let arguments = (ins
  <type-constraint>:$<operand-name>,
  ...
  <attr-constraint>:$<attr-name>,
  ...
);
```

这里`<type-straint>`是来自`TypeConstraint`类层次结构的TableGen`def`。类似地，`<attr-straint>`是来自`AttrConstraint`类层次结构的TableGen`def`。有关更多信息，请参阅[约束](#约束)。

对操作数和属性的相对顺序没有要求；它们可以自由混合。操作数本身的相对顺序很重要。从每个命名参数将生成一个命名的getter，它返回具有返回类型的参数(对于属性，返回类型将从存储类型构造，而对于操作数，它将是`Value`)。每个属性的原始值(例如，存储的原始值)也可以通过生成的`<name>Attr`getter来访问，以便在较不适合用户友好的返回类型的转换过程中使用。

所有参数都应该命名为1)提供文档，2)驱动getter方法的自动生成，3)为其他地方(如约束)提供引用的句柄。

#### 可变操作数

要声明变量操作数，请将操作数的`TypeConstraint`用`Variadi<.>`包装起来。

通常，运算没有可变操作数，或者只有一个可变操作数。对于后一种情况，很容易推导出哪些动态操作数用于静电可变操作数定义。但是，如果一个操作具有多个可变长度操作数(可选或可变)，则在没有来自该操作的进一步信息的情况下，不可能将动态操作数赋予相应的静电可变操作数定义。因此，需要`SameVariadicOperandSize`或`AttrSizedOperandSegments`特性来表示所有可变长度操作数具有相同数量的动态值。

#### 可选操作数

要声明可选操作数，请将操作数的`TypeConstraint`用`Optional<.>`包装起来。

通常，操作没有可选操作数或只有一个可选操作数。对于后一种情况，很容易推导出哪些动态操作数用于静电操作数定义。但是，如果一个操作具有多个可变长度操作数(可选或可变)，则在没有来自该操作的进一步信息的情况下，不可能将动态操作数赋予相应的静电可变操作数定义。因此，需要`SameVariadicOperandSize`或`AttrSizedOperandSegments`特性来表示所有可变长度操作数具有相同数量的动态值。

#### 可选属性

若要声明可选属性，请使用`OptionalAttr<.>`包装该属性的`AttrConstraint`。

#### 具有默认值的属性

要使用默认值声明属性，请用`DefaultValuedAttr<.，“.”>`包装属性的`AttrConstraint`。

`DefaultValuedAttr`的第二个参数应该是包含C++默认值的字符串。例如，浮点默认值应指定为类似`“0.5F”`，整数数组默认值应指定为类似于`“{1，2，3}”`。

#### 限制属性

提供`Confined`作为通用机制，帮助对值类型带来的属性以外的属性进行进一步的约束建模。您可以使用`Confined`将更原始的约束组合成复杂的约束。例如，最小值必须为10的32位整数属性可以表示为`Confined<I32Attr，[IntMinValue<10>]>`。

目前，支持以下原语约束：

* `IntMinValue<N>`：指定整数属性大于等于`N`
* `IntMaxValue<N>`：指定整数属性小于等于`N`
* `ArrayMinCount<N>`：指定数组属性至少有`N`
元素
* `IntArrayNthElemEq<I，N>`：指定整数数组属性的`I`元素等于`N`
* `IntArrayNthElemMinValue<I，N>`：指定整数数组属性的`I`元素大于等于`N`

TODO：设计和实现更多原语约束

### 作业区

操作的地域在`dag`类型的`regions`内指定，以`region`为首：

```tablegen
let regions = (region
  <region-constraint>:$<region-name>,
  ...
);
```

#### 多变区

与用于各种操作数和结果的`Variadic`类类似，`VariadicRegion<.>`也可以用于地域。可变区域当前只能指定为区域列表中的最后一个区域。

### 运行结果

与操作数类似，结果在`dag`类型的`result`中指定，以`outs`为首：

```tablegen
let results = (outs
  <type-constraint>:$<result-name>,
  ...
);
```

#### 各种结果

与变量操作数类似，`变量<.>`也可以用于结果.类似地，多个变量的`SameVariadicResultSize`在同一操作中得到结果。

### 运营后继者

对于终止符操作，后继符是在`dag`类型的`sucessors`中指定的，前缀为`suggeror`：

```tablegen
let successors = (successor
  <successor-constraint>:$<successor-name>,
  ...
);
```

#### 多种多样的继承人

与用于各种操作数和结果的`Variadic`类类似，`VariadicSuccessor<.>`也可以用于后继。当前只能将可变继任者指定为继任者列表中的最后一个继任者。

### 运营特点和制约因素

特征是影响语法或语义的操作属性。MLIR C++在`mlir：：OpTrait`命名空间中对各种特征进行建模。

操作特征、[interfaces](Interfaces.md/#utilizing-the-ods-framework)，和涉及多个操作数/属性/结果的约束都作为第二个模板参数提供给`Op`类。它们应该派生自`OpTrait`类。有关更多信息，请参阅[约束](#约束)。

### 构建器方法

对于每个操作，都有几个基于参数和返回类型自动生成的构建器。例如，给定以下OP定义：

```tablegen
def MyOp : ... {
  let arguments = (ins
    I32:$i32_operand,
    F32:$f32_operand,
    ...,

    I32Attr:$i32_attr,
    F32Attr:$f32_attr,
    ...
  );

  let results = (outs
    I32:$i32_result,
    F32:$f32_result,
    ...
  );
}
```

将生成以下构建器：

```c++
// All result-types/operands/attributes have one aggregate parameter.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ArrayRef<Type> resultTypes,
                  ValueRange operands,
                  ArrayRef<NamedAttribute> attributes);

// Each result-type/operand/attribute has a separate parameter. The parameters
// for attributes are of mlir::Attribute types.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...);

// Each result-type/operand/attribute has a separate parameter. The parameters
// for attributes are raw values unwrapped with mlir::Attribute instances.
// (Note that this builder will not always be generated. See the following
// explanation for more details.)
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  APInt i32_attr, StringRef f32_attr, ...);

// Each operand/attribute has a separate parameter but result type is aggregate.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ArrayRef<Type> resultTypes,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...);

// All operands/attributes have aggregate parameters.
// Generated if return type can be inferred.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ValueRange operands, ArrayRef<NamedAttribute> attributes);

// (And manually specified builders depending on the specific op.)
```

第一种形式提供了基本的一致性，因此我们可以使用相同的形式创建OP，而不管确切的OP是什么。这对于实现声明性模式重写特别有用。

第二种和第三种形式很适合在手动编写的代码中使用，因为它们通过签名提供了更好的保证。

如果OP的任何一个属性的`Attr.rereturn Type`与`Attr.storageType`不同，并且我们知道如何从展开的值构建属性(即定义了`Attr.constBuilderCall`)，则会生成第三种形式。此外，对于第三种形式，如果后面出现在`arguments`列表中的属性具有默认值，则在声明中将提供默认值。目前对`BoolAttr`、`StrAttr`、`EnumAttr`有效，以后列表可以扩大。因此，如果可能，默认值属性应该放在`arguments`列表的末尾，以利用此功能。(此行为本质上是由于C++函数参数默认值的放置限制。)否则，仍将生成第三种形式的构建器，但不会在构建器签名中提供不在`arguments`列表末尾的属性的默认值。

如果满足以下条件，ODS将生成不需要指定返回类型的构建器

* OP实现InferTypeOpInterface接口；
* 所有返回类型要么是可构建类型，要么与给定操作数相同(例如，操作数和结果之间的`AllTypesMatch`约束)；

根据具体的OP，可能还存在其他构建器；完整列表请参考[生成的C++file](#run-mlir-tblgen-to-see-the-generated-content)]。

#### 自定义构建器方法

但是，如果上述情况不能满足所有需求，您可以在`builders`字段中定义额外的便捷构建方法，如下所示。

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val)>
  ];
}
```

`builders`字段是添加到Op类的自定义构建器的列表。在本例中，我们提供了一个方便的构建器，它接受浮点值而不是属性。ODS中的许多函数声明都使用`ins`前缀，这些函数声明使用TableGen[`dag`](#tablegen-语法)。下面是一个逗号分隔的类型列表(带引号的字符串)和前缀有`$`符号的名称。这将生成一个构建器方法的声明，如下所示：

```c++
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val);
};
```

请注意，该方法还有两个额外的前导参数。这些参数对于构造操作很有用。特别是，该方法必须使用要构造的操作的属性、操作数、区域和结果类型填充`state`。`builder`可以构造属于Op的任何IR对象，如类型或嵌套操作。由于类型和名称是在C++代码中生成的，因此它们应该是类型(在Op的命名空间中)和标识符(例如，`class`不是有效标识符)的有效C++构造。

构建器的实现可以直接在ODS中提供，使用TableGen代码挡路，如下所示。

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```

`builder`和`state`参数的等价物可以作为`$_builder`和`$_state`特殊变量。`ins`部分列出的命名参数可以直接获取，例如`val`。构建器的主体将通过替换特殊变量生成，否则应该是有效的C++。虽然对代码大小没有限制，但我们鼓励在ODS中只定义内联较短的构建器，并将较长构建器的定义放在C++文件中。

最后，如果某些参数需要默认值，可以使用`CArg`对类型和该值进行定义，如下所示。

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins CArg<"float", "0.5f">:$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```

生成的代码将在声明中使用默认值，但不会像C++要求的那样在定义中使用默认值。

```c++
/// Header file.
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val = 0.5f);
};

/// Source file.
MyOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
            float val) {
  state.addAttribute("attr", builder.getF32FloatAttr(val));
}
```

**已弃用：**`OpBuilder`类允许您将自定义构建器签名指定为原始字符串，而无需将参数分隔为不同的`dag`参数。同时支持`OpBuilder&`和`OperationState&`类型的前导参数，如果有，将不使用自动生成的前导参数。

### 自定义解析器和打印机方法

用于分析和打印操作的自定义程序集表单的函数。

### 自定义验证器代码

对于OP的各个实体指定的[Constraints](#Constraints)，系统会自动生成验证码。要执行*附加内容*验证，您可以使用

```tablegen
let verifier = [{
  ...
}];
```

放置在`verifier`中的代码将在自动生成的验证码之后调用。不应依赖不包括“验证者”的性状验证顺序。

### 声明性程序集格式

操作的自定义程序集形式可以在与操作操作数、属性等匹配的声明性字符串中指定，并且能够表达构建操作所需解析的附加信息：

```tablegen
def CallOp : Std_Op<"call", ...> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}
```

该格式由三个组件组成：

#### 指令

指令是一种内置函数，具有一组可选的参数。可用的指令如下：

* `attr-diced`
  
  - 表示操作的属性字典。
* `attr-dict-with-keyword`
  
  - 表示操作的属性字典，但在字典前面加上一个`Attributes`关键字。
* `custom`<UserDirective>(Params)
  
  - 表示用户用C++实现的自定义指令。
  - 有关详细信息，请参阅下面的[自定义指令](#Custom-Directions)部分。
* `function-type`(input，result)
  
  - 将`inputs`和`result`参数格式化为[函数类型](方言/Builtin.md/#functiontype)。
  - `inputs`和`result`的约束与`type`指令的`input`相同。
* `操作数`
  
  - 表示运算的所有操作数。
* `ref`(输入)
  
  - 表示对必须已解析的变量或指令的引用，该变量或指令可以用作`custom`指令的参数。
  - 用于将以前解析的实体传递给自定义指令。
  - 输入可以是除`function-type`和`custom`之外的任何指令或变量。
* `区域`
  
  - 表示操作的所有区域。
* `结果`
  
  - 表示操作的所有结果。
* “接班人”，你们是谁？
  
  - 表示操作的所有后继者。
* `type`(输入)
  
  - 表示给定输入的类型。
  - `input`必须是操作数或Result[Variable](#Variables)、`Operands`指令或`Results`指令。

#### 文字

文字可以是关键字，也可以是用\`\`括起来的标点符号。

以下是一组有效的标点符号：

`：`、`=`、`<`、`>`、`(`、`)`、`{`、`}`、`[`、`]`、`->`、`？`、`+`、`*`

以下是有效的空格标点符号：

`\n`，``

`\n‘文本发出一个换行符，缩进到操作开始处。示例如下所示：

```tablegen
let assemblyFormat = [{
  `{` `\n` ` ` ` ` `this_is_on_a_newline` `\n` `}` attr-dict
}];
```

```mlir
%results = my.operation {
  this_is_on_a_newline
}
```

空文字\`\`可用于删除在某些文字元素后隐式插入的空格，如`)`/`]`/等。例如，“`]`”可能会导致输出为`]`，它不是格式中的最后一个元素。在这种情况下，“`]`\`\`”会修剪尾随空格。

#### 变量

变量是在操作本身注册的实体，即参数(属性或操作数)、区域、结果、后继者等。在上面的`CallOp`示例中，变量为`$callee`和`$args`。

属性变量使用其各自的值类型打印，除非该值类型是可构建的。在这些情况下，属性的类型被省略。

#### 自定义指令

声明性程序集格式规范允许在格式化操作时处理大多数常见情况。对于需要或希望以声明性语法不支持的形式指定部分操作的操作，可以指定自定义指令。自定义指令实质上允许用户使用C++打印和解析以其他方式声明性指定的格式的子节。查看上面的自定义指令规范：

```
custom-directive ::= `custom` `<` UserDirective `>` `(` Params `)`
```

自定义指令有两个主要部分：`UserDirective`和`Params`。自定义指令在生成格式的C++代码时转换为对`print*`和`parse*`方法的调用。`UserDirective`是作为这两个调用的后缀的标识符，即`Custom<MyDirective>(.)`会导致在解析器和打印机内部分别调用`parseMyDirective`和`printMyDirective`。`Params`可以是变量(即属性、操作数、后续变量等)、类型指令和`attr-dicate`的任意组合。类型指令必须引用变量，但该变量不必同时是自定义指令的参数。

`parse<UserDirective>`方法的参数首先是对`OpAsmParser`(`OpAsmParser&`)的引用，其次是格式中指定的参数对应的一组输出参数。声明性参数到`parse`方法参数的映射详细说明如下：

* 可变属性
  - 单身：可以使用属性存储类型>(e.g.属性).
  - 可选：可使用属性存储类型>(e.g.属性).
* 歌剧及变体
-Single：`OpAsmParser：：OperandType&`
  - 可选：可选的Asmper：操作类型>
-变量：`SmallVectorImpl<OpAsmParser：：OperandType>&`
* 引用指令
  - 引用指令使用与输入操作数相同的映射传递给解析器。例如，单个地域将作为`Region&`传递。
* 地区可变
  - 单曲：`Region&`
-变量：`SmallVectorImpl<std：：Unique_ptr<Region>>&`
* 后继变量
  - 单张：`Block*&`
  - 变量：`SmallVectorImpl<挡路*>&`
* 类型指令
  - 单曲：`类型&`
  - 可选：“Type”
-Variadi：`SmallVectorImpl<Type>&`
* 指令：`NamedAttrList‘

如果变量是可选的，则只有在该变量存在的情况下才应指定值。否则，取值应为`None`或NULL。

`print<UserDirective>`方法的参数首先是对`OpAsmPrinter`(`OpAsmPrinter&`)的引用，其次是op(例如`FooOp op`，也可以是`operation*op`)，最后是一组与格式中指定的参数相对应的输出参数。声明性参数到`print`方法参数的映射详细说明如下：

* 可变属性
  - 单曲：可实现属性存储类型>(e.g.属性)
  - 可选：可为属性存储类型>(e.g.属性)
* 歌剧及变体
  - 单张：`Value`
  - 可选：“价值”
-各种：`OperandRange‘
* 引用指令
  - 引用指令使用与输入操作数相同的映射传递给打印机。例如，单个地域将作为`Region&`传递。
* 地区可变
  - 单曲：`Region&`
-变量：`MutableArrayRef<Region>`
* 后继变量
  - 单张：`Block*`
-Variatic：`SuccessorRange‘
* 类型指令
  - 单曲：`Type`
  - 可选：“类型”
-Variatic：`TypeRange`
* 指令：`DictionaryAttr‘

如果变量是可选的，则提供的值可能为NULL。

#### 可选组

在某些情况下，操作可能具有“可选”信息，例如属性或一组空的可变操作数。在这些情况下，可以基于该信息的存在将汇编格式的一部分标记为“可选”。可选组的定义如下：

```
optional-group: `(` elements `)` (`:` `(` else-elements `)`)? `?`
```

可选组的`elements`有以下要求：

* 组的第一个元素必须是属性、文字、操作数或区域。
  - 这是因为第一个元素必须是可选的可解析元素。
* 组中必须正好有一个参数变量或类型指令标记为组的锚点。
  - 锚是其存在控制是否应该打印/解析组的元素。
  - 元素通过添加尾随的`^`标记为锚。
  - 第一个元素是*不*，需要作为组的锚。
  - 当非可变区域锚定组时，如果该区域为空，则用于打印组的检测器为空。
* 文字、变量、自定义指令和类型指令是组中唯一有效的元素。
  - 可以使用任何属性变量，但只能将可选属性标记为锚点。
  - 只能使用可变或可选的结果和操作数参数。
  - 可以使用所有区域变量。当使用非可变长度区域时，如果组不存在，则区域为空。

具有可选组的操作的一个示例是具有可变数量的操作数的‘std.rereturn’。

```tablegen
def ReturnOp : ... {
  let arguments = (ins Variadic<AnyType>:$operands);

  // We only print the operands and types if there are a non-zero number
  // of operands.
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
```

##### 单位属性

在MLIR中，[`unit`属性](方言/Builtin.md/#unitattr)的特殊之处在于它只有一个可能的值，即它从它的存在派生意义。当单元属性用于锚定可选组并且不是该组的第一个元素时，该单元属性的存在可以与该可选组本身的存在直接相关。因此，在这些情况下，单位属性将不会打印或显示在输出中，并且在解析时会根据可选组本身的存在自动推断。

例如，以下操作：

```tablegen
def FooOp : ... {
  let arguments = (ins UnitAttr:$is_read_only);

  let assemblyFormat = "attr-dict (`is_read_only` $is_read_only^)?";
}
```

的格式如下：

```mlir
// When the unit attribute is present:
foo.op is_read_only

// When the unit attribute is not present:
foo.op
```

##### 可选的“Else”组

可选组还支持“Else”元素组。如果可选组的`Anchor`元素存在*不*，则解析/打印这些元素。与主元素组不同，Else组对第一个元素没有限制，所有元素都不能充当可选元素的`锚`。示例如下所示：

```tablegen
def FooOp : ... {
  let arguments = (ins UnitAttr:$foo);

  let assemblyFormat = "attr-dict (`foo_is_present` $foo^):(`foo_is_absent`)?";
}
```

的格式如下：

```mlir
// When the `foo` attribute is present:
foo.op foo_is_present

// When the `foo` attribute is not present:
foo.op foo_is_absent
```

#### 要求

格式规范有一组必须遵守的特定要求：

输出和操作名称永远不会显示，因为它们是固定的，不能更改。
运算中的所有操作数必须单独出现在格式中，或者与`operands`指令一起出现。
操作中的所有区域必须单独出现在格式中，或者与`regions`指令一起出现。
操作中的所有后继都必须出现在格式中，可以单独出现，也可以与`sustrors`指令一起出现。
所有操作数和结果类型都必须使用各种`type`指令出现在格式中，可以单独出现，也可以与`operands`或`result`指令一起出现。
‘attr-didic`指令必须始终存在。
不得包含重叠信息；例如，“attr-dict”、类型、操作数等的多个实例。
- 请注意，`attr-didic`与单个属性不重叠。打印属性字典时，这些属性将被简单地省略。

##### 类型推理

格式的一个要求是操作数类型和结果必须始终存在。在某些情况下，可以通过类型约束或其他可用的信息来推断变量的类型。在这些情况下，可以从格式中省略该变量的类型。

* 可构建类型

某些类型约束可能只有一种表示形式，允许直接构建；例如，`I32`或`Index`类型。`ODS`中的类型可以通过设置`builderCall`字段或继承`BuildableType`类将自己标记为可构建。

* 特质平等约束

有许多操作将已知的类型相等约束注册为操作上的特征；例如，“select`”操作的true、false和result值通常具有相同的类型。汇编格式可以检查这些相等约束以辨别丢失变量的类型。目前支持的特征有：`AllTypesMatch`、`TypesMatchWith`、`SameTypeOperands`和`SameOperandsAndResultType`。

### ‘hasCanonicalizer’

此布尔字段指示是否已为此操作定义规范化模式。如果是`1`，则需要定义`：：getCanonicalizationPatterns()`。

### `hasCanonicalizeMethod`

当该布尔字段设置为`true`时，表示OP对简单的matchAndRewrite风格的规范化模式实现了`canonicalize`方法。如果`hasCanonicalizer`为0，则实现`：：getCanonicalizationPatterns()`来调用该函数。

### “仇恨”

此布尔值字段指示是否已为此操作定义常规折叠规则。如果是`1`，则需要定义`：：Fold()`。

### 额外声明

表驱动OP定义的目标之一是自动生成每个OP所需的尽可能多的逻辑和方法。尽管如此，总会有一些长尾案例不会被覆盖。对于这种情况，可以使用`ExtraClassDeclaration`。`ExtraClassDeclaration`中的代码会被逐字复制到生成的C++op类中。

需要注意的是，`extraClassDeclaration`是一种针对高级用户长尾情况的机制，对于尚未实现广泛适用的情况，最好是完善基础设施。

### 生成的C++代码

[OpDefinitionsGen](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp)处理OP定义规范文件并生成两个包含相应C++代码的文件：一个用于声明，另一个用于定义。前者通过`-gen-op-decls`命令行选项生成，后者通过`-gen-op-defs`选项生成。

定义文件包含所有OP方法定义，可以通过定义`Get_OP_CLASSES`来包含和启用这些定义。对于每个操作，OpDefinitionsGen生成一个操作类和一个[操作数适配器](#Operand-Adaptors)类。此外，它还包含以逗号分隔的所有已定义操作的列表，可以通过定义`Get_OP_LIST`来包含和启用该列表。

#### 类名和命名空间

对于每个操作，其生成的C++类名是删除方言前缀的TableGen的符号‘defed’。第一个`_`用作分隔符。例如，`def TF_AddOp`，C++类名为`AddOp`。我们去掉了`TF‘前缀，因为它用于限定操作范围；其他方言也可以定义它们自己的’AddOps‘。

生成的C++类的命名空间将来自方言的`cppNamespace`字段。例如，如果某方言的`cppNamespace`为`A：：B`，则该方言的op会放在`Namespace A{Namespace B{.}}`中。如果某个方言没有指定`cppNamespace`，则使用该方言的名称作为命名空间。

这意味着生成的C++类的限定名不一定与[操作名](#operation-name)中解释的操作名完全匹配。这是为了允许灵活命名以满足编码样式要求。

#### 操作数适配器

对于每个操作，我们都会自动生成一个*操作数适配器*。这个类解决了访问作为`Value列表提供的操作数而不使用“魔术”常量的问题。操作数适配器引用`Value`数组，并提供与操作类中的方法同名的方法来访问它们。例如，对于二进制算术运算，它可以提供`.lhs()`来访问第一个操作数，并提供`.rhs()`来访问第二个操作数。

操作数适配器类与操作类位于相同的命名空间中，在OP类内部具有操作名称后跟`Adaptor`以及别名`Adaptor`。

操作数适配器可以在也处理操作的函数模板中使用：

```c++
template <typename BinaryOpTy>
std::pair<Value, Value> zip(BinaryOpTy &&op) {
  return std::make_pair(op.lhs(), op.rhs());;
}

void process(AddOp op, ArrayRef<Value> newOperands) {
  zip(op);
  zip(Adaptor<AddOp>(newOperands));
  /*...*/
}
```

## 约束条件

约束是表驱动操作定义中的一个核心概念：操作验证和图操作匹配都是建立在满足约束的基础上的。因此，操作定义和重写规则规范都涉及到编写约束。我们在[`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td)中的`Constraint‘类具有所有约束的公共基类。

一个操作的约束可以覆盖不同的范围；它可以

* 仅涉及单个属性(例如，是大于5的32位整数)，
* 多个操作数和结果(例如，第一个结果的形状必须与第一个操作数相同)，或者
* 手术本身固有的(例如，没有副作用)。

我们将它们分别称为单实体约束、多实体约束和特征。

### 单一实体约束

范围为单个操作数、属性或结果的约束在实体的声明位置指定，如[操作参数](#operation-arguments)和[操作结果](#operation-result)中所述。

为了帮助对常见类型的约束进行建模，创建了一组“TypeConstraint”；它们是“Type`”子类层次结构。包括`F32`表示浮点型约束，`TensorOf<[F32]>`表示浮点型张量约束等。

类似地，创建了一组`AttrConstraint‘s，用于帮助对常见属性类型的约束进行建模。它们是`Attr`子类层次结构。其中`F32Attr`表示浮点型属性的约束，`F32ArrayAttr`表示浮点型数组属性的约束，依此类推。

### 多实体约束

涉及多个操作数/属性/结果的约束在操作上非常常见，比如操作数和结果之间的元素类型和形状关系。这些约束应指定为`Op`类模板参数，如[操作特征和constraints](#operation-traits-and-constraints).]中所述

多实体约束在[`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td).A中建模为`PredOpTrait`(`OpTrait`的子类)，提供了一组约束原语以帮助规范。有关完整列表，请参阅[`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td)。

### 特性

特征是操作的固有属性，如是否有副作用、是否可交换、是否为终止符等，这些约束应指定为`Op`类模板参数，如[操作特征和constraints](#operation-traits-and-constraints).]中所述

在[`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td).中，特征被建模为`NativeOpTrait`(`OpTrait`的子类它们会被备份，并会被翻译成对应的C++`mlir：：OpTrait`类。

### 如何指定新约束

要编写约束，您需要提供它的谓词并给它一个描述性名称。使用`Pred`类建模的谓词是组成约束的主力。约束的谓词通常使用以下两类谓词以嵌套方式构建：

`CPred`：原始叶谓词。
复合谓词：子谓词使用谓词组合器(CONNECTION：`AND`，DISCOCT：`OR`，NEVERATION：`Neg`，SUBSION：`SubstLeaves`，CONCATENATION：`Concat`)组合而成的谓词。

`CPred`是组成更复杂谓词的基础。从TableGen的角度来看，它是“ATOM”谓词，是TableGen和C++之间的“接口”。里面已经是C++代码，它将被视为带有要替换的特殊占位符的不透明字符串。

您可以将任何返回布尔值的C++代码放入`CPred`中，包括计算表达式、调用函数、调用类方法等。

为了帮助与C++环境交互，提供了一些特殊的占位符来引用使用此谓词的上下文中的实体。它们是封闭环境的“钩子”。包括`$_builder`、`$_op`和`$_self`：

* `$_builder`将替换为一个`mlir：：Builder`实例，这样您就可以访问常用的构建方法。
* `$_op`将替换为当前操作，以便您可以访问当前操作的信息。
* `$_self`将替换为该谓词所附加的实体。例如，`BoolAttr`是包装`CPred<“$_self.isa<BoolAttr>()>`的属性约束。那么对于`f32：$attr`，`$_self`将替换为`$attr`。对于类型约束，它有点特殊，因为我们希望每个类型定义上的约束都能自然读取，并且希望将类型约束直接附加到操作数/结果，`$_self`将被操作数/结果的类型替换。例如`F32：$operand`中的`F32`，其`$_self`将扩展为`getOperand(.).getType()`。

TODO：重新考虑特殊占位符的前导符号。最终，我们希望允许引用操作数/结果$-name；这样的$-name可以以下划线开头。

例如，写一个属性`attr`是一个`IntegerAttr`，在C++中只需调用`attr.isa<IntegerAttr>()`即可。代码可以封装在`CPred`中，封装为`$_self.isa<IntegerAttr>()`，`$_self`作为特殊占位符，在扩展时替换为当前属性`attr`。

对于更复杂的谓词，可以将其封装在一个`CPred`中，也可以使用谓词组合器进行组合。例如，要编写属性`attr`是32位或64位整数的约束，可以将其写为

```tablegen
And<[
  CPred<"$_self.isa<IntegerAttr>()">,
  Or<[
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(32)">,
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(64)">
  ]>
]>
```

(请注意，以上只是用一个熟悉的示例说明如何使用`CPred`和谓词组合器编写复杂的谓词。具体对于整数属性，[`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td)已经定义了`I32Attr`和`I64Attr`。因此，您实际上可以重用它们将其写为`或<[I32Attr.dicate，I64Attr.dicate]>`。)

TODO：构建可重用原语约束库

如果使用`CPred`和谓词组合器编写谓词非常复杂，您也可以将其作为一个普通的C++函数编写，并使用`CPred`作为“调用”该函数的一种方式。例如，要验证属性`attr`是否具有某些属性，可以编写如下C++函数

```cpp
bool HasSomeProperty(Attribute attr) { ... }
```

然后将操作定义为：

```tablegen
def HasSomeProperty : AttrConstraint<CPred<"HasSomeProperty($_self)">,
                                     "has some property">;

def MyOp : Op<...> {
  let arguments = (ins
    ...
    HasSomeProperty:$attr
  );
}
```

至于是用包装整个表达式的单个`CPred`定义谓词，还是用多个带有谓词组合子的`CPred`定义谓词，还是用一个`CPred`“调用”一个函数，目前还没有明确的标准。使用`CPred`和谓词组合器进行定义更可取，因为它在OP定义规范中公开了更多信息(而不是隐藏C++函数背后的所有逻辑)，因此它可以潜在地驱动更多的自动生成用例。但它将需要一个很好的公共谓词库作为构建块，以避免重复，目前正在进行这项工作。

## 属性定义

属性是操作的编译时已知常量。

ODS在C++属性类上提供属性包装器。在MLIR的core IR库中定义了一些常见的C++[Attribute classes](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Attributes.h)]，其中一个可以自由定义特定于方言的属性类。ODS允许在TableGen中使用这些属性来定义操作，可能具有更细粒度的约束。例如，`StrAttr`直接映射到`StringAttr`；`F32Attr`/`F64Attr`要求`FloatAttr`另外具有一定的位宽。

ODS属性被定义为具有存储类型(对应于支持*商店*该属性的`mlir：：Attribute`)、返回类型(对应于生成的助手Getter的C++*返回*类型)以及在内部存储器和助手方法之间进行转换的方法。

### 属性修饰符

有几个重要的属性适配器/修饰符/修饰符可以应用于ODS属性，以指定常见的附加属性，如可选性、默认值等：

* `DefaultValuedAttr`：指定
属性的[默认值](#Attributes-with-Default-Values)。
* `OptionalAttr`：指定属性为[可选](#可选-属性)。
* `Confined`：使用
[进一步约束](#confining-properties)。

### 枚举属性

一些属性只能从预定义的枚举中取值，例如比较类型的比较OP。为了定义这些属性，ODS提供了几种机制：`StrEnumAttr`、`IntEnumAttr`和`BitEnumAttr`。

* `StrEnumAttr`：每个枚举案例都是一个字符串，属性存储为
[`StringAttr`](Dialects/Builtin.md/#stringattr)在行动中
* `IntEnumAttr`：每个枚举大小写都是整数，属性存储为
[`IntegerAttr`](Dialects/Builtin.md/#integertype)在行动中
* `BitEnumAttr`：每个枚举大小写都是一个位，属性存储为
[`IntegerAttr`](Dialects/Builtin.md/#integertype)在行动中

所有这些`*EnumAttr`属性都需要通过其对应的`*EnumAttrCase`完整指定所有允许的用例。有了这一点，ODS能够生成额外的核查，以便只接受允许的情况。为了方便`*EnumAttrs与其C++消费者的交互，[`EnumsGen`](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/EnumsGen.cpp)TableGen后台可以生成一些常用的实用工具：C++枚举类、枚举类的`llvm：：DenseMapInfo`、字符串转换函数。通过`mlir-tblgen`的`-gen-enum-decls`和`-gen-enum-defs`命令行选项控制。

例如，给定以下`EnumAttr`：

```tablegen
def Case15: I32EnumAttrCase<"Case15", 15>;
def Case20: I32EnumAttrCase<"Case20", 20>;

def MyIntEnum: I32EnumAttr<"MyIntEnum", "An example int enum",
                           [Case15, Case20]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
```

以下内容将通过`mlir-tblgen-gen-enum-decls`生成：

```c++
namespace Outer {
namespace Inner {
// An example int enum
enum class MyIntEnum : uint32_t {
  Case15 = 15,
  Case20 = 20,
};

llvm::Optional<MyIntEnum> symbolizeMyIntEnum(uint32_t);
llvm::StringRef ConvertToString(MyIntEnum);
llvm::Optional<MyIntEnum> ConvertToEnum(llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMyIntEnum() {
  return 20;
}

} // namespace Inner
} // namespace Outer

namespace llvm {
template<> struct DenseMapInfo<Outer::Inner::MyIntEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline Outer::Inner::MyIntEnum getEmptyKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getEmptyKey());
  }

  static inline Outer::Inner::MyIntEnum getTombstoneKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Outer::Inner::MyIntEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const Outer::Inner::MyIntEnum &lhs, const Outer::Inner::MyIntEnum &rhs) {
    return lhs == rhs;
  }
};
}
```

以下内容将通过`mlir-tblgen-gen-enum-defs`生成：

```c++
namespace Outer {
namespace Inner {
llvm::StringRef ConvertToString(MyIntEnum val) {
  switch (val) {
    case MyIntEnum::Case15: return "Case15";
    case MyIntEnum::Case20: return "Case20";
  }
  return "";
}

llvm::Optional<MyIntEnum> ConvertToEnum(llvm::StringRef str) {
  return llvm::StringSwitch<llvm::Optional<MyIntEnum>>(str)
      .Case("Case15", MyIntEnum::Case15)
      .Case("Case20", MyIntEnum::Case20)
      .Default(llvm::None);
}
llvm::Optional<MyIntEnum> symbolizeMyIntEnum(uint32_t value) {
  switch (value) {
  case 15: return MyIntEnum::Case15;
  case 20: return MyIntEnum::Case20;
  default: return llvm::None;
  }
}

} // namespace Inner
} // namespace Outer
```

类似于下面的`BitEnumAttr`定义：

```tablegen
def None: BitEnumAttrCase<"None", 0x0000>;
def Bit1: BitEnumAttrCase<"Bit1", 0x0001>;
def Bit2: BitEnumAttrCase<"Bit2", 0x0002>;
def Bit3: BitEnumAttrCase<"Bit3", 0x0004>;

def MyBitEnum: BitEnumAttr<"MyBitEnum", "An example bit enum",
                           [None, Bit1, Bit2, Bit3]>;
```

我们可以有：

```c++
// An example bit enum
enum class MyBitEnum : uint32_t {
  None = 0,
  Bit1 = 1,
  Bit2 = 2,
  Bit3 = 4,
};

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t);
std::string stringifyMyBitEnum(MyBitEnum);
llvm::Optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef);
inline MyBitEnum operator|(MyBitEnum lhs, MyBitEnum rhs) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline MyBitEnum operator&(MyBitEnum lhs, MyBitEnum rhs) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(MyBitEnum bits, MyBitEnum bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

namespace llvm {
template<> struct DenseMapInfo<::MyBitEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline ::MyBitEnum getEmptyKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getEmptyKey());
  }

  static inline ::MyBitEnum getTombstoneKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::MyBitEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::MyBitEnum &lhs, const ::MyBitEnum &rhs) {
    return lhs == rhs;
  }
};
```

```c++
std::string stringifyMyBitEnum(MyBitEnum symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  llvm::SmallVector<llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("Bit1"); val &= ~1u; }
  if (2u & val) { strs.push_back("Bit2"); val &= ~2u; }
  if (4u & val) { strs.push_back("Bit3"); val &= ~4u; }

  if (val) return "";
  return llvm::join(strs, "|");
}

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return MyBitEnum::None;

  llvm::SmallVector<llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<llvm::Optional<uint32_t>>(symbol)
      .Case("Bit1", 1)
      .Case("Bit2", 2)
      .Case("Bit3", 4)
      .Default(llvm::None);
    if (bit) { val |= *bit; } else { return llvm::None; }
  }
  return static_cast<MyBitEnum>(val);
}

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return MyBitEnum::None;

  if (value & ~(1u | 2u | 4u)) return llvm::None;
  return static_cast<MyBitEnum>(value);
}
```

## 类型定义

MLIR定义了TypeDef类层次结构，以支持从其规范生成数据类型。通过使用TypeDef类所需的所有字段的具体内容专门化TypeDef类来定义类型。例如，整数类型可以定义为：

```tablegen
// All of the types will extend this class.
class Test_Type<string name> : TypeDef<Test_Dialect, name> { }

// An alternate int type.
def IntegerType : Test_Type<"TestInteger"> {
  let mnemonic = "int";

  let summary = "An integer type with special semantics";

  let description = [{
    An alternate integer type. This type differentiates itself from the
    standard integer type by not having a SignednessSemantics parameter, just
    a width.
  }];

  let parameters = (ins "unsigned":$width);

  // We define the printer inline.
  let printer = [{
    $_printer << "int<" << getImpl()->width << ">";
  }];

  // The parser is defined here also.
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, width);
  }];
}
```

### 类型名称

生成的C++类名称默认为`<classParamName>Type`(如上例中的`TestIntegerType`)。可通过`cppClassName`字段覆盖。`mnemonic`字段用于指定要解析的ASM名称。它是可选的，不指定它将意味着没有解析器或打印机方法附加到这个类。

### 类型文档

`Summary‘和`description ption`字段存在，其使用方式与操作中相同。也就是说，摘要应该是一行，而`description‘应该是更长的解释。

### 类型参数

`parameters`字段是类型参数的列表。如果未指定参数(默认)，则此类型被视为单一类型。参数格式为`“c++Type”：$paramName`。要将C++类型用作存储构造函数中需要分配的参数，有两种选择：

- 设置`hasCustomStorageConstructor`生成带有刚刚声明的构造函数的TypeStorage类--没有定义--这样您就可以自己编写了。
- 使用`TypeParameter`tablegen类代替“c++Type”字符串。

### TypeParameter Tablegen类

这用于进一步指定有关每个类型参数的属性。它包括文档(`摘要`和`语法`)、要使用的C++类型、要在存储构造函数方法中使用的自定义分配器，以及用于确定参数类型的两个实例是否相等的自定义比较器。

```tablegen
// DO NOT DO THIS!
let parameters = (ins "ArrayRef<int>":$dims);
```

默认存储构造函数按值盲目复制字段。它对类型一无所知。在这种情况下，ArrayRef<int>需要使用`dims=allocator.copy Into(Dims)`进行分配。

您可以通过专门化`TypeParameter`tblgen类来指定所需的构造函数：

```tablegen
class ArrayRefIntParam :
    TypeParameter<"::llvm::ArrayRef<int>", "Array of ints"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

...

let parameters = (ins ArrayRefIntParam:$dims);
```

`分配器‘代码挡路有如下替换：

- `$_allocator`是要分配对象的TypeStorageAllocator。
- `$_dst`是要放置分配数据的变量。

‘比较器’代码挡路有以下替换：

- `$_lhs`是该参数类型的实例。
- `$_rhs`是该参数类型的实例。

MLIR包括几个针对常见情况的专门类：

-`StringRefParameter<description ptionOfParam>`表示字符串引用。
-值类型的数组引用参数`数组引用参数<数组Of，Description的ptionOfParam>`
- 对于包含`allocateInto(StorageAllocator&allocator)`方法的C++类，使用`selfAllocationParameter<description ptionOfParam>`将自身分配到`allocator`中。
- `ArrayRefOfSelfAllocationParameter<arrayOf，description ptionOfParam>`表示根据上次专门化自分配的Object数组。

如果我们使用以下包含的专业化认证之一：

```tablegen
let parameters = (ins
  ArrayRefParameter<"int", "The dimensions">:$dims
);
```

### 解析和打印

如果指定助记符，则`printer`和`parser`代码字段处于活动状态。两者的规则都是：

- 如果为NULL，则仅生成声明。
- 如果非空和非空，请使用定义中的代码。`$_printer`或`$_parser`替换是有效的，应该使用。
- 挡路代码为空是错误的。

对于每种方言，将创建两个“调度”函数：一个用于解析，另一个用于打印。您应该在您的`Dialect：：printType`和`Dialect：：parseType`方法中添加对这些函数的调用。它们是放置在类型类定义旁边的静电函数，具有以下函数签名：

```c++
static Type generatedTypeParser(MLIRContext* ctxt, DialectAsmParser& parser, StringRef mnemonic);
LogicalResult generatedTypePrinter(Type type, DialectAsmPrinter& printer);
```

助记符、解析器和打印机字段是可选的。如果未定义它们，则生成的代码将不包括任何解析或打印代码，并从上面的分派函数中省略该类型。在这种情况下，方言作者负责解析/打印`Dialect：：printType`和`Dialect：：parseType`中的类型。

### 其他字段

- 如果`genStorageClass`字段设置为1(默认值)，则会生成一个存储类，其中包含与每个指定的`parameters`对应的成员变量。
- 如果`genAccessors`字段为1(默认值)，则会在Type类上生成访问器方法(例如上例中的`int getWidth()const`)。
- 如果设置了`genVerifyDecl`字段，则会在类中添加`static LogicalResult Verify(emitErrorFn，Parameters.)`方法的声明和`getChecked(emitErrorFn，Parameters.)`方法，该方法在调用`get`之前检查`verify`的结果。
- `storageClass`字段用于设置存储类的名称。
- `storageNamespace`字段用于设置存储类所在的命名空间。默认为“Detail”。
- `ExtraClassDeclaration`字段用于在类声明中包含额外的代码。

### 类型生成器方法

每种类型都有几个构建器(`get`/`getChecked`)是根据该类型的参数自动生成的。例如，给定以下类型定义：

```tablegen
def MyType : ... {
  let parameters = (ins "int":$intParam);
}
```

将生成以下构建器：

```c++
// Type builders are named `get`, and return a new instance of a type for a
// given set of parameters.
static MyType get(MLIRContext *context, int intParam);

// If `genVerifyDecl` is set to 1, the following method is also generated.
static MyType getChecked(function_ref<InFlightDiagnostic()> emitError,
                         MLIRContext *context, int intParam);
```

如果不需要这些自动生成的方法，例如当它们与自定义构建器方法冲突时，类型可以将`skipDefaultBuilders`设置为1，以发出不应该生成它们的信号。

#### 自定义类型生成器方法

默认的构建方法可能涵盖了大多数与类型构造相关的简单情况，但是当它们不能满足类型的需要时，您可以在`builders`字段中定义额外的便捷GET方法，如下所示：

```tablegen
def MyType : ... {
  let parameters = (ins "int":$intParam);

  let builders = [
    TypeBuilder<(ins "int":$intParam)>,
    TypeBuilder<(ins CArg<"int", "0">:$intParam)>,
    TypeBuilder<(ins CArg<"int", "0">:$intParam), [{
      // Write the body of the `get` builder inline here.
      return Base::get($_ctxt, intParam);
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$typeParam), [{
      // This builder states that it can infer an MLIRContext instance from
      // its arguments.
      return Base::get(typeParam.getContext(), ...);
    }]>,
  ];
}
```

`builders`字段是添加到类型类中的自定义构建器的列表。在此示例中，我们提供了几个在不同场景中有用的不同便利构建器。ODS中的许多函数声明都使用`ins`前缀，这些函数声明使用TableGen[`dag`](#tablegen-语法)。下面是一个逗号分隔的类型列表(带引号的字符串或Carg)和前缀为`$`符号的名称。使用`CArg`可以为该参数提供默认值。让我们逐个来看一下这些建造者

第一个构建器将生成一个构建器方法的声明，如下所示：

```tablegen
  let builders = [
    TypeBuilder<(ins "int":$intParam)>,
  ];
```

```c++
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam);
};
```

该构建器与`MyType`将自动生成的构建器相同。`context`参数由生成器隐式添加，在构建文件类型实例时使用(与`base：：get`配合使用)。这里的区别在于我们可以提供这个`get`方法的实现。这种构建器定义方式只生成声明，MyType的实现者需要提供`MyType：：get`的定义。

第二个构建器将生成一个构建器方法的声明，如下所示：

```tablegen
  let builders = [
    TypeBuilder<(ins CArg<"int", "0">:$intParam)>,
  ];
```

```c++
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam = 0);
};
```

这里的约束与第一个构建器示例相同，只是`intParam`现在附加了一个默认值。

第三个构建器将生成一个构建器方法的声明，如下所示：

```tablegen
  let builders = [
    TypeBuilder<(ins CArg<"int", "0">:$intParam), [{
      // Write the body of the `get` builder inline here.
      return Base::get($_ctxt, intParam);
    }]>,
  ];
```

```c++
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam = 0);
};

MyType MyType::get(::mlir::MLIRContext *context, int intParam) {
  // Write the body of the `get` builder inline here.
  return Base::get(context, intParam);
}
```

这与第二个构建器示例相同。不同之处在于，现在，将使用提供的代码挡路作为主体自动生成构建器方法的定义。指定Body内联时，可以使用`$_ctxt`访问`MLIRContext*`参数。

第四个构建器将生成一个构建器方法的声明，如下所示：

```tablegen
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$typeParam), [{
      // This builder states that it can infer an MLIRContext instance from
      // its arguments.
      return Base::get(typeParam.getContext(), ...);
    }]>,
  ];
```

```c++
class MyType : /*...*/ {
  /*...*/
  static MyType get(Type typeParam);
};

MyType MyType::get(Type typeParam) {
  // This builder states that it can infer an MLIRContext instance from its
  // arguments.
  return Base::get(typeParam.getContext(), ...);
}
```

在此构建器示例中，与第三个构建器示例3的主要区别在于不再添加`MLIRContext`参数。这是因为使用的构建器类型`TypeBuilderWithInferredContext`暗示上下文参数不是必需的，因为它可以从构建器的自变量中推断出来。

## 调试提示

### 运行`mlir-tblgen`查看生成的内容

TableGen语法有时可能比较模糊；阅读生成的内容对于理解和调试问题非常有帮助。要构建`mlir-tblgen`，可以运行`cmake--build。--瞄准您的build目录下的mlir-tblgen`，在`bin/`子目录下找到`mlir-tblgen`二进制文件。所有支持的生成器都可以通过`mlir-tblgen--help`找到。例如，[生成C++代码](#Generated-c-code)中解释的`--gen-op-decls`和`--gen-op-defs`。

要查看生成的代码，请通过`-I`提供include路径，使用特定生成器调用`mlir-tblgen`。例如,

```sh
# To see op C++ class declaration
mlir-tblgen --gen-op-decls -I /path/to/mlir/include /path/to/input/td/file
# To see op C++ class definition
mlir-tblgen --gen-op-defs -I /path/to/mlir/include /path/to/input/td/file
# To see op documentation
mlir-tblgen --gen-dialect-doc -I /path/to/mlir/include /path/to/input/td/file

# To see op interface C++ class declaration
mlir-tblgen --gen-op-interface-decls -I /path/to/mlir/include /path/to/input/td/file
# To see op interface C++ class definition
mlir-tblgen --gen-op-interface-defs -I /path/to/mlir/include /path/to/input/td/file
# To see op interface documentation
mlir-tblgen --gen-op-interface-doc -I /path/to/mlir/include /path/to/input/td/file
```

## 附录

### 需求和现有机制分析

OP描述应该尽可能具有声明性，以允许广泛的工具使用它们以及从它们生成的查询方法。具体地说，这意味着以易于分析的方式指定特征、约束和形状推断信息(例如，在可能的情况下避免对C++函数的不透明调用)。

我们考虑了几个当代系统的方法，并将重点放在需要的需求上：

* 使用独立于C++代码的注册表注册的操作。
  * MLIR中允许未知的操作，因此无需注册操作。编译器优化那些操作或包含这些操作的图的能力是有限的，但是正确的。
  * 当前的建议不包括运行时操作描述，但它不排除这样的描述，可以稍后添加。
  * OP注册表对于生成C++类至关重要，这些类通过提供类型化表示和访问器，使得操作操作、验证C++中的正确构造等变得更容易。
* OP注册表将在[TableGen](https://llvm.org/docs/TableGen/index.html)中定义，并用于生成C++类和实用函数(构建器/验证器/解析器/打印机)。
  * TableGen是LLVM后端使用的建模规范语言，非常适合基于特征的建模。这是一个实施决策，有其他方法可以做到这一点。但是，规范语言很好地满足了对特征建模的要求(从LLVM处理器后端建模的使用情况来看)，并且易于扩展，因此是一个实用的选择。如果出现另一个好的选择，我们会考虑的。
* MLIR既允许定义的操作，也允许未定义的操作。
  * 定义的操作应该具有固定的语义，并且可以定义相应的引用实现。
  * 方言完全由方言所有者控制，通常生活在方言的框架中。
* OP的特征(例如，交换性)与OP一起在注册表中建模。
* OP的操作数/返回类型约束在注册表中与OP一起建模(参见下面的[Shape Inference](ShapeInference.md)讨论)，这允许(例如)文本转储中的优化简明语法。
* OP的行为与OP一起记录，并附有摘要和描述。描述以标记形式编写并提取，以便包含在生成的方言的LangRef部分中。
* 打印和解析的通用汇编形式照常可用，但是可以指定自定义解析器和打印机，也可以从显示“Assembly”字符串到操作数/类型的映射的可选字符串表示自动生成自定义解析器和打印机。
  * 作为解析器生成的一部分，将支持解析器级别的重新映射(例如，将`eq`映射到枚举)。
* 匹配模式与OP描述分开指定。
  * 与LLVM相比，没有每个后端都需要知道的“基本”操作集。相反，有许多不同的方言，这些方言之间的转换/合法化形成了一个转换图。
* 参考实现可以与OP定义一起提供。
  
  * 参考实现可以是标准OPS或其他参考实现。
  
  TODO：如果依赖OP的定义更改，则记录期望。

