# 第7章：向Toy添加复合类型

在[上一章](zh-Ch-6.md)中，我们演示了从Toy前端到LLVM IR的端到端编译流程。在本章中，我们将扩展Toy语言以支持新的复合`struct`类型。

## 在Toy中定义“struct`”

我们需要定义的第一件事是用我们的“Toy”源语言定义这种类型的接口。Toy中`struct`类型的通用语法如下：

```toy
# A struct is defined by using the `struct` keyword followed by a name.
struct MyStruct {
  # Inside of the struct is a list of variable declarations without initializers
  # or shapes, which may also be other previously defined structs.
  var a;
  var b;
}
```

现在，通过使用结构的名称而不是`var`，可以在函数中将结构用作变量或参数。该结构的成员通过`.`访问运算符进行访问。`struct`类型的值可以用复合初始值设定项初始化，也可以用`{}`括起来的其他初始值设定项的逗号分隔列表进行初始化。示例如下所示：

```toy
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

## 在MLIR中定义`struct`

在MLIR中，我们还需要结构类型的表示形式。MLIR不能提供完全符合我们需要的类型，因此我们需要定义自己的类型。我们将简单地将我们的`struct`定义为一组元素类型的未命名容器。`struct`的名称及其元素只对我们的`toy`编译器的AST有用，所以我们不需要在MLIR表示中对其进行编码。

### 定义类型类

#### 定义类型类

如[第2章](zh-Ch-2.md)中所述，MLIR中的[`Type`](../../LangRef.md#type-system)对象是值类型的，并且依赖于拥有保存该类型的实际数据的内部存储对象。`Type`类本身充当内部`TypeStorage`对象的简单包装，该对象在`MLIRContext`的实例中是唯一的。在构造`Type`时，我们在内部只是构造并唯一化一个存储类的实例。

在定义包含参数数据的新`Type`时(例如`struct`类型，需要额外的信息来保存元素类型)，我们需要提供派生的存储类。没有额外数据的`singleton`类型(如[`index`type](../../LangRef.md#index-type))不需要存储类，使用默认的`TypeStorage`。

##### 定义存储类

类型存储对象包含构造和唯一类型实例所需的所有数据。派生存储类必须继承自基本`mlir：：TypeStorage`，并提供一组别名和钩子，供`MLIRContext`用于唯一类型。下面是我们的`struct`类型的存储实例的定义，每个必需的要求都内联了详细说明：

```c++
/// This class represents the internal storage of the Toy `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself, see the `StructType::get` method further below.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

##### 定义类型类

定义存储类后，我们可以为用户可见的`StructType`类添加定义。这是我们将实际与之交互的类。

```c++
/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first parameter is the context to unique in. The
    // parameters after are forwarded to the storage instance.
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes() {
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
  }

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
```

我们在`Toy Dialect`构造函数中注册此类型的方式与我们处理操作的方式类似：

```c++
ToyDialect::ToyDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addTypes<StructType>();
}
```

有了这个，我们现在可以在从Toy生成MLIR时使用我们的`StructType`。有关更多详细信息，请参见`Examples/toy/ch7/mlir/MLIRGen.cpp`。

### 解析和打印

此时，我们可以在MLIR生成和转换过程中使用我们的`StructType`，但不能输出或解析`.mlir`。为此，我们需要增加对`StructType`实例的解析和打印支持。这可以通过覆盖`Toy Dialect`上的`parseType`和`printType`方法来实现。

```c++
class ToyDialect : public mlir::Dialect {
public:
  /// Parse an instance of a type registered to the toy dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to the toy dialect.
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};
```

这些方法采用允许轻松实现必要功能的高级解析器或打印类的实例。在开始实现之前，让我们先考虑一下打印的IR中的`struct`类型所需的语法。如[MLIR语言参考](../../LangRef.md#dialect-types)中所述，方言类型通常表示为：`！dialect-namespace<type-data>`，在某些情况下可以使用漂亮的形式。我们的`toy`解析器和打印类的职责是提供`type-data`位。我们将我们的`StructType`定义为具有以下形式：

```
  struct-type ::= `struct` `<` type (`,` type)* `>`
```

#### 解析

解析器的实现如下所示：

```c++
/// Parse an instance of a type registered to the toy dialect.
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<mlir::TensorType, StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}
```

#### 打印

打印类的实现如下所示：

```c++
/// Print an instance of a type registered to the toy dialect.
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
```

在继续之前，让我们先来看一下展示我们现在拥有的功能的快速示例：

```toy
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
}
```

它会生成以下内容：

```mlir
module {
  func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) {
    toy.return
  }
}
```

### 在`StructType`上操作

现在已经定义了`struct`类型，我们可以往返于IR之间。下一步是添加对在我们的操作中使用它的支持。

#### 更新现有操作

我们现有的一些操作需要更新以处理`StructType`。第一步是让ODS框架知道我们的Type，这样我们就可以在操作定义中使用它。下面是一个简单的示例：

```tablegen
// Provide a definition for the Toy StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef.
def Toy_StructType :
    Type<CPred<"$_self.isa<StructType>()">, "Toy struct type">;

// Provide a definition of the types that are used within the Toy dialect.
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

然后我们可以更新我们的操作，例如`ReturnOp`，以也接受`Toy_StructType`：

```tablegen
def ReturnOp : Toy_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  ...
  let arguments = (ins Variadic<Toy_Type>:$input);
  ...
}
```

#### 添加新的`TOY`操作

除了现有的操作之外，我们还将添加一些新的操作，这些操作将提供对`structs`的更具体的处理。

##### `toy.struct_constant`

这个新操作实现了结构的常量值。在我们当前的建模中，我们只使用了一个[数组属性](../../LangRef.md#array-attribute)，它为每个`struct`元素包含一组常量值。

```mlir
  %0 = toy.struct_constant [
    dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  ] : !toy.struct<tensor<*xf64>>
```

##### `toy.struct_access`

这个新操作实现了`struct`值的第N个元素。

```mlir
  // Using %0 from above
  %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
```

通过这些操作，我们可以重新查看最初的示例：

```toy
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

并最终获得完整的MLIR模块：

```mlir
module {
  func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
    %0 = toy.struct_access %arg0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.struct_access %arg0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %3 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %4 = toy.mul %1, %3 : tensor<*xf64>
    toy.return %4 : tensor<*xf64>
  }
  func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.generic_call @multiply_transpose(%0) : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    toy.print %1 : tensor<*xf64>
    toy.return
  }
}
```

#### 优化`StructType`的操作

现在我们有几个操作在“StructType”上，我们也有许多新的常量折叠机会。

内联后，上一节中的MLIR模块如下所示：

```mlir
module {
  func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.struct_access %0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %4 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.mul %2, %4 : tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
```

我们有几个访问`toy.struct_constant`的`toy.struct_access`操作。如[第3章](zh-Ch-3.md)(FoldConstantReshape)所述，我们可以通过在操作定义上设置`hasFolder`位并提供`*Op：：fold`方法的定义来为这些`toy`操作添加folder操作。

```c++
/// Fold constants.
OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) { return value(); }

/// Fold struct constants.
OpFoldResult StructConstantOp::fold(ArrayRef<Attribute> operands) {
  return value();
}

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(ArrayRef<Attribute> operands) {
  auto structAttr = operands.front().dyn_cast_or_null<mlir::ArrayAttr>();
  if (!structAttr)
    return nullptr;

  size_t elementIndex = index().getZExtValue();
  return structAttr[elementIndex];
}
```

为了确保MLIR在折叠我们的`Toy`操作时生成正确的常量操作，即`TensorType`的`ConstantOp`和`StructType`的`StructConstant`，我们需要提供方言钩子`MaterializeConstant`的覆盖。这允许通用MLIR操作在必要时为`TOY`方言创建常量。

```c++
mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (type.isa<StructType>())
    return builder.create<StructConstantOp>(loc, type,
                                            value.cast<mlir::ArrayAttr>());
  return builder.create<ConstantOp>(loc, type,
                                    value.cast<mlir::DenseElementsAttr>());
}
```

有了这一点，我们现在可以生成可以生成到LLVM的代码，而不需要对我们的流程进行任何更改。

```mlir
module {
  func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.mul %1, %1 : tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```

您可以构建`toyc-ch7`并亲自试用：`toyc-ch7 test/examples/Toy/ch7/struct-codegen.toy -emit=mlir`。有关定义自定义类型的更多详细信息，请参阅[DefiningAttributesAndTypes](../DefiningAttributesAndTypes.md).
