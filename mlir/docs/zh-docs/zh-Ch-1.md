# 第1章：玩具语言和AST

[TOC]

## 语言

本教程将用一种玩具语言来说明，我们称之为“玩具”(命名很难……)。Toy是一种基于张量的语言，允许您定义函数、执行一些数学计算和打印结果。

考虑到我们希望保持简单，编码生成将被限制为秩<=2的张量，并且Toy中唯一的数据类型是64位浮点类型(在C中也称为“DOUBLE”)。因此，所有值都是隐式双精度的，‘Values`是不可变的(即，每个操作都返回一个新分配的值)，并且释放是自动管理的。但长篇大论已经足够了；没有什么比通过一个例子来更好地理解更好的了：

```toy
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose() and print() are the only builtin, the following will transpose
  # a and b and perform an element-wise multiplication before printing the result.
  print(transpose(a) * transpose(b));
}
```

类型检查是通过类型推断静态执行的；该语言仅在需要时需要类型声明来指定张量形状。函数是通用的：它们的参数是未分级的(换句话说，我们知道这些是张量，但我们不知道它们的维数)。它们专门用于调用点的每个新发现的签名。让我们通过添加一个用户定义函数来回顾上一个示例：

```toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shape will
  # trigger a shape inference error.
  var f = multiply_transpose(transpose(a), c);
}
```

## AST

上面代码中的AST相当简单；下面是它的一个转储：

```
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1'
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1'
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: b @test/Examples/Toy/Ch1/ast.toy:25:30
          var: c @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:30
            var: a @test/Examples/Toy/Ch1/ast.toy:28:40
          ]
          var: c @test/Examples/Toy/Ch1/ast.toy:28:44
        ]
    } // Block
```

您可以重现此结果，并在`Examples/Toy/Ch1/`目录中使用示例；尝试运行`path/to/build/bin/toyc-ch1test/Examples/Toy/Ch1/ast.toy-emit=ast`。

lexer的代码相当简单；所有代码都在一个头文件中：`Examples/Toy/Ch1/Include/Toy/Lexfor.h`。解析器可以在`Examples/Toy/ch1/include/toy/Parser.h`中找到，它是一个递归下降解析器。如果您不熟悉这样的词法分析器/解析器，它们与[Kaleidcope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl02.html).]的前两章中详细介绍的LLVM万花筒等效物非常相似

[下一章](CH-2.md)将演示如何将此AST转换为MLIR。
