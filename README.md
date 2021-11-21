# llvm-tutorial
llvm-tutorial部分翻译，目前初步完成了入门教程Kaleidoscope部分和mlir入门教程部分的翻译。*目前限于本人对llvm理解，翻译还有很多待完善地方，后面会抽时间完善，也欢迎感兴趣的同学来共同完善，欢迎任何形式的pr。*

**说明：**
目前文档是基于llvm-12.0.1分支构建的，工程结构如下：
```
├── LICENSE
├── mlir
│   ├── code                    // llvm-project/mlir/examples/toy/路径下代码
│   ├── docs
│   │   ├── ......
│   │   ├── Toy                // mlir教程原版文档
│   │   └── zh-docs            // 中文文档
│   └── testcode                // llvm-project/mlir/test/Examples/Toy/路径下代码
├── MyFirstLanguageFrontend             // Kaleidoscope教程
│   ├── answer                         // 官方仓库的示例代码
│   ├── code                           // 学习过程对照实现
│   └── doc
│       ├── markdown                   // 原版英文rst文档转换成markdown
│       ├── pics
│       ├── rst                        // 原版rst文档
│       └── zh-md                      // 翻译后中文markdown文档
└── README.md

```
