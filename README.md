# llvm-tutorial
llvm-tutorial部分翻译，目前初步完成了入门教程Kaleidoscope部分和mlir入门教程部分的翻译。*目前限于本人对llvm理解，翻译还有很多待完善地方，后面会抽时间完善，也欢迎感兴趣的同学来共同完善，欢迎任何形式的pr。*

## MLIR部分
MLIR是多级IR表示，目的是通过多级IR表示提高编译框架的可扩展性和可重用性，详情参见[MLIR文档](https://mlir.llvm.org/getting_started/)。

`mlir/standalone`中从官方仓库中迁移来了一个可独立编译的mlir工程，该工程需要先编译安装`llvm&mlir`，安装过程参考[MLIR文档](https://mlir.llvm.org/getting_started/)。项目独立编译参考[mlir/standalone/README.md](./mlir/standalone/README.md)。

`mlir/code`中是mlir的Toy教程对应的源码，为了方便学习，按照`standalone`的cmakelists改写了一下，可以独立编译验证。

**说明：**
目前文档是基于llvm-12.0.1分支构建的，工程结构如下：
```
├── LICENSE
├── BuildingAJIT
│   ├── markdown                       // 原版rst文档转换的markdown文档
│   ├── rst                            // 原版rst文档
│   └── zh-md                          // 机器翻译中文文档，!!!暂未校对
├── mlir
│   ├── code                           // llvm-project/mlir/examples/toy/路径下代码
│   ├── standalone                     // 可以独立编译的mlir项目示例
│   ├── docs
│   │   ├── ......
│   │   ├── Toy                       // mlir教程原版文档
│   │   └── zh-docs                   // 中文文档
│   └── testcode                       // llvm-project/mlir/test/Examples/Toy/路径下代码
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

## 致谢
感谢@[darionyaphet](https://github.com/darionyaphet)对项目的贡献。