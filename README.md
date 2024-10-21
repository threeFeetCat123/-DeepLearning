# -DeepLearning
### 介绍：

**我的深度学习之旅**

从经典的深度学习算法出发，一步一步深入。

从原理入门，到小demo实践，是你打开深度学习大门的不二人选。

从自己理解出发，加上整合网上优秀资料，打造贴心的学习路线。

### 内容：

目前的框架的学习路线：MP -> BP -> RNN -> LSTM -> Transformer
代码都是手搓实现，仅使用numpy等基础工具件，不涉及pytorch/cuda/tensflow等，对新手友好。

新增加Optimizer优化器内容。

### 叮嘱：

**不同的深度学习算法，差别只在于不同的参数框架。本质都是利用反向传播进行梯度下降。**

**不同的深度学习算法，差别只在于不同的参数框架。本质都是利用反向传播进行梯度下降。**

**不同的深度学习算法，差别只在于不同的参数框架。本质都是利用反向传播进行梯度下降。**

要理解深度学习的精髓。

### 参考：(⭐为推荐阅读指数)

入门知识：

- **⭐⭐⭐[深度学习概论](https://blog.csdn.net/illikang/article/details/82019945)**
- **⭐[分类模型评估指标](https://easyai.tech/ai-definition/accuracy-precision-recall-f1-roc-auc/)**
- **⭐⭐[模型评估指标](https://blog.csdn.net/SeizeeveryDay/article/details/117757664)**

基础知识：
- **⭐⭐⭐[反向传播的数学推导](https://www.cnblogs.com/jsfantasy/p/12177275.html)**
- **⭐⭐⭐[反向传播的代码实现](https://www.cnblogs.com/jsfantasy/p/12177216.html)**
- **⭐⭐[Optimizer优化器](Optimizer_优化器\优化器介绍.md)**
  
MP：
- **⭐⭐[一切的起源——MP](https://zhuanlan.zhihu.com/p/142904870)**
- **⭐⭐[MP代码实现](https://zhuanlan.zhihu.com/p/140060038)**

RNN:
- **⭐⭐[介绍与定义](https://blog.csdn.net/beiye_/article/details/123526075)**
- **⭐⭐[数学推导+代码](https://blog.csdn.net/qq_43601378/article/details/124540267)**

LSTM:
- **⭐⭐[LSTM原理 浅显版本](https://blog.csdn.net/beiye_/article/details/123621086?spm=1001.2014.3001.5501)**
- **⭐⭐[LSTM原理+数学+代码](https://blog.csdn.net/qq_73462282/article/details/132073333)**（PS：链接代码有些许bug）

Transformer 框架学习：
- **⭐⭐[理论学习](https://zhuanlan.zhihu.com/p/82312421)**
- **⭐[Llama源码解析](https://zhuanlan.zhihu.com/p/648365207)**

大模型：
- **⭐⭐[了解分布式训练模型](https://huggingface.co/blog/zh/bloom-megatron-deepspeed)**

### 其他

.ipynb 文件形式的 demo 可以在 **colab** 上实践使用。



$\surd$

Feature：

| Description                                             | State                      |
|---------------------------------------------------------|----------------------------|
| 优化代码结构，解耦                                        |     <ul> &#10003; 已完成 </ul> |
| LSTM: 可自定义隐藏层深度                                  | <ul> &#10003; 已完成 </ul>    |
| LSTM: 解耦隐藏层的 **state_width** 与 输出层 **output_width** | <ul> &#10007; 未完成(存在技术上的困难)</ul> |
| 添加 **Transformer** or **GPT** 模型的手搓代码内容         | <ul> &#10007; 未完成 </ul>   |
| Optimizer优化器介绍与代码                                          | <ul> &#10003; 已完成 </ul> |
