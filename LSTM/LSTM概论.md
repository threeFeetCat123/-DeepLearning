## 理论：

一样的，网上有众多优秀的资料，本人不再赘述，可参阅下面的链接（不是因为懒）

教程1：[LSTM基本概念介绍](https://blog.csdn.net/beiye_/article/details/123621086?spm=1001.2014.3001.5501)

教程2：[数学推导+相关代码工程](https://blog.csdn.net/qq_73462282/article/details/132073333)（注：链接的源代码有些许bug，参考需谨慎，建议看该项目的LSTM.py）

## 项目代码：

LSTM.py 代码工程在[源代码](https://blog.csdn.net/qq_73462282/article/details/132073333)的基础上修bug，实现预测 sin 函数

LSTM_beta.py 是在 LSTM.py 的基础上实现的 **结构多层** 的 **lstm**

## 实验结论：
1. 根据实验发现，单层的LSTM 拟合能力比较弱（3*3的隐藏层去拟合 **正弦函数** + **线性插值** 的复合，loss平均在0.33）。

2. 时序数据拟合比较好

3. LSTM，主要是解决如何利用所有的信息，来生成一个最佳答案。那么有没有其他更好的方法呢？
 
   那么请见伟大的 **《Attention Is All You Need》** 的 **Transformer** 结构

4. 未来计划拓展该 LSTM 模块，有以下目标
    
    2.1.(solved)  可自定义 LSTM 隐藏层深度（目前只支持一层）     

    2.2. 自定义隐藏层的 **state_width** 与 输出层 **output_width** 大小 (目前state_width == output_width)
