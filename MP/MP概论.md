## 理论：

网上有众多优秀的资料，本人不再赘述，可参阅下面的链接

教程1：[MP模型](https://zhuanlan.zhihu.com/p/142904870)

教程2：[MP模型的代码详解](https://zhuanlan.zhihu.com/p/140060038)

## 项目代码：

现在的代码是用MP实现 **加减法** , 修改了原本的激活函数实现的

原代码是实现 **与运算** 的MP感知器，如果需要复现与运算，只需要需要修改源代码的 **get_training_dataset** 即可

```python
def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [0, 0, 1, 1]  # 修改标签，使其符合 AND 的真值表
    return input_vecs, labels
```

## 实验结论：

1. 关于 **异或运算** 感知器的实现？
   
   MP不能实现异或，得BP网络才行

2. 替换激活函数，能实现原本MP不能实现的 **"+/-法"**