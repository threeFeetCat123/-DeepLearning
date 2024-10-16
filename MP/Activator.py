def ReLUActivator(x):
    # 下面的是经典MP的激活函数
    return 1 if x > 0 else 0

#   这个不知道是啥玩意，自己捣鼓的，能实现MP不能整的加减法，但都是线性的，异或不行（异或要BP）
def IdentityActivator(x):
    return x