# 导入 PyTorch 库，它是一个用于深度学习的开源机器学习库
import torch
# 从 torch 库中导入 nn 模块，nn 模块提供了构建神经网络所需的各种类和函数
from torch import nn

# 定义一个名为 mo 的自定义神经网络模块，继承自 nn.Module 类
# nn.Module 是 PyTorch 中所有神经网络模块的基类
class mo(nn.Module):
    # 类的初始化方法，在创建类的实例时会自动调用
    def __init__(self):
        # 调用父类 nn.Module 的初始化方法，确保正确初始化父类的属性和方法
        super(mo, self).__init__()

    # 定义前向传播方法，这是神经网络模块的核心方法
    # 当调用模块实例时，会自动调用该方法进行计算
    def forward(self, x):
        # 对输入 x 进行加 1 操作
        output = x + 1
        # 返回计算结果
        return output

# 创建 mo 类的一个实例 qi
qi = mo()
# 创建一个值为 1.0 的 PyTorch 张量 x
x = torch.tensor(1.0)
# 调用 qi 实例，实际上是调用其 forward 方法，对输入 x 进行计算
output = qi(x)
# 打印计算结果
print(output)