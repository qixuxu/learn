# 导入 torchvision 库，它提供了许多用于计算机视觉任务的工具，
# 如数据集加载、图像变换等
import torchvision
# 从 torch.utils.tensorboard 模块导入 SummaryWriter 类，
# 用于将数据写入 TensorBoard 日志文件，方便后续可视化分析
from torch.utils.tensorboard import SummaryWriter

# 使用 torchvision.transforms.Compose 函数创建一个图像变换组合对象
# 这里仅包含一个变换：将图像转换为张量（Tensor）
# 图像在输入到神经网络之前通常需要转换为张量形式
datadet_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 加载 CIFAR - 10 训练数据集
# root='./dataset' 表示数据集将存储在当前目录下的 dataset 文件夹中
# train=True 表示加载的是训练集
# transform=datadet_transform 表示对加载的图像应用之前定义的变换
# download=False 表示不下载数据集，如果数据集不存在则会报错
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=datadet_transform, download=False)

# 加载 CIFAR - 10 测试数据集
# root='./dataset' 同样指定数据集存储目录
# train=False 表示加载的是测试集
# transform=datadet_transform 应用相同的图像变换
# download=False 不下载数据集
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=datadet_transform, download=False)

# 以下几行代码被注释掉了，它们的作用是查看测试集中单个样本的信息
# print(test_set[0]) 打印测试集第一个样本的信息
# print(test_set.classes) 打印 CIFAR - 10 数据集的所有类别名称
# img, target = test_set[0] 解包第一个样本，得到图像和对应的标签
# print(img) 打印图像的张量信息
# print(target) 打印标签信息
# print(test_set.classes[target]) 根据标签索引打印对应的类别名称
# img.show() 显示图像（不过由于 img 是张量，此方法可能无法直接使用）
# print(test_set[0]) 再次打印测试集第一个样本的信息

# 创建一个 SummaryWriter 对象，指定日志文件存储在名为 "p10" 的目录中
writer = SummaryWriter("p10")

# 循环遍历测试集的前 10 个样本
for i in range(10):
    # 解包当前样本，得到图像和对应的标签
    img, target = test_set[i]
    # 使用 SummaryWriter 的 add_image 方法将图像添加到 TensorBoard 日志中
    # "test_set" 是该图像在 TensorBoard 中的标签
    # img 是要添加的图像张量
    # i 是全局步长，用于区分不同的图像
    writer.add_image("test_set", img, i)

# 关闭 SummaryWriter 对象，确保所有数据都被正确写入日志文件，并释放相关资源
writer.close()
