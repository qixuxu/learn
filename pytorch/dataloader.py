# 从 torch.utils.data 模块导入 DataLoader 类，用于批量加载数据
from torch.utils.data import DataLoader
# 导入 torchvision 库，它提供了很多用于计算机视觉任务的工具，如数据集加载、图像变换等
import torchvision
# 从 torch.utils.tensorboard 模块导入 SummaryWriter 类，用于将数据写入 TensorBoard 日志，方便后续可视化分析
from torch.utils.tensorboard import SummaryWriter

# 使用 torchvision.datasets.CIFAR10 加载 CIFAR - 10 测试数据集
# 第一个参数 "./dataset" 表示数据集的存储路径，如果该路径下没有数据集，会根据 download 参数决定是否下载
# train=False 表示加载的是测试集，若为 True 则加载训练集
# transform=torchvision.transforms.ToTensor() 表示对加载的图像进行转换，将其转换为 PyTorch 张量
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

# 使用 DataLoader 对测试数据集进行包装，实现批量加载数据
# dataset=test_data 表示要加载的数据集
# batch_size=64 表示每个批次加载 64 个样本
# shuffle=True 表示在每个 epoch 开始时打乱数据的顺序，有助于提高模型的泛化能力
# num_workers=0 表示使用主进程加载数据，在 Windows 系统中通常设置为 0 以避免一些兼容性问题
# drop_last=False 表示如果最后一个批次的样本数量不足 batch_size，仍然保留该批次
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 从测试数据集中获取第一个样本，将其图像和对应的标签分别赋值给 img 和 target
img, target = test_data[0]
# 打印图像的形状，通常形状为 (通道数, 高度, 宽度)
print(img.shape)
# 打印该样本对应的标签
print(target)

# 创建一个 SummaryWriter 对象，指定日志文件存储在名为 "logs" 的目录中
writer = SummaryWriter("logs")
# 初始化一个变量 step，用于记录当前处理的批次数量，作为全局步长
step = 0

# 遍历测试数据加载器中的每个批次的数据
for data in test_loader:
    # 从当前批次的数据中解包出图像数据和对应的标签数据
    imgs, targets = data
    # 以下两行代码被注释掉了，原本用于打印当前批次图像的形状和对应的标签
    # print(imgs.shape)
    # print(targets)
    # 使用 SummaryWriter 的 add_images 方法将当前批次的图像数据写入 TensorBoard 日志
    # "test_data" 是该图像数据在 TensorBoard 中的标签
    # imgs 是当前批次的图像数据
    # step 是全局步长，用于区分不同批次的图像
    writer.add_images("test_data", imgs, step)
    # 每处理完一个批次的数据，将 step 的值加 1，更新全局步长
    step = step + 1

# 关闭 SummaryWriter 对象，确保所有数据都被正确写入日志文件，并释放相关资源
writer.close()