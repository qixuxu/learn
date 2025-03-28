# 从 torchvision 库中导入 transforms 模块
# transforms 模块提供了一系列用于图像预处理和数据增强的工具
from torchvision import transforms
# 从 torch.utils.tensorboard 库中导入 SummaryWriter 类
# SummaryWriter 类用于将训练过程中的各种信息（如损失值、图像等）写入 TensorBoard 日志文件，方便后续可视化分析
from torch.utils.tensorboard import SummaryWriter
# 从 PIL 库（Python Imaging Library）中导入 Image 模块
# Image 模块用于处理和操作图像文件，例如打开、保存、调整大小等
from PIL import Image

# 定义图像文件的路径
# 这里指定了一张蚂蚁图像的文件路径，该图像位于 'data/train/ants_image' 目录下，文件名为 '7759525_1363d24e88.jpg'
img_path = 'data/train/ants_image/7759525_1363d24e88.jpg'
# 使用 Image.open() 函数打开指定路径的图像文件
# 并将打开的图像对象赋值给变量 img，后续可以对该图像进行各种操作
img = Image.open(img_path)
# 创建一个 SummaryWriter 对象
# 指定日志文件的保存目录为 'logs'，后续写入的信息都会存储在这个目录下的日志文件中
writer = SummaryWriter('logs')

# 创建一个 ToTensor 类型的转换对象
# ToTensor 是 transforms 模块中的一个类，用于将 PIL 图像或 NumPy 数组转换为 PyTorch 张量（Tensor）
# 转换后的张量数据类型为 torch.FloatTensor，并且像素值会被归一化到 [0, 1] 范围内
tensor_transform = transforms.ToTensor()

# 使用刚刚创建的 tensor_transform 对象对图像进行转换
# 将 PIL 图像 img 转换为 PyTorch 张量，并将结果赋值给变量 tensor_img
tensor_img = tensor_transform(img)

# 向 TensorBoard 日志中添加一张图像
# 第一个参数 'ants_image' 是图像在 TensorBoard 中的标签，用于在可视化界面中标识该图像
# 第二个参数 tensor_img 是要添加的图像张量
# 第三个参数 0 是图像的全局步数（global step），通常用于在训练过程中区分不同阶段的图像，这里设置为 0
writer.add_image('ants_image', tensor_img, 0)
# 关闭 SummaryWriter 对象
# 确保所有缓存的数据都被写入日志文件，释放相关资源
writer.close()



