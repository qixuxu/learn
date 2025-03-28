# 从 torch.utils.tensorboard 模块导入 SummaryWriter 类，
# 用于将数据记录到 TensorBoard 中，以便可视化分析
from torch.utils.tensorboard import SummaryWriter
# 导入 numpy 库，用于处理数值数组和进行数值计算
import numpy as np
# 从 PIL 库中导入 Image 模块，用于处理图像（如打开、转换等操作）
from PIL import Image

# 创建一个 SummaryWriter 对象，指定日志文件存储在名为 "logs" 的目录中
# 后续将使用该对象将数据写入到对应的日志文件中
writer = SummaryWriter("logs")

# 定义图像文件的路径，这里指向了一个特定的蚂蚁图像文件
image_path = "data/val/ants/1119630822_cd325ea21a.jpg"

# 使用 Image.open() 方法打开指定路径的图像文件，
# 并将其存储为一个 PIL 图像对象
img_PIL = Image.open(image_path)

# 将 PIL 图像对象转换为一个 NumPy 数组，
# 因为 TensorBoard 的 add_image 方法在处理图像数据时，
# 更方便接收 NumPy 数组形式的数据
img_array = np.array(img_PIL)

# 打印转换后数组的类型，通常会输出 <class 'numpy.ndarray'>
print(type(img_array))

# 打印数组的形状，例如 (高度, 宽度, 通道数)，
# 这有助于了解图像的尺寸和通道信息
print(img_array.shape)

# 使用 SummaryWriter 的 add_image 方法将图像数据添加到 TensorBoard 日志中
# "test" 是该图像在 TensorBoard 中的标签，用于标识和区分不同的图像数据
# img_array 是要添加的图像数据，为之前转换得到的 NumPy 数组
# dataformats="HWC" 表示数据格式为 (Height, Width, Channels)，即高度、宽度和通道数的顺序
writer.add_image("test", img_array, dataformats="HWC")

# 使用循环，从 0 到 99 迭代 100 次
for i in range(100):
    # 使用 SummaryWriter 的 add_scalar 方法将标量数据添加到 TensorBoard 日志中
    # "y=2x" 是该标量数据在 TensorBoard 中的标签
    # 3*i 是要记录的标量值，这里不是严格按照 y = 2x 的关系，而是 3x
    # i 是全局步长（通常表示训练步数等），用于标识数据的顺序和位置
    writer.add_scalar("y=2x", 3*i, i)

# 关闭 SummaryWriter 对象，确保所有数据都被正确写入到日志文件中，
# 并释放相关资源
writer.close()