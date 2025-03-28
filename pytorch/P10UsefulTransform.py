# 从 PIL 库中导入 Image 模块，用于处理图像文件
from PIL import Image
# 从 torch.utils.tensorboard 库中导入 SummaryWriter 类，用于将数据写入 TensorBoard 日志文件
from torch.utils.tensorboard import SummaryWriter
# 从 torchvision 库中导入 transforms 模块，该模块提供了一系列图像变换操作
from torchvision import transforms

# 创建一个 SummaryWriter 对象，指定日志文件存储的目录为 'logs'
# 后续将使用该对象将图像数据和其他信息写入 TensorBoard 日志
writer = SummaryWriter('logs')

# 使用 Image.open 方法打开指定路径的图像文件
# 这里的图像文件是 dataset1 文件夹下 train 子文件夹中 bees 子文件夹里的 39747887_42df2855ee.jpg 图片
img = Image.open('dataset1/train/bees/39747887_42df2855ee.jpg')

# 创建一个 ToTensor 变换对象，该变换可以将 PIL 图像或 NumPy 数组转换为 PyTorch 张量
trans_totensor = transforms.ToTensor()
# 对打开的图像进行 ToTensor 变换，将图像转换为张量形式
img_tensor = trans_totensor(img)
# 使用 SummaryWriter 对象的 add_image 方法将转换后的张量图像写入 TensorBoard 日志
# 第一个参数 'ToTensor' 是该图像在 TensorBoard 中的标签，第二个参数是要写入的图像张量
writer.add_image("ToTensor", img_tensor)
# 打印图像张量中第一个通道、第一行、第一列的像素值
print(img_tensor[0][0][0])

# 创建一个 Normalize 变换对象，用于对图像张量进行归一化处理
# 第一个参数 [1, 3, 5] 是每个通道的均值，第二个参数 [3, 2, 1] 是每个通道的标准差
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
# 对之前得到的图像张量进行归一化变换
img_norm = trans_norm(img_tensor)
# 打印归一化后图像张量中第一个通道、第一行、第一列的像素值
print(img_norm[0][0][0])
# 将归一化后的图像张量写入 TensorBoard 日志，设置全局步长为 2
# 全局步长可以用于区分不同阶段的数据
writer.add_image("Normalize", img_norm, 2)

# 打印原始图像的尺寸，尺寸以 (宽度, 高度) 的形式输出
print(img.size)
# 创建一个 Resize 变换对象，指定将图像调整为 (512, 512) 的尺寸
trans_resize = transforms.Resize((512, 512))
# 对原始图像进行尺寸调整变换
img_resize = trans_resize(img)
# 将调整尺寸后的图像转换为张量形式
img_resize = trans_totensor(img_resize)
# 将调整尺寸后的图像张量写入 TensorBoard 日志，设置全局步长为 0
writer.add_image("Resize", img_resize, 0)
# 打印调整尺寸并转换为张量后的图像数据
print(img_resize)

# 创建另一个 Resize 变换对象，指定将图像的短边调整为 512 像素，长边按比例缩放
trans_resize_2 = transforms.Resize(512)
# 创建一个 Compose 变换对象，将多个变换组合在一起
# 这里先进行 Resize 变换，再进行 ToTensor 变换
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
# 对原始图像依次应用组合变换
img_resize_2 = trans_compose(img)
# 将经过组合变换后的图像张量写入 TensorBoard 日志，设置全局步长为 1
writer.add_image("Resize", img_resize_2, 1)

# 创建一个 RandomCrop 变换对象，用于随机裁剪图像，裁剪后的尺寸为 224x224
trans_random = transforms.RandomCrop(224)
# 创建另一个 Compose 变换对象，先进行随机裁剪变换，再进行 ToTensor 变换
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
# 循环 10 次，每次对原始图像应用组合变换得到不同的随机裁剪结果
for i in range(10):
    img_crop = trans_compose_2(img)
    # 将每次随机裁剪并转换为张量后的图像写入 TensorBoard 日志，全局步长依次为 0 到 9
    writer.add_image("RandomCrop", img_crop, i)

# 关闭 SummaryWriter 对象，释放资源并确保所有数据都被写入日志文件
writer.close()