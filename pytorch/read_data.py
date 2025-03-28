from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    # 初始化函数，接收一个idx参数
    def __init__(self,root_dir,label_dir):
        # 初始化函数，传入根目录和标签目录
        self.root_dir=root_dir
        # 将根目录赋值给self.root_dir
        self.label_dir=label_dir
        # 将标签目录赋值给self.label_dir
        self.path= os.path.join(self.root_dir,self.label_dir)
        # 将根目录和标签目录拼接成路径，赋值给self.path
        self.img_path = os.listdir(self.path)
        # 获取路径下的所有文件名，赋值给self.img_path


    # 获取指定索引的数据
    def __getitem__(self, idx):
        # 获取图片路径
        img_name =self.img_path[idx]
        # 获取图片的完整路径
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        # 打开图片
        img = Image.open(img_item_path)
        # 获取标签
        label = self.label_dir
        # 返回图片和标签
        return img,label

# 定义一个函数，用于返回对象的长度
    def __len__(self):
        # 返回对象的img_path属性的长度
        return len(self.img_path)

# 定义根目录，这里是训练数据集所在的根目录
root_dir = 'dataset1/train'
# 定义蚂蚁图片所在的标签目录
ants_label_dir = 'ants'
# 创建一个 MyDataset 类的实例 ants_dataset，用于表示蚂蚁图片数据集
ants_dataset = MyDataset(root_dir, ants_label_dir)

# 定义蜜蜂图片所在的标签目录
bees_label_dir = 'bees'
# 创建一个 MyDataset 类的实例 bees_dataset，用于表示蜜蜂图片数据集
bees_dataset = MyDataset(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset

# 从合并后的数据集（假设已正确合并）中获取索引为 0 的样本
# 得到图片和对应的标签
img, label = train_dataset[0]
# 调用图片对象的 show 方法，显示该图片
img.show()
# 打印该图片对应的标签
print(label)



