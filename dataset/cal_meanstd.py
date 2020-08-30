import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from PIL import Image

class DatasetGID(Dataset):
    """
     root：图像存放地址根路径
     augment：是否需要图像增强
    """
    def __init__(self, root):
        super(DatasetGID, self).__init__()
        # 这个list存放所有图像的地址
        self.image_files = glob.glob(os.path.join(
            root,
            '*'
        ))

    def __getitem__(self, index):
        # 读取图像数据并返回
        # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
        image = Image.open(self.image_files[index])
        image = TF.to_tensor(image)
        return image

    def __len__(self):
        # 返回图像的数量
        return len(self.image_files)

def mean_std_calculator(dataset):

    # mean and std all both from the data used for training, excluding the test data
    loader = DataLoader(
        dataset,
        batch_size = 1,
        num_workers = 1,
        shuffle = False
    )

    mean = 0.
    std = 0.
    num_samples = 0.

    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1) # N C HxW
        mean += data.mean(2).sum(0) # channel-wise independent
        std += data.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    return mean, std


if __name__ == '__main__':
    data = DatasetGID(root='L:\Dataset\Beijing_Google_Buildings\BGB\\train\image')
    mean,std = mean_std_calculator(data)
    print(mean ,std)