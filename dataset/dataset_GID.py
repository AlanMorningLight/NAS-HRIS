import os
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import glob
import random
import numpy as np
from torch.utils.data import Dataset

rgb_mean = (0.3704, 0.3902, 0.3629)
rgb_std = (0.1869, 0.1768, 0.1651)

class MyDataset(Dataset):
    def __init__(self,
                 args,
                 subset):
        super(MyDataset, self).__init__()
        assert subset == 'train' or subset == 'valid' or subset == 'test'

        self.args = args
        self.root = args.data
        self.subset = subset
        self.data = self.args.data_folder_name # image
        self.target = self.args.target_folder_name # label

        #self.data_transforms = data_transforms if data_transforms!=None else TF.to_tensor
        #self.target_transforms = target_transforms if target_transforms!= None else TF.to_tensor

        self.mapping = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
            (0, 255, 0): 2,
            (0, 255, 255): 3,
            (255, 255, 0): 4,
            (0, 0, 255): 5,
        }
        #讲原图和标记数据列表读入
        self.data_list = sorted(glob.glob(os.path.join(
            self.root,
            subset,
            self.data,
            '*'
        )))
        self.target_list = sorted(glob.glob(os.path.join(
            self.root,
            subset,
            self.target,
            '*'
        )))


    def mask_to_class(self, mask):
        for k in self.mapping:
            #knp = np.array(k)
            #ktensor = torch.from_numpy(knp).byte()
            mask[(mask == torch.tensor(k, dtype = torch.uint8)).all(dim = 2)] = self.mapping[k]
            #mask[mask == ktensor] = self.mapping[k]
        return mask[:,:,0]
        #return mask

    def train_transforms(self, image, mask):

        #将短边resize到input_size
        resize = transforms.Resize(size=(self.args.input_size, self.args.input_size), interpolation=0)
        image = resize(image)
        mask = resize(mask)

        #随机水平翻转图像
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        #随机垂直翻转图像
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        #将图像转为tensor，并进行归一化处理
        image = TF.to_tensor(image) # scale 0-1
        image = TF.normalize(image, mean=rgb_mean, std=rgb_std) # normalize

        #从
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        mask = self.mask_to_class(mask)
        mask = mask.long()
        return image, mask

    def untrain_transforms(self, image, mask):

        resize = transforms.Resize(size=(self.args.input_size, self.args.input_size), interpolation=0)
        image = resize(image)
        mask = resize(mask)

        # 没有旋转的变化

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=rgb_mean, std=rgb_std)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        mask = self.mask_to_class(mask)
        mask = mask.long()
        return image, mask

    #实例化对象P，可以用P[key]进行取值
    def __getitem__(self, index):

        datas = Image.open(self.data_list[index])
        targets = Image.open(self.target_list[index])
        if self.subset == 'train':
            '''print(index)
            print(self.data_list[index])
            print(self.target_list[index])'''
            t_datas, t_targets = self.train_transforms(datas, targets)
            return t_datas, t_targets
        elif self.subset == 'valid':
            t_datas, t_targets = self.untrain_transforms(datas, targets)
            return t_datas, t_targets
        elif self.subset == 'test':
            t_datas, t_targets = self.untrain_transforms(datas, targets)
            return t_datas, t_targets, self.data_list[index]

    #返回数据的数量
    def __len__(self):
        return len(self.data_list)