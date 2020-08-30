# coding: utf-8
import os
import random
import shutil


def moveFile():
    rootpath = 'L:\\NAS\\Neural Architecture Search for High-resolution Remote Sensing Image Segmentation\GID_512\GID_dataset'

    imagepath = os.path.join(rootpath,"image")
    labelpath = os.path.join(rootpath,"label")

    path = os.path.join(rootpath,"GID_1398")
    train_dir = os.path.join(path,"train")
    train_image_dir =os.path.join(train_dir,"image")
    train_label_dir =os.path.join(train_dir,"label")

    valid_dir = os.path.join(path,"valid")
    valid_image_dir =os.path.join(valid_dir,"image")
    valid_label_dir =os.path.join(valid_dir,"label")

    test_dir = os.path.join(path, "test")
    test_image_dir = os.path.join(test_dir, "image")
    test_label_dir = os.path.join(test_dir, "label")

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)

    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(valid_image_dir):
        os.makedirs(valid_image_dir)
    if not os.path.exists(valid_label_dir):
        os.makedirs(valid_label_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(test_image_dir):
        os.makedirs(test_image_dir)
    if not os.path.exists(test_label_dir):
        os.makedirs(test_label_dir)


    pathDir = os.listdir(imagepath)  # 取图片的原始路径
    filenumber = 1398
    rate_validAndtest = 0.4 #valid 和 test 集所占比例
    rate_test = 0.2 #test集所占比例
    rate_test = rate_test / rate_validAndtest
    number_validAndtest = int(filenumber * rate_validAndtest)  # 按照rate比例从文件夹中取数据
    sample_validAnetest = random.sample(pathDir, number_validAndtest)  # 随机选取number数量的数据
    number_test = int(number_validAndtest * rate_test)  # 按照rate比例从文件夹中取数据
    sample_test = random.sample(sample_validAnetest, number_test)
    # print (sample)
    for name in pathDir:
        num = name.split('_')[6].split('.')[0]
        first_name = name.split('_')[0] + '_' + name.split('_')[1] + '_'+ name.split('_')[2] + '_' \
                     + name.split('_')[3] + '_' + name.split('_')[4] + '_' + name.split('_')[5]
        labelname = first_name + '_label_' + num + '.tif'
        #labelname = name
        print(name)
        if (name in sample_test):
            shutil.copy(imagepath + '/' + name, test_image_dir)
            shutil.copy(labelpath + '/' + labelname, test_label_dir)
        elif (name in sample_validAnetest):
            shutil.copy(imagepath + '/' + name, valid_image_dir)
            shutil.copy(labelpath + '/' + labelname, valid_label_dir)
        else:
            shutil.copy(imagepath + '/' + name, train_image_dir)
            shutil.copy(labelpath + '/' + labelname, train_label_dir)
    return

if __name__ == '__main__':
    moveFile()
