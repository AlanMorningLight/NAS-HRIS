from PIL import Image
import argparse
import os
import cv2
import sys
import numpy as np
sys.path.append("..")

parser = argparse.ArgumentParser(description='configurations')
parser.add_argument('--gpu', type=int, default=0,help='0 and 1 means gpu id, and -1 means cpu')
parser.add_argument('-i', '--input', type=str, default='L:\Dataset\Land-use datasetGID\GID_dataset_3000\GID_dataset',
                    help='directory of input images, including images used to train and predict')
args = parser.parse_args()

source_image = os.path.join(args.input, 'image')
source_label = os.path.join(args.input, 'label')
#filename = source_image_path.split('\\')[-1].split('.')[0]+'.'+source_image_path.split('\\')[-1].split('.')[1]+'.'+source_image_path.split('\\')[-1].split('.')[2]
pathDir_image = os.listdir(source_image)
for image_name in pathDir_image:
    label_name = image_name.split('_')[0] + '_' + image_name.split('_')[1] + '_' + \
                 image_name.split('_')[2] + '_' + image_name.split('_')[3] + '_' + \
                 image_name.split('_')[4] + '_' + image_name.split('_')[5] + '_' + \
                 "label_"+image_name.split('_')[6]
    source_image_path = os.path.join(source_image, image_name)
    source_label_path = os.path.join(source_label, label_name)
    print(source_label_path)
    img = Image.open(source_label_path)#读取系统的内照片
    width = img.size[0]#长度
    height = img.size[1]#宽度
    if (width != 512) or (height != 512):
        if os.path.exists(source_image_path):
            os.remove(source_image_path)
            print("^---delate---^")
        if os.path.exists(source_label_path):
            os.remove(source_label_path)
    #f = open('/home/zhangmingwei/NAS/NAS-RSI1/dataset/mistake_label.txt','a+')#保存一下图片像素看一下
    for i in range(0,width):#遍历所有长度的点
        for j in range(0,height):#遍历所有宽度的点
            data = (img.getpixel((i,j)))#打印该图片的所有点
            #f.write(source_label_path + "\n")
            #f.write(str(data)+"("+str(i)+","+str(j)+")\n")#看一下像素
            #print (str(data))#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            x = [data[0],data[1],data[2]]
            y = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255]]
            if (x not in y) :
                #判断条件就是一个像素范围范围
                #print (str(x))#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
                if os.path.exists(source_image_path):
                    os.remove(source_image_path)
                    print("^---delate---^")
                if os.path.exists(source_label_path):
                    os.remove(source_label_path)
    #f.close()
    img = img.convert("RGB")#把图片强制转成RGB