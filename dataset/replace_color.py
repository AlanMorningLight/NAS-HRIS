import os

dataset_path = "/home/zhangmingwei/dataset/GBG"
dataset = ['train','valid','test']
for set in dataset :
    source = os.path.join(dataset_path,set)
    source_image = os.path.join(source, 'image')
    source_label = os.path.join(source, 'label')
    pathDir_image = os.listdir(source_image)
    for image_name in pathDir_image :
        label_name = image_name
        old_image_path = source_image + '\\' + image_name
        old_label_path = source_label + '\\' + label_name
        new_image_path = source_image + '\\' + image_name.split('.')[0] + '.png'
        new_label_path = source_label + '\\' + label_name.split('.')[0] + '.png'
        os.rename(old_image_path,new_image_path)
        os.rename(old_label_path,new_label_path)