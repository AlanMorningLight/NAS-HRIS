import argparse
import os
import cv2
import sys
import numpy as np
sys.path.append("..") # 这句是为了导入_config
from utils.cropper import Cropper


def config_parser():

    parser = argparse.ArgumentParser(description='configurations')
    parser.add_argument('--gpu', type=int, default=0,help='0 and 1 means gpu id, and -1 means cpu')
    parser.add_argument('-i', '--input', type=str, default='L:\Dataset\Land-use datasetGID\GID_dataset_3000',
                        help='directory of input images, including images used to train and predict')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('L:\Dataset\Land-use datasetGID\GID_dataset_3000', 'GID_dataset'),
                        help='directory of output images, for predictions')
    parser.add_argument('--ckpt_path', type=str, default=os.path.join('.', 'checkpoint-best.pth'),
                        help='path to the checkpoint file, default name checkpoint-best.pth')
    # dataloader settings
    parser.add_argument('--pin_memory', type=bool, default=False,
                        help='When True, it will accelerate the prediction phase but with high CPU-Utilization, and it '
                             'will also allocate additional GPU-Memory')
    parser.add_argument('--nb_workers', type=int, default=1, help='workers for DataLoader')
    parser.add_argument('--cropped_size', type=int, default=512,help='the horizontal step of cropping images')
    parser.add_argument('--step_x', type=int, default=512,help='the horizontal step of cropping images')
    parser.add_argument('--step_y', type=int, default=512,help='the vertical step of cropping images')
    # patches settings
    parser.add_argument('--image_margin_color', type=list, default=[0, 0, 0],help='the color of image margin color')
    parser.add_argument('--label_margin_color', type=list, default=[0, 0, 0],help='the color of label margin color')

    return parser.parse_args()

def patchify_and_unpatchify():

    args = config_parser()
    source_image = os.path.join(args.input, 'images')
    source_label = os.path.join(args.input, 'labels')
    #filename = source_image_path.split('\\')[-1].split('.')[0]+'.'+source_image_path.split('\\')[-1].split('.')[1]+'.'+source_image_path.split('\\')[-1].split('.')[2]
    pathDir_image = os.listdir(source_image)
    for image_name in pathDir_image:
        label_name = image_name[:-4] + "_label.tif"
        source_image_path = os.path.join(source_image, image_name)
        source_label_path = os.path.join(source_label, label_name)
        print(source_image_path)
        print(source_label_path)
        c = Cropper(args=args, predict=False)
        patches, label_patches, n_w, n_h, image_h, image_w = c.image_processor(image_path=source_image_path,label_path=source_label_path)
    '''np_patches = np.asarray(patches)
    np_patches = np_patches.reshape(n_h, n_w, config.cropped_size, config.cropped_size, 3)
    img = unpatchify(np_patches, image_h, image_w)
    save_path = os.path.join(args.output, 'remerge', filename+'.tif')
    cv2.imwrite(save_path, img)

    np_label_patches = np.asarray(label_patches)
    np_label_patches = np_label_patches.reshape(n_h, n_w, config.cropped_size, config.cropped_size, 3)
    lab = unpatchify(np_label_patches, image_h, image_w)
    save_path = os.path.join(args.output, 'remerge_label', filename + '.tif')
    cv2.imwrite(save_path, lab)'''

if __name__ == '__main__':
    patchify_and_unpatchify()