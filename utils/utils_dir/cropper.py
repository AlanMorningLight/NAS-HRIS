import argparse
import os
import glob
import cv2
import numpy as np
from functools import partial
from tqdm import tqdm


class Cropper(object):
    def __init__(self, args, configs, predict=None):
        super(Cropper, self).__init__()

        self.args = args
        self.configs = configs
        self.predict = predict

        self.input_path = self.args.input
        self.input_patches_path = os.path.join(self.input_path, 'image_patches')
        self.input_label_path = os.path.join(self.input_path, 'label_patches')

        self.output_path = self.args.output

        if predict:
            self.ensure_and_mkdir(self.input_path)
            self.ensure_and_mkdir(self.input_patches_path)
        else:
            self.ensure_and_mkdir(self.input_path)
            self.ensure_and_mkdir(self.input_patches_path)
            self.ensure_and_mkdir(self.input_label_path)
        # by default
        self.size_x = self.configs.cropped_size
        self.size_y = self.configs.cropped_size
        self.step_x = self.args.step_x
        self.step_y = self.args.step_y

        self.image_margin_color = args.image_margin_color
        self.label_margin_color = args.label_margin_color

    def get_filename(self, path):
        return path.split('\\')[-1].split('.')[0] + '.' + path.split('\\')[-1].split('.')[1] + '.' + path.split('\\')[-1].split('.')[2]

    def ensure_and_mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def pad_and_crop_images(self ,image, margin_color):

        # TODO padding odd and even
        # should the value of border be equal to background ?
        # padding
        image_x = image.shape[1]
        image_y = image.shape[0]
        border_y = 0
        if image_y % self.size_y != 0:
            border_y_double = (self.size_y - (image_y % self.size_y))
            if border_y_double % 2 == 0:
                image = cv2.copyMakeBorder(image, border_y_double//2, border_y_double//2, 0, 0, cv2.BORDER_CONSTANT, value=margin_color)
            else:
                image = cv2.copyMakeBorder(image, border_y_double//2, border_y_double//2+1, 0, 0, cv2.BORDER_CONSTANT, value=margin_color)
            image_y = image.shape[0]
        border_x = 0
        if image_x % self.size_x != 0:
            border_x_double = (self.size_x - (image_x % self.size_x))
            if border_x_double % 2 == 0:
                image = cv2.copyMakeBorder(image, 0, 0, border_x_double//2, border_x_double//2, cv2.BORDER_CONSTANT, value=margin_color)
            else:
                image = cv2.copyMakeBorder(image, 0, 0, border_x_double//2, border_x_double//2+1, cv2.BORDER_CONSTANT, value=margin_color)
            image_x = image.shape[1]

        # calculate n_w and n_h
        n_w = int(image_x / self.size_x)
        n_h = int(image_y / self.size_y)

        # cropping
        # cannot adopt for loop, otherwise the cropped images will include very little margin
        patches = []
        start_y = 0
        while (start_y + self.size_y) <= image_y:
            start_x = 0
            while (start_x + self.size_x) <= image_x:
                patches.append(image[start_y:start_y + self.size_y, start_x:start_x + self.size_x])
                start_x += self.step_x
            start_y += self.step_y

        return patches, n_w, n_h, image_y, image_x

    def save_images(self, patches, save_path, father_name):
        for i, patch in enumerate(patches):
            cv2.imwrite(os.path.join(save_path, father_name + '_' + str(i) + '.tif'), patch)

    def image_processor(self, image_path, label_path):

        image = cv2.imread(image_path)
        filename = self.get_filename(image_path)
        print(filename)
        patches, n_w, n_h, image_h, image_w = self.pad_and_crop_images(image=image, margin_color=self.image_margin_color)
        # patches is saved in input_path/image_patches
        input_path = os.path.join(self.input_path, 'image_patches')
        self.save_images(patches, input_path, filename)

        if self.predict is False:
            assert label_path is not None, \
                'label_path is None'
            label = cv2.imread(label_path)
            label_filename = self.get_filename(label_path)
            label_patches, _, _, _, _ = self.pad_and_crop_images(image=label, margin_color=self.label_margin_color)
            # label_patches is saved in input_path/gt_patches
            input_path = os.path.join(self.input_path, 'label_patches')
            self.save_images(label_patches, input_path, label_filename)

        return patches, label_patches, n_w, n_h, image_h, image_w

