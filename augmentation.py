import PIL.Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os

data_path = r"data"


def imshow(img, transform):
    """helper function to show data augmentation
    :param img: path of the image
    :param transform: data augmentation technique to apply"""

    img = PIL.Image.open(img)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].set_title(f'original image {img.size}')
    ax[0].imshow(img)
    img = transform(img)
    ax[1].set_title(f'transformed image {img.size}')
    ax[1].imshow(img)


data_dir_list = os.listdir(os.path.join(data_path))
print(data_dir_list)
for data_dir in data_dir_list:
    data_list = os.listdir(os.path.join(data_path, data_dir))
    print(data_dir)
    for i in range(len(data_list)):
        img_name = data_list[i]
        print(img_name)
