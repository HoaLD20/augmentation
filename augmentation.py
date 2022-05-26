import os

import cv2
import random
import numpy as np

"""
/train
    /ade
        a.png
        b.png
    /large
        a.png
        b.png
    /...
"""


"""
Horizontal Shift
Vertical Shift
Brightness
Zoom
Channel Shift
Horizontal Flip
Vertical Flip
Rotation 30 60 90 120
"""


def fill(image, h, w):
    return cv2.resize(image, (h, w), cv2.INTER_CUBIC)


def horizontal_shift(image, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print("Value should be less than 1 and greater than 0")
        return image
    ratio = random.uniform(-ratio, ratio)
    h, w = image.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        image = image[:, :int(w - to_shift), :]
    if ratio < 0:
        image = image[:, int(-1 * to_shift):, :]
    return fill(image, h, w)


def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h * ratio
    if ratio > 0:
        img = img[:int(h - to_shift), :, :]
    if ratio < 0:
        img = img[int(-1 * to_shift):, :, :]
    img = fill(img, h, w)
    return img


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value * h)
    w_taken = int(value * w)
    h_start = random.randint(0, h - h_taken)
    w_start = random.randint(0, w - w_taken)
    img = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
    img = fill(img, h, w)
    return img


def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:, :, :][img[:, :, :] > 255] = 255
    img[:, :, :][img[:, :, :] < 0] = 0
    img = img.astype(np.uint8)
    return img


def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img


def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


path = os.getcwd()
data_path = r"data/"
data_dir_list = os.listdir(os.path.join(data_path))

for data_dir in data_dir_list:
    data_list = os.listdir(os.path.join(data_path, data_dir))
    url_img = os.path.join(data_path, data_dir) + "/"
    os.chdir(os.path.join(data_path, data_dir))
    for i in range(len(data_list)):
        img_name = data_list[i]
        full_url = path + "/" + url_img + img_name
        save_url = path + "/" + url_img + "/"

        img_1 = cv2.imread(full_url)
        img_1 = horizontal_shift(img_1, 0.7)
        cv2.imwrite(save_url + img_name[0:6] + "_horizontal_shift_" + str(i+1) + ".png", img_1)

        img_2 = cv2.imread(full_url)
        img_2 = vertical_shift(img_2, 0.7)
        cv2.imwrite(save_url + img_name[0:6] + "_vertical_shift_" + str(i+1) + ".png", img_2)

        img_3 = cv2.imread(full_url)
        img_3 = zoom(img_3, 0.5)
        cv2.imwrite(save_url + img_name[0:6] + "_zoom_" + str(i+1) + ".png", img_3)

        img_4 = cv2.imread(full_url)
        img = brightness(img_4, 0.5, 3)
        cv2.imwrite(save_url + img_name[0:6] + "_brightness_" + str(i+1) + ".png", img_4)

        img_5 = cv2.imread(full_url)
        img_5 = rotation(img_5, 30)
        cv2.imwrite(save_url + img_name[0:6] + "_rotation30_" + str(i+1) + ".png", img_5)
        img_6 = cv2.imread(full_url)
        img_6 = vertical_flip(img_6, True)
        cv2.imwrite(save_url + img_name[0:6] + "_verticalflip_" + str(i+1) + ".png", img_6)

        img_7 = cv2.imread(full_url)
        img_7 = horizontal_flip(img_7, True)
        cv2.imwrite(save_url + img_name[0:6] + "_horizontalflip_" + str(i+1) + ".png", img_7)

        img_8 = cv2.imread(full_url)
        img_8 = channel_shift(img_8, 60)
        cv2.imwrite(save_url + img_name[0:6] + "_channelshift_" + str(i+1) + ".png", img_8)

        img_9 = cv2.imread(full_url)
        img_9 = rotation(img_9, 60)
        cv2.imwrite(save_url + img_name[0:6] + "_rotation60_" + str(i+1) + ".png", img_9)

        img_10 = cv2.imread(full_url)
        img_10 = rotation(img_10, 120)
        cv2.imwrite(save_url + img_name[0:6] + "_rotation120_" + str(i+1) + ".png", img_10)

        img_11 = cv2.imread(full_url)
        img_11 = rotation(img_11, 90)
        cv2.imwrite(save_url + img_name[0:6] + "_rotation90_" + str(i+1) + ".png", img_11)
    os.chdir(path)
print("******************* Done Processing Task ******************* ")


for data_dir in data_dir_list:
    print('Renaming files from folder: {}'.format(data_dir))
    data_list = os.listdir(os.path.join(data_path, data_dir))
    os.chdir(os.path.join(data_path, data_dir))

    # The base name of image files
    base_name = data_dir.replace(" ", "_")

    for i in range(len(data_list)):
        img_name = data_list[i]
        if (img_name.lower().endswith('.jpg')):
            img_rename = base_name + '_{:06d}'.format(i + 1) + '.jpg'  # here the file name is base_name_000001.jpg
        elif (img_name.lower().endswith('.jpeg')):
            img_rename = base_name + '_{:06d}'.format(i + 1) + '.jpeg'  # here the file name is base_name_000001.jpeg
        else:
            img_rename = base_name + '_{:06d}'.format(i + 1) + '.png'  # here the file name is base_name_000001.png

        if not os.path.exists(img_rename):
            os.rename(img_name, img_rename)

    print('------Complete renaming all folders: {}------'.format(data_dir))
    os.chdir(path)
print('----Complete renaming files from all folders----')