import os
import cv2
import numpy as np
from walk import Walk


def read_coordinates_from_txt(txt_file):
    coordinates = []
    id_list = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            img = line.strip().split()
            x = float(img[1])
            y = float(img[2])
            width = float(img[3])
            height = float(img[4])
            id = int(img[0])
            # x, y, width, height = map(int, line.strip().split())
            id_list.append(id)
            coordinates.append((x, y, width, height))
    return coordinates, id_list


def crop_images_from_coordinates(image_file, coordinates):
    image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape
    cropped_images = []
    for (x, y, width, height) in coordinates:
        x = int(x * image_width)  # 这个是中心坐标
        y = int(y * image_height)
        crop_width = int(width * image_width)
        crop_height = int(height * image_height)
        x_start = x - crop_width // 2  # 裁剪区域的左上角x坐标
        # x_start = x  # 裁剪区域的左上角x坐标
        y_start = y - crop_height // 2  # 裁剪区域的左上角y坐标
        # y_start = y  # 裁剪区域的左上角y坐标
        x_end = x_start + crop_width  # 裁剪区域的右下角x坐标
        y_end = y_start + crop_height  # 裁剪区域的右下角y坐标
        # 正方形裁剪
        if crop_width > crop_height:
            y_start = y - crop_width // 2
            y_end = y_start + crop_width
        else:
            x_start = x - crop_height // 2
            x_end = x_start + crop_height
        # 判断边界
        cropped_image = image[np.clip(y_start-15, 0, image_height):np.clip(y_end+15, 0, image_height), np.clip(x_start-15, 0, image_width):np.clip(x_end+15, 0, image_width)]
        cropped_images.append(cropped_image)
    return cropped_images


def save(cropped_images, id_list):
    for i, (cropped_image, id) in enumerate(zip(cropped_images, id_list)):
        if id == 0:
            path = '/home/indemind/Project/data_crop/cropped/shoes'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        elif id == 1:
            path = '/home/indemind/Project/data_crop/cropped/bin'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        elif id == 2:
            path = '/home/indemind/Project/data_crop/cropped/pedestal'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        elif id == 3:
            path = '/home/indemind/Project/data_crop/cropped/wire'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        elif id == 4:
            path = '/home/indemind/Project/data_crop/cropped/socket'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        elif id == 5:
            path = '/home/indemind/Project/data_crop/cropped/cat'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        elif id == 6:
            path = '/home/indemind/Project/data_crop/cropped/dog'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        elif id == 7:
            path = '/home/indemind/Project/data_crop/cropped/desk_rect'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        elif id == 8:
            path = '/home/indemind/Project/data_crop/cropped/desk_round'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)
        else:
            path = '/home/indemind/Project/data_crop/cropped/weighing-scale'
            cv2.imwrite(f'{path}/{img_name}_{i}.jpg', cropped_image)


# 获取标签文件路径
suffix_label = ['txt']
label = Walk('/home/indemind/datasets/c1.0.2/ABBY/labels/TRAIN', suffix_label)

# 获取图片路径
suffix_img = ['jpg', 'png', 'JPEG']
img = Walk('/home/indemind/datasets/c1.0.2/ABBY/JPEGImages/TRAIN', suffix_img)

for i, txt_file in enumerate(label):
    txt_name1 = txt_file.split('/')[-1]
    txt_name = txt_name1.split('.')[0]
    coordinates, id_list = read_coordinates_from_txt(txt_file)

    for image_file in img:
        img_name1 = image_file.split('/')[-1]
        img_name = img_name1.split('.')[0]

        if txt_name == img_name:
            cropped_images = crop_images_from_coordinates(image_file, coordinates)
            save(cropped_images, id_list)
            break

# 生成裁剪后图片路径加标签
list = ['/home/indemind/Project/data_crop/cropped/shoes', '/home/indemind/Project/data_crop/cropped/bin', '/home/indemind/Project/data_crop/cropped/pedestal',
        '/home/indemind/Project/data_crop/cropped/wire', '/home/indemind/Project/data_crop/cropped/socket', '/home/indemind/Project/data_crop/cropped/cat',
        '/home/indemind/Project/data_crop/cropped/dog', '/home/indemind/Project/data_crop/cropped/desk_rect', '/home/indemind/Project/data_crop/cropped/desk_round',
        '/home/indemind/Project/data_crop/cropped/weighing-scale']
for path in list:
    crop_suffix_img = ['jpg', 'png', 'JPEG']
    crop_img = Walk(path, crop_suffix_img)
    label = str(path.split('/')[-1])
    class_to_idx = {'shoes': 0, 'bin': 1, 'pedestal': 2, 'wire': 3, 'socket': 4, 'cat': 5, 'dog': 6, 'desk_rect': 7,
                    'desk_round': 8, 'weighing-scale': 9}
    id = class_to_idx.get(label)
    name = label + '.txt'
    f = open(name, 'w', encoding='utf-8')
    for num, line in enumerate(crop_img):
        f.writelines(line + ' ' + str(id) + '\n')
    f.close()

