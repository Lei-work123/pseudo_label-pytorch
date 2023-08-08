from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

def SquarePad(image):
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, mode='constant', value=0)


# 入参是：源图，目标宽度，目标长度
# 长宽不等的图片，本函数resize后会在短的那个边上做填充，从而保证图片比例不变
def resize_image(srcimg, targetHeight, targetWidth):
    padh, padw, newh, neww = 0, 0, targetHeight, targetWidth
    if srcimg.shape[0] != srcimg.shape[1]:
        hw_scale = srcimg.shape[0] / srcimg.shape[1]  # shape[0]-h, shape[1]-w
        if hw_scale > 1:  # 高度比较大
            newh, neww = targetHeight, int(targetWidth / hw_scale)
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            padw = int((targetWidth - neww) * 0.5)
            img = cv2.copyMakeBorder(img, 0, 0, padw, targetWidth - neww - padw, cv2.BORDER_CONSTANT,
                                     value=0)  # add border
            return img
        else:
            newh, neww = int(targetHeight * hw_scale) + 1, targetWidth
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            padh = int((targetHeight - newh) * 0.5)
            img = cv2.copyMakeBorder(img, padh, targetHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
            return img
    else:
        img = cv2.resize(srcimg, (targetWidth, targetHeight), interpolation=cv2.INTER_AREA)
        return img  # newh, neww, padh, padw


def Cropsize(img):
    width, height = img.size
    cropsize = 0
    if width > height:
        cropsize = height
    else:
        cropsize = width

    return cropsize





