import os
import re


def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))

    return file_list


# suffix_label = ['txt']
# label = Walk('/home/indemind/datasets/c1.0.2/ABBY/labels/TRAIN/', suffix_label)
# print(label[54370])
# print(len(label))
# print('-------------------------------')
#
# suffix_img = ['jpg', 'png', 'JPEG']
# img = Walk('/home/indemind/datasets/c1.0.2/ABBY/JPEGImages/TRAIN/', suffix_img)
# print(img[54370])
# print(len(img))

# 有的标签没有图片，怎么解决，是重新处理图片，还是对比删除
# 在裁剪的时候增加判断，不同的图片类型顺序还是不对
# 强制循环，找一样的

