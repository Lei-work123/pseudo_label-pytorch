import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from torchvision import transforms
from architectures.arch import arch
from SquarePad import SquarePad, resize_image, Cropsize
from PIL import Image
import time

if __name__ == '__main__':
    list = ["./label/airplane", "./label/automobile", "./label/bird", "./label/cat", "./label/deer", "./label/dog",
            "./label/frog", "./label/horse", "./label/ship", "./label/truck"]
    start = time.time()
    for dir_loc in list:
        # dir_loc = "./label/dog"
        lis = dir_loc.split('/')
        # print(lis[-1])
        with torch.no_grad():
            for a, b, c in os.walk(dir_loc):
                all = 0
                corrcet = 0
                for filei in c:
                    full_path = os.path.join(a, filei)
                    device = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'

                    # 读取要预测的图片,并作预处理
                    img = Image.open(full_path)
                    img = img.convert('RGB')

                    # 进行padding操作
                    # img = resize_image(img, 260, 260)
                    # img = Image.fromarray(img)
                    # size = Cropsize(img)
                    # corp = transforms.Compose([
                    #     transforms.CenterCrop(size)
                    # ])
                    #
                    # img = corp(img)
                    # image = np.ascontiguousarray(img)

                    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])

                    test_transforms = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize(**channel_stats)
                    ])

                    img = test_transforms(img)
                    img = img.to(device)
                    img = img.unsqueeze(0)  # 图片扩展多一维,增加batch_size维度

                    # 加载模型
                    model = arch['resnet18'](10)
                    model.to(device)
                    checkpoint = torch.load('/home/indemind/Project/pseudo_label/results/model1_epoch_495.pth',
                                            map_location=device)
                    model.load_state_dict(checkpoint["weight"])
                    model.eval()

                    # 进行预测
                    output = model(img)
                    prob = F.softmax(output, dim=1)  # prob是10个分类的概率
                    value, predicted = torch.max(prob, 1)

                    # 计算准确率
                    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
                    all = all + 1  # 总共预测图片数量
                    if classes[predicted.item()] == lis[-1]:
                        corrcet = corrcet + 1
                    else:
                        image = cv2.imread(full_path)
                        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (320, 320), 1.0, 1.0, cv2.INTER_CUBIC)
                        cv2.putText(image, classes[predicted.item()], (0, 28),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                        # cv2.imwrite('{}.jpg'.format(filei), image)
                        path = './test_result'
                        cv2.imwrite('{}/{}.jpg'.format(path, filei), image)

                    # print("预测类别为： ", classes[predicted.item()], " 可能性为: ", value.item() * 100, "%")

                    # 把标签添加到原始图片上
                    # image = cv2.resize(image, (320, 320), 1.0, 1.0, cv2.INTER_CUBIC)
                    # cv2.putText(image, classes[predicted.item()], (0, 28),
                    #             cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                    # cv2.imwrite('{}.jpg'.format(filei), image)
                    # # 展示图片
                    # cv2.imshow('image with text', image)
                    # key = cv2.waitKey(0)
                    # if key & 0xFF == ord('q'):
                    #     exit()
                # 计算准确率
                acc = corrcet / all

                print("预测类别为：{}".format(lis[-1]), "预测图片数量为：{}".format(all),
                      "预测正确图片数量：{}".format(corrcet), "准确率为：{}%".format(acc * 100))
    end = time.time()
    all_time = end - start
    print("预测时间为：{}".format(all_time))
