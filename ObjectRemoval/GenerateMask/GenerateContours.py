import torch
import numpy as np
import ExtractContour.augmentation as augment
from PIL import Image
import cv2
from ExtractContour.model import *
import torchvision.transforms as transformer


def readPoints():
    file = open("points.txt", "r")
    x0, x1, y0, y1 = file.readline().strip().split('\t')
    fig_path = file.readline().strip('\n')
    return fig_path, [int(x0), int(x1)], [int(y0), int(y1)]


def getContour(path, point1, point2):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    img = Image.open(path)
    path = path.replace(".png", ".jpg")
    img = img.convert("RGB")
    img = img.crop((point1[0], point1[1], point2[0], point2[1]))
    # cut_path = fig_path.replace(".jpg", "_cut.jpg")
    # img.save("demo_picture/demo_cut.jpg")
    # img.show()
    output_size = (256, 256)
    input_size = img.size
    # print(input_size)
    img = img.resize(output_size, Image.BICUBIC)
    img = transformer.ToTensor()(img)
    print(img.shape)
    img = transformer.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    test_pic = img.unsqueeze(0).to(device)
    # img = (img + 1) / 2.0 * 255.0
    # img = img.detach().cpu().float().numpy()
    # img = img.transpose((1, 2, 0))
    # picture_pic = Image.fromarray(img.astype(np.uint8))
    # picture_pic.save("test_picture/goose_resize.jpg")
    test_generator = Generator().to(device)
    tmp_path = fig_path.split('/')[1].split('.')[0]
    final_binary = np.zeros((input_size[1], input_size[0]))
    for i in range(1, 10):
        test_generator.load_state_dict(torch.load("../ExtractContour/Models/generator_%d00.pth" % i))
        pic_result = test_generator(test_pic)
        img = pic_result.detach()[0][0].cpu().float().numpy()
        img = (img + 1) / 2.0 * 255.0
        img_pil = img.astype(np.uint8)
        img_pil = cv2.resize(img_pil, input_size, cv2.INTER_CUBIC)
        # print(img_pil.shape)
        # image_pil = Image.fromarray(img.astype(np.uint8))
        # image_pil = image_pil.resize(input_size, Image.BICUBIC)
        # print(image_pil.size)
        # cv2.imshow("new", img_pil)
        # cv2.waitKey(0)
        cv2.imwrite("test_demo/tmp_pics/" + tmp_path + "_tmp%d.jpg" % i, img_pil)
        final_binary += (255 - img_pil)

    # cv2.imshow("new", final_binary)
    # cv2.waitKey(0)


if __name__ == '__main__':
    fig_path, point1, point2 = readPoints()
    print(point1, point2)
    getContour(fig_path, point1, point2)

