import matplotlib as plt
import matplotlib.pyplot
import numpy as np
import torchvision
import math
import PIL
import cv2 as cv


def imshow(img, img2):
    # img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    fig = plt.pyplot.figure()
    fig.add_subplot(1, 2, 1)
    plt.pyplot.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.pyplot.imshow(img2.permute(1,2,0))
    fig.savefig('result.png')


def test_transforms(path, transform):
    img_ref = PIL.Image.open(path)
    img_transform = transform(PIL.Image.open(path))
    # imshow(torchvision.utils.make_grid([img_ref, img_transform]))
    imshow(img_ref, img_transform)


def image_preprocessing(path):
    print(path)
    # img = np.array(PIL.Image.open(path))
    #
    # for x, array in enumerate(img):  # for every p
    #     for y, pixel in enumerate(array):  # for every p
    #         if not 0 < pixel[0] < 200:
    #             tmp = 0
    #             img[x, y] = [tmp, tmp, tmp]
    #
    # PIL.Image.fromarray(img).save('result.png')
    # path = "result.png"

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(img, 30, 80)
    cv.imwrite("result.png", edges)
