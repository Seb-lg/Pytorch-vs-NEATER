import torch
import os
import shutil
import logging
from os import listdir
from os.path import exists

import torchvision.datasets

from CharacterDataset import CharacterDataset, create_and_test_dataloader, create_dataloader
from torchvision import transforms

from TorchModel import train_model_save
from Neater.Neater import train_neater


def setDS(test_path, train_path):
    if exists(os.path.join(os.getcwd(), 'data/train')) or \
            exists(os.path.join(os.getcwd(), 'data/test')):
        logging.info('Data already set')
        return
    os.mkdir('data/train')
    os.mkdir('data/test')
    for i in range(1, 29):
        os.mkdir('data/test/' + str(i))
        os.mkdir('data/train/' + str(i))

    test_images_path = listdir(test_path)
    for image_path in test_images_path:
        path_split = image_path.split('.')[0].split('_')
        a = test_path + '/' + image_path
        b = 'data/test/'+str(path_split[3])+'/'+str(path_split[1]+'.png')
        shutil.copy(a, b)

    train_images_path = listdir(train_path)
    for image_path in train_images_path:
        path_split = image_path.split('.')[0].split('_')
        a = train_path + '/' + image_path
        b = 'data/train/'+str(path_split[3])+'/'+str(path_split[1]+'.png')
        shutil.copy(a, b)
    exit(84)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    setDS('data/Test Images 3360x32x32/test', 'data/Train Images 13440x32x32/train')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
    ])

    # train_dataset = CharacterDataset('data/csvTrainImages 13440x1024.csv', 'data/csvTrainLabel 13440x1.csv', transform)
    # test_dataset = CharacterDataset('data/csvTestImages 3360x1024.csv', 'data/csvTestLabel 3360x1.csv', transform)
    train_dataset = torchvision.datasets.ImageFolder('data/train', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder('data/test', transform=transform)
    train_dataloader = create_dataloader(train_dataset)
    test_dataloader = create_dataloader(test_dataset)

    # train_model_save(train_dataloader, test_dataloader, 20, 'torch_model_arab.cwd')

    train_neater(train_dataloader, test_dataloader)
    