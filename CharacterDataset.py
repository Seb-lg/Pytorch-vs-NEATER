import torch
import matplotlib
import matplotlib.pyplot
from torch.utils.data import Dataset, DataLoader


class CharacterDataset(Dataset):
    def __init__(self, image_csv_path, label_csv_path, transform=None):
        self.img = []
        self.label = []
        self.transform = transform

        with open(image_csv_path) as f:
            for line in f:
                tmp = [int(pxl) for pxl in line.replace('\n', '').split(',')]
                tmp = torch.tensor(tmp).view(32, 32)
                self.img.append(tmp)

        with open(label_csv_path) as f:
            for line in f:
                self.label.append(torch.tensor(int(line)).int())

    def __getitem__(self, item):
        if self.transform:
            return self.transform(self.img[item].unsqueeze(0)), self.label[item]
        return self.img[item].unsqueeze(0), self.label[item]

    def __len__(self):
        return len(self.label)


def create_dataloader(dataset):
    return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)


def create_and_test_dataloader(dataset):
    dataloader = create_dataloader(dataset)
    img, label = next(iter(dataloader))
    matplotlib.pyplot.imshow(img[0].view(32, 32))
    matplotlib.pyplot.xlabel('Label: ' + str(label[0].item()))
    matplotlib.pyplot.show()
    return dataloader


if __name__ == '__main__':
    dataset = CharacterDataset('data/csvTestImages 3360x1024.csv', 'data/csvTestLabel 3360x1.csv', None)
    assert(len(dataset) > 0)
    create_and_test_dataloader(dataset)
