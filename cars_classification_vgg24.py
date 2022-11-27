

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import check_accuracy, load_checkpoint, save_checkpoint, make_prediction
import config
from dataset import MyImageFolder
import json

train_ds = MyImageFolder(
    root_dir="cars_classification_dataset/train/", transform=config.train_transforms)
val_ds = MyImageFolder(
    root_dir="cars_classification_dataset/valid/", transform=config.val_transforms)
train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                          num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=True)


# Reference taken for the code
# Architecture Structure referenced from and modified for ourselves
# Programmed by Aladdin Persson(2021)
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_vgg_implementation.py
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", ],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M", ],
    "VGG24": [64, 64, 64, "M", 128, 128, 128, "M", 256, 256, 256, 256, 256, "M", 512, 512, 512, 512, 512, "M", 512, 512, 512, 512, 512, "M", ],
}

device = 'cuda'


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG24"])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    i = 0
    loss = 0
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward

        scores = model(data)
        loss = loss_fn(scores, targets.to(torch.int64))
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    model.train()
    return loss, model


loss_fn = nn.CrossEntropyLoss()

model = VGG_net(in_channels=3, num_classes=196).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()


data_training = {}
for epoch in range(config.NUM_EPOCHS):
    loss, model = train_fn(train_loader, model, optimizer,
                           loss_fn, scaler, config.DEVICE)
    print('loss is : ', loss)

    # print(model)
    checkpoint = {'state_dict': model.state_dict(
    ), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)
    data_training[f'epoch_{epoch+1}'] = {'loss': float(loss)}
    with open('training.json', 'w') as j_file:
        json.dump(data_training, j_file, indent=4)


with open('training.json', 'rb') as file:
    js = json.load(file)

key_list = []
ls = list(js.keys())
for key in ls:

    key_list.append(js[key]['loss'])
print(key_list)
key_list = sorted(key_list, reverse=True)


# Plpt
plt.plot(key_list, 'g', label='Training loss')
plt.title('Complete Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('vgg24_cars_training.png')
plt.show()
