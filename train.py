import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Sequential
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def reshape_and_rotate(image):
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


def normalize_dataset(df):
    y = df.loc[:, 0].values
    x = df.loc[:, 1:]
    x = np.apply_along_axis(reshape_and_rotate, 1, x.values)
    x = x.astype('float32') / 255

    cond = np.isin(y, allowed_chars)
    x = x[cond]
    y = y[cond]

    y = np.array(list(map(lambda c: allowed_chars.index(c), y)))
    return x, y


def get_model():
    n_classes = len(allowed_chars)
    model = Sequential(
        nn.Conv2d(1, 32, (5,5), padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 48, (5,5), padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1200, 256),
        nn.ReLU(),
        nn.Linear(256, 84),
        nn.ReLU(),
        nn.Linear(84, n_classes)
    )
    return model


def train(model, x_train, y_train, x_val, y_val):
    
    train_loader = DataLoader(TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train)), batch_size=32)
    val_loader = DataLoader(TensorDataset(torch.Tensor(x_val), torch.LongTensor(y_val)), batch_size=32)

    lr = 1e-3
    optimizer = Adam(model.parameters(), lr=lr)
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop((28, 28), scale=(.8, 1)),
                    transforms.ToTensor()
                ])

    for epoch in range(3):
        model.train()
        for inputs, target in tqdm(train_loader):
            optimizer.zero_grad()
            # unpack batch
            inputs = inputs.unsqueeze(1)
            inputs = list(inputs)
            # transform each image and return to batch
            inputs = torch.stack(tuple(map(transform, inputs)), axis=0)

            loss = F.cross_entropy(model(inputs), target)
            loss.backward()
            optimizer.step()

        running_acc = 0.
        model.eval()
        for inputs, target in tqdm(val_loader):
            running_acc += (torch.argmax(model(inputs.unsqueeze(1)), dim=-1) == target).float().mean().item()
        print(f"Val Accuracy:  {running_acc / len(val_loader):.2f}")
