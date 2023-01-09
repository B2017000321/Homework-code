
from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import base_model
from model import BasicBlock
from model import Bottleneck
from model import ResNet
from model import ResNet_AE
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np


def main(config):
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # transform_test = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.TenCrop(40),
    #     transforms.Lambda(lambda crops: torch.stack(
    #         [transforms.ToTensor()(crop) for crop in crops])),
    #     transforms.Lambda(lambda tensors: torch.stack(
    #         [transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
    # ])

    # transform_train = transforms.Compose([
    #         transforms.Grayscale(),
    #         transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
    #         transforms.RandomApply([transforms.ColorJitter(
    #             brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
    #         transforms.RandomApply(
    #             [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
    #         transforms.FiveCrop(40),
    #         transforms.Lambda(lambda crops: torch.stack(
    #             [transforms.ToTensor()(crop) for crop in crops])),
    #         transforms.Lambda(lambda tensors: torch.stack(
    #             [transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
    #         transforms.Lambda(lambda tensors: torch.stack(
    #             [transforms.RandomErasing()(t) for t in tensors])),
    #     ])

    train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                              pin_memory=True, num_workers=8)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=True,
                            num_workers=8)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False,
                             pin_memory=True, num_workers=8)

    model = base_model(class_num=config.class_num)

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.1, weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1,
                                                     last_epoch=-1)
    # model = ResNet(BasicBlock, [2, 2, 2, 2])
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.1, weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
    #         optimizer, T_max=10)

    creiteron = torch.nn.CrossEntropyLoss()

    train_numbers, train_losses, train_accuracy = train(config, train_loader, model, optimizer, scheduler,
                                                        creiteron)

    val_accuracy = test(val_loader, model)
    test_accuracy = test(test_loader, model)
    print('===========================')
    print("val accuracy:{}%".format(val_accuracy * 100))
    print("test accuracy:{}%".format(test_accuracy * 100))
    plt.plot(train_losses)
    plt.show()

    x_0 = PCA(config, train_loader, model, 0)
    x_1 = PCA(config, train_loader, model, 1)
    x_2 = PCA(config, train_loader, model, 2)
    x_3 = PCA(config, train_loader, model, 3)
    x_4 = PCA(config, train_loader, model, 4)
    x_5 = PCA(config, train_loader, model, 5)
    x_6 = PCA(config, train_loader, model, 6)
    plt.plot(x_0[:, 0], x_0[:, 1], 'o', color='red', markersize=1, label='angry')
    plt.plot(x_1[:, 0], x_1[:, 1], 'o', color='black', markersize=1, label='disgust')
    plt.plot(x_2[:, 0], x_2[:, 1], 'o', color='orange', markersize=1, label='fear')
    plt.plot(x_3[:, 0], x_3[:, 1], 'o', color='yellow', markersize=1, label='happy')
    plt.plot(x_4[:, 0], x_4[:, 1], 'o', color='green', markersize=1, label='neutral')
    plt.plot(x_5[:, 0], x_5[:, 1], 'o', color='blue', markersize=1, label='sad')
    plt.plot(x_6[:, 0], x_6[:, 1], 'o', color='purple', markersize=1, label='surprise')
    plt.legend()
    plt.grid()
    plt.show()


def PCA(config, data_loader, model, type):
    model.train()
    model.cuda()
    DATA = []

    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.cuda(), label.cuda()
        output = model(data)
        out = output.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        label = np.array(label)
        for i in range(len(label)):
            if label[i] == type:
                DATA.append(out[i])
    DATA = np.array(DATA)
    n_samples, n_features = DATA.shape
    mean = np.zeros(7)
    norm = DATA
    for i in range(n_features):
        mean[i] = np.array(np.mean(DATA[i]))
        norm[i] = np.array(DATA[i] - mean[i])

    scatter = np.dot(np.transpose(norm), norm)
    eig_val, eig_vec = np.linalg.eig(scatter)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    eig_val = np.abs(eig_val)
    index = eig_val.argsort()
    feature = [[0] * 7] * 2

    for i in range(2):
        feature[i] = eig_vec[index[6 - i]]

    dim_re_data = np.dot(norm, np.transpose(feature))
    return dim_re_data


def train(config, data_loader, model, optimizer, scheduler, creiteron):
    model.train()
    model.cuda()
    train_losses = []
    train_numbers = []
    train_accuracy = []
    counter = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = creiteron(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            accuracy = ((label == output.argmax(dim=1)).sum() * 1.0).item() / output.shape[0]
            if batch_idx % 20 == 0:
                print('Train Epoch: {} / {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch, config.epochs, batch_idx * len(data), len(data_loader.dataset),
                                          100. * batch_idx / len(data_loader), loss.item(),
                    accuracy))
                train_losses.append(loss.item())
                train_numbers.append(counter)
                train_accuracy.append(accuracy)
        scheduler.step()
        torch.save(model.state_dict(), './model.pth')
    return train_numbers, train_losses, train_accuracy


def test(data_loader, model):
    model.eval()
    model.cuda()
    correct = 0
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum()
    accuracy = (correct * 1.0).item() / len(data_loader.dataset)
    return accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=0.075)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 25])

    config = parser.parse_args()
    main(config)
