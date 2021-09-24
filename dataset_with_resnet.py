import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet50

from CenterLoss import CenterLoss

warnings.filterwarnings('ignore')


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(2048, 2)  # 加上一层参数修改好的全连接层

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x, F.log_softmax(x, dim=1)


def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#0000ff']
    plt.clf()
    for i in range(2):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1'], loc='upper right')
    plt.xlim(xmin=-8, xmax=8)
    plt.ylim(ymin=-8, ymax=8)
    plt.text(-7.8, 7.3, "epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    return TP, FP, TN, FN


def visualize_test(loss, precision, epoch):
    plt.ion()
    c = {'loss': '#ff0000', 'precision': '#0000ff'}
    plt.clf()
    plt.plot(epoch, loss, '-', c=c['loss'])
    plt.plot(epoch, precision, '--', c=c['precision'])
    plt.xlim(xmin=0, xmax=51)
    plt.ylim(ymin=0, ymax=1.5)
    plt.legend(['loss', 'precision'], loc='upper right')
    plt.savefig('./images/model/epoch=%d.jpg' % epoch[-1])
    plt.pause(0.001)


def count_precision(pred, target):
    global right_num
    global epoch_len
    for p, t in zip(pred, target):
        if p == 1 and t == 1:
            right_num += 1
        if p == 1:
            epoch_len += 1


def train(epoch_num):
    global right_num
    global epoch_len
    model.train()
    for epoch in range(epoch_num):
        epoch = epoch + 1
        right_num = 0
        epoch_len = 0
        print("Training... Epoch = %d" % epoch)
        pos_loader = []
        idx_loader = []
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            pos, pred1 = model(data)
            loss = nllloss(pred1, target) + loss_weight * centerloss(target, pos)
            count_precision(pred1.argmax(dim=1), target)

            optimizer4nn.zero_grad()
            optimzer4center.zero_grad()

            loss.backward()

            optimizer4nn.step()
            optimzer4center.step()

            pos_loader.append(pos)
            idx_loader.append((target))

        precision = right_num / epoch_len
        feat = torch.cat(pos_loader, 0)
        labels = torch.cat(idx_loader, 0)
        visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        print('epoch: %d, loss: %f, precision: %f' % (epoch, loss.item(), precision))
        sheduler.step()
        torch.save(model, './models/model_epoch=%d.pkl' % epoch)


def test(path):
    global right_num
    global epoch_len
    pred = []
    true = []
    model = torch.load(path)
    model.eval()
    right_num = 0
    epoch_len = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        pos, pred1 = model(data)
        loss = nllloss(pred1, target)
        count_precision(pred1.argmax(dim=1), target)
        pred.append(pred1.argmax(dim=1).item())
        true.append(target.item())
        print('\rcorrect rate: %d/%d' % (right_num, epoch_len), flush=True, end='')
    precision = right_num / epoch_len
    print('\ntest loss: %f, test precision: %f' % (loss.item(), precision))
    TP, FP, TN, FN = perf_measure(true, pred)
    confusion_matrix = np.array([[TP, FP], [FN, TN]])
    print(confusion_matrix)
    return loss.item(), precision


def get_epoch(model_path):
    return int(model_path.split('=')[1].split('.')[0])


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataset
    trainset = datasets.ImageFolder(r'./dataset/train', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 100)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    train_loader = DataLoader(trainset, batch_size=8,
                              shuffle=True, num_workers=4)

    testset = datasets.ImageFolder(r'./dataset/test', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 100)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    test_loader = DataLoader(testset, batch_size=1,
                             shuffle=True, num_workers=4)

    # Model
    resnet = resnet50(pretrained=True).to(device)
    model = Net(resnet).to(device)

    # NLLLoss
    nllloss = nn.NLLLoss().to(device)  # CrossEntropyLoss = log_softmax + NLLLoss
    # CenterLoss
    loss_weight = 0.8
    centerloss = CenterLoss(2, 2).to(device)

    # optimzer4nn
    # optimizer4nn = optim.SGD(model.parameters(),lr=1e-5,momentum=0.9, weight_decay=0.0005)
    optimizer4nn = optim.Adam(model.parameters(), lr=5e-6)
    sheduler = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.8)

    # optimzer4center
    optimzer4center = optim.SGD(centerloss.parameters(), lr=0.5)

    right_num = 0
    epoch_len = 0
    epoch_num = 50

    test_loss = []
    test_precision = []
    test_epoch = []
    train(epoch_num)
    models = glob('./models/model_epoch=*.pkl')
    models = sorted(models, key=get_epoch)
    for model_path in models:
        loss, precision = test(model_path)
        test_loss.append(loss)
        test_precision.append(precision)
        test_epoch.append(int(model_path.split('=')[1].split('.')[0]))
        visualize_test(test_loss, test_precision, test_epoch)
