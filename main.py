import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms
from datetime import datetime


class MnistDS(Dataset):

    def __init__(self, path):
        data = []
        for line in open(path):
            line.strip()
            target_as_str, img_data_as_str_arr = line.split(',', 1)
            img_data_as_int_arr = list(map(int, img_data_as_str_arr.split(',')))
            data.append(
                (torch.tensor(img_data_as_int_arr).type(torch.FloatTensor),
                 torch.tensor(int(target_as_str)).type(torch.LongTensor))
            )

        self.data = data

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class FNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def rightness(output, target):
    score_card = []
    _, predicted = torch.max(output, 1)
    for x, y in zip(predicted, target):
        if x.item() == y.item():
            score_card.append(1)
        else:
            score_card.append(0)
    return torch.sum(torch.tensor(score_card)) / len(score_card)


if __name__ == '__main__':
    begin = datetime.now()
    trans = transforms.Compose([transforms.ToTensor()])
    train_ds = MnistDS('./data/mnist_train.csv')
    test_ds = MnistDS('./data/mnist_test.csv')

    train_dl = DataLoader(dataset=train_ds, batch_size=100, shuffle=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=100, shuffle=False)
    print('data loaded,cost:{}s'.format((datetime.now() - begin).seconds))
    net = FNN(28 ** 2, 500, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for e_inx in range(5):
        for bch_inx, (img_data, target) in enumerate(train_dl):
            optimizer.zero_grad()
            output = net(img_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print('epoch:{}, batch:{}, accuracy:{}'.format(e_inx, bch_inx, rightness(output, target)))
    pass

    rightness_scores = []
    for bch_inx, (img_data, target) in enumerate(test_dl):
        output = net(img_data)
        rightness_scores.append(rightness(output, target))
    print('avg rightness:{}, cost:{}s'.format(torch.mean(torch.tensor(rightness_scores)),
                                              (datetime.now() - begin).seconds))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
