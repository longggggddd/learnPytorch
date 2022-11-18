# -*- coding:utf-8-*-
# @Author:ZHN
# @Date:2022.06.06

import numpy as np
from sklearn.datasets import load_iris, load_digits
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class SetSeed:
    def same_seeds(self, seed):
        torch.manual_seed(seed)  # 固定随机种子（CPU）
        if torch.cuda.is_available():  # 固定随机种子（GPU)
            torch.cuda.manual_seed(seed)  # 为当前GPU设置
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
        np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
        torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
        torch.backends.cudnn.deterministic = True  # 固定网络结构


class ProcessData:
    def __init__(self):
        self.dataX, self.dataY = load_digits(return_X_y=True)
        self.dataX, self.dataY = np.float32(self.dataX), torch.Tensor(self.dataY).long()

    def dataToLoader(self):
        self.trainDataX, self.midDataX, self.trainDataY, self.midDataY = train_test_split(
            self.dataX,
            self.dataY,
            stratify=self.dataY,
            random_state=1,
            test_size=0.2
        )
        self.testDataX, self.devDataX, self.testDataY, self.devDataY = train_test_split(
            self.midDataX,
            self.midDataY,
            stratify=self.midDataY,
            random_state=1,
            test_size=0.5
        )
        trainDataForLoader = list(zip(self.trainDataX, self.trainDataY))
        devDataForLoader = list(zip(self.devDataX, self.devDataY))
        testDataForLoader = list(zip(self.testDataX, self.testDataY))

        self.trainLoader = torch.utils.data.DataLoader(dataset=trainDataForLoader,
                                                       batch_size=8,
                                                       # collate_fn=False,
                                                       shuffle=True,
                                                       drop_last=True)
        self.devLoader = torch.utils.data.DataLoader(dataset=devDataForLoader,
                                                     batch_size=8,
                                                     # collate_fn=False,
                                                     shuffle=True,
                                                     drop_last=True)
        self.testLoader = torch.utils.data.DataLoader(dataset=testDataForLoader,
                                                      batch_size=8,
                                                      # collate_fn=False,
                                                      shuffle=True,
                                                      drop_last=True)
        return self.trainLoader, self.devLoader, self.testLoader


class FNN(torch.nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc = torch.nn.Linear(64, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.tanh = torch.nn.Tanh()
        # self.bn = torch.nn.BatchNorm1d()

    def forward(self, x):
        # out = self.bn(x)
        # print(x)
        out = self.fc(x)
        out = self.dropout(out)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.dropout(out)
        logits = self.tanh(out)
        # print(logits)
        # prob = self.softmax(logits)
        return logits


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 1))

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)  # (B, L, H) -> (B , L, 1)
        weights = F.softmax(energy, dim=1)
        outputs = encoder_outputs * weights  # (B, L, H)
        return outputs


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.fc = torch.nn.Linear(200, 10)
        self.fc2 = torch.nn.Linear(256, 10)
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.tanh = torch.nn.Tanh()
        self.bn = torch.nn.BatchNorm1d(64)
        self.lstm = torch.nn.LSTM(64, 100, 2, bidirectional=True, batch_first=True, dropout=0.5)

    def forward(self, x):
        # out = self.bn(x)
        x = x.unsqueeze(dim=1)
        out, _ = self.lstm(x)
        out = self.tanh(out)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.tanh(out)
        # out = self.fc2(out)
        # out = self.tanh(out)
        # logits = self.dropout(out)
        # prob = self.softmax(logits)
        prob = out.squeeze(dim=1)
        return prob


class RNN_ATT(torch.nn.Module):
    def __init__(self):
        super(RNN_ATT, self).__init__()
        self.fc = torch.nn.Linear(200, 10)
        self.fc2 = torch.nn.Linear(256, 10)
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.tanh = torch.nn.Tanh()
        self.bn = torch.nn.BatchNorm1d(64)
        self.lstm = torch.nn.LSTM(64, 100, 2, bidirectional=True, batch_first=True, dropout=0.5)
        self.attention = SelfAttention(100 * 2)

    def forward(self, x):
        # out = self.bn(x)
        x = x.unsqueeze(dim=1)
        out, _ = self.lstm(x)
        out = self.tanh(out)
        out = self.attention(out)
        out = self.tanh(out)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.tanh(out)
        # out = self.fc2(out)
        # out = self.tanh(out)
        # logits = self.dropout(out)
        # prob = self.softmax(logits)
        prob = out.squeeze(dim=1)
        return prob


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc = torch.nn.Linear(64, 256)
        self.fc2 = torch.nn.Linear(64, 10)
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.tanh = torch.nn.Tanh()
        self.conv1d = torch.nn.Conv1d(64, 10, kernel_size=1)
        self.conv1d_t = torch.nn.Conv1d(8, 8, kernel_size=1)
        # self.maxpool1d = torch.nn.MaxPool1d(kernel_size=4, stride=None, padding=0, dilation=1,
        #                                     return_indices=False, ceil_mode=False)
        self.flatten = torch.nn.Flatten(start_dim=0, end_dim=1)
        # self.bn = torch.nn.BatchNorm1d()

    def forward(self, x):
        x = x.unsqueeze(dim=2)
        print(x.size())
        out = self.conv1d(x)
        out = self.flatten(out)
        prob = self.softmax(out)
        # ------------------------------------
        # x = x.unsqueeze(dim=0)
        # out = self.conv1d_t(x)
        # # out = self.maxpool1d(out)
        # out = self.flatten(out)
        # out = self.fc2(out)
        # # print(out)
        # # print(out.size())
        # prob = self.softmax(out)
        return prob


class CNN2(torch.nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=2)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=2)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(720, 50)
        self.fc2 = torch.nn.Linear(490, 10)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = np.array(x)
        x = x.reshape(8, 1, 8, 8)
        x = np.float32(x)
        x = torch.tensor(x)
        x = self.conv1(x)
        # x = F.dropout2d(x)
        x = F.max_pool2d(x, kernel_size=1)
        x = F.relu(x)
        x = self.flatten(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Main(ProcessData, SetSeed):
    def __init__(self, model):
        super(Main, self).__init__()
        self.model = model
        self.trainLoader = self.dataToLoader()[0]
        self.devLoader = self.dataToLoader()[1]
        self.testLoader = self.dataToLoader()[2]
        self.optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # 使用Adam优化器
        # 设置学习率
        self.schedule = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=len(self.trainLoader),
                                                        num_training_steps=10 * len(self.testLoader))
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.same_seeds(20)
        bestAcc = 0.0
        for i in range(200):
            self.model.train()  # 开始训练
            print("***************Running training epoch{}************".format(i + 1))
            train_loss_sum = 0.0
            for idx, (X, y) in enumerate(self.trainLoader):
                yPred = self.model(X)
                # print(yPred)
                # print(y)
                loss = self.criterion(yPred, y)
                # loss=self.criterion()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                train_loss_sum += loss.item()
            devAcc = self.__evaluate(self.model, self.devLoader)
            print(devAcc)
            if devAcc > bestAcc:
                bestAcc = devAcc
                torch.save(self.model.state_dict(), "checkpoint/best_model.pth")
        testAcc = self.__predict(self.model, self.testLoader)
        torch.save(self.model.state_dict(), "checkpoint/best_model2.pth")
        print(testAcc)

    def __evaluate(self, model, data_loader):
        model.eval()  # 防止模型训练改变权值
        devTrue, devPred = [], []
        with torch.no_grad():  # 计算的结构在计算图中,可以进行梯度反转等操作
            for idx, (X, y) in enumerate(data_loader):  # 得到的y要转换一下数据格式
                yPred = model(X)  # 此时得到的是概率矩阵
                yPred = torch.argmax(yPred, dim=1).detach().cpu().numpy().tolist()  # 将概率矩阵转换成标签并变成list类型
                devPred.extend(yPred)  # 将标签值放入列表
                devTrue.extend(y.squeeze().cpu().numpy().tolist())  # 将真实标签转换成list放在列表中
        return accuracy_score(devTrue, devPred)

    def __predict(self, model, data_loader):
        model.eval()  # 防止模型训练改变权值
        testTrue, testPred = [], []
        with torch.no_grad():  # 计算的结构在计算图中,可以进行梯度反转等操作
            for idx, (X, y) in enumerate(data_loader):  # 得到的y要转换一下数据格式
                yPred = model(X)  # 此时得到的是概率矩阵
                yPred = torch.argmax(yPred, dim=1).detach().cpu().numpy().tolist()  # 将概率矩阵转换成标签并变成list类型
                testPred.extend(yPred)  # 将标签值放入列表
                testTrue.extend(y.squeeze().cpu().numpy().tolist())  # 将真实标签转换成list放在列表中
        return classification_report(testTrue, testPred)


class Test(SetSeed):
    def __init__(self):
        self.path = 'checkpoint/best_model2.pth'

    def test(self, model, data):
        self.same_seeds(1)
        data = torch.Tensor(data)
        model = model
        model.eval()
        model.load_state_dict(torch.load(self.path))
        y_pred_prob = model(data)  # 此时得到的是概率矩阵
        y_pred = torch.argmax(y_pred_prob, dim=1).detach().cpu().numpy().tolist()
        # print(y_pred_prob)
        print('预测值：', y_pred[0])


if __name__ == '__main__':
    model = FNN()

    main = Main(model)
    main.train()

    index = 348
    dataX, _ = load_digits(return_X_y=True)
    dataMidList = []
    data = dataX[index, :].reshape(8, -1)
    print('真实数据：', _[index])
    import matplotlib.pyplot as plt

    plt.imshow(data, cmap='gray', interpolation='none')
    plt.show()
    data = dataX[index, :]
    dataMidList.append(data)
    for _ in range(7):
        vecMid = [0 for _ in range(64)]
        dataMidList.append(vecMid)
    # print(dataMidList)
    data = dataX[1, :].reshape(1, -1)
    # data = np.array(dataMidList).reshape(8, 1, 8, 8)
    # print(data)
    test = Test()
    test.test(model, data)

    data = torch.rand([3, 3])
    # print(data)
    data1 = data.unsqueeze(dim=0)
    print(data1)
