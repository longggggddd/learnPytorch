# -*-coding:utf-8-*-
# @Author:KK
# @Date:1
import numpy as np
from sklearn.datasets import load_iris, load_digits
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
import warnings

warnings.filterwarnings('ignore')

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

# dataX, dataY = load_iris(return_X_y=True)
dataX, dataY = load_digits(return_X_y=True)
dataX, dataY = np.float32(dataX), torch.Tensor(dataY).long()

trainDataX, midDataX, trainDataY, midDataY = train_test_split(
    dataX,
    dataY,
    stratify=dataY,
    random_state=1,
    test_size=0.2
)
testDataX, devDataX, testDataY, devDataY = train_test_split(
    midDataX,
    midDataY,
    stratify=midDataY,
    random_state=1,
    test_size=0.5
)
trainDataForLoader = list(zip(trainDataX, trainDataY))
devDataForLoader = list(zip(devDataX, devDataY))
testDataForLoader = list(zip(testDataX, testDataY))

trainLoader = torch.utils.data.DataLoader(dataset=trainDataForLoader,
                                          batch_size=8,
                                          # collate_fn=False,
                                          shuffle=True,
                                          drop_last=True)
devLoader = torch.utils.data.DataLoader(dataset=devDataForLoader,
                                        batch_size=8,
                                        # collate_fn=False,
                                        shuffle=True,
                                        drop_last=True)
testLoader = torch.utils.data.DataLoader(dataset=testDataForLoader,
                                         batch_size=8,
                                         # collate_fn=False,
                                         shuffle=True,
                                         drop_last=True)


class FNN(torch.nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc = torch.nn.Linear(64, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.tanh = torch.nn.Tanh()
        self.bn = torch.nn.BatchNorm1d(64)


    def forward(self, x):
        out = self.bn(x)
        out = self.fc(out)
        out = self.dropout(out)
        # out = self.tanh(out)
        out = self.fc2(out)
        logits = self.dropout(out)
        prob = self.softmax(logits)
        return prob


model = FNN()
print(model)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # 使用Adam优化器
# 设置学习率
schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(trainLoader),
                                           num_training_steps=10 * len(testLoader))
criterion = torch.nn.CrossEntropyLoss()


def evaluate(model, data_loader):
    model.eval()  # 防止模型训练改变权值
    val_true, val_pred = [], []
    with torch.no_grad():  # 计算的结构在计算图中,可以进行梯度反转等操作
        for idx, (X, y) in enumerate(data_loader):  # 得到的y要转换一下数据格式
            y_pred = model(X)  # 此时得到的是概率矩阵
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()  # 将概率矩阵转换成标签并变成list类型
            val_pred.extend(y_pred)  # 将标签值放入列表
            val_true.extend(y.squeeze().cpu().numpy().tolist())  # 将真实标签转换成list放在列表中
    return accuracy_score(val_true, val_pred)


def predict(model, data_loader):
    model.eval()  # 防止模型训练改变权值
    val_true, val_pred = [], []
    with torch.no_grad():  # 计算的结构在计算图中,可以进行梯度反转等操作
        for idx, (X, y) in enumerate(data_loader):  # 得到的y要转换一下数据格式
            y_pred = model(X)  # 此时得到的是概率矩阵
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()  # 将概率矩阵转换成标签并变成list类型
            val_pred.extend(y_pred)  # 将标签值放入列表
            val_true.extend(y.squeeze().cpu().numpy().tolist())  # 将真实标签转换成list放在列表中
    return classification_report(val_true, val_pred)


def train(model, data_loader):
    bestAcc = 0.0
    for i in range(200):
        model.train()  # 开始训练
        print("***************Running training epoch{}************".format(i + 1))
        train_loss_sum = 0.0
        for idx, (X, y) in enumerate(data_loader):
            yPred = model(X)
            loss = criterion(yPred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            train_loss_sum += loss.item()
        devAcc = evaluate(model, devLoader)
        print(devAcc)
        if devAcc > bestAcc:
            bestAcc = devAcc
            torch.save(model.state_dict(), "checkpoint/best_model.pth")
    testAcc = predict(model, testLoader)
    print(testAcc)


if __name__ == '__main__':
    train(model, trainLoader)

    a = dataX[1, :].reshape(1, -1)
    a = torch.Tensor(a)
    model = FNN()
    model.load_state_dict(torch.load("checkpoint/best_model.pth"))
    y_pred_prob = model(a)  # 此时得到的是概率矩阵
    y_pred = torch.argmax(y_pred_prob, dim=1).detach().cpu().numpy().tolist()
    print(y_pred_prob)
    print(y_pred)
