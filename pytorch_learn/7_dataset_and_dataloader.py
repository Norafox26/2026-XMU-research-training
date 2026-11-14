import torch
import numpy as np
from torch.utils.data import Dataset # Dataset是一个抽象类，需要被继承
from torch.utils.data import DataLoader # DataLoader可以被实例化

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32) # 习惯32位，较老显卡只支持32位
        self.len = xy.shape[0] 
        # xy.shape返回的是元组(a,b),a为行数，b为列数,(a,b)[0]相当于样本数a
        self.x_data = torch.from_numpy(xy[:,0:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index): # 实现通过索引获取数据
        return self.x_data[index], self.y_data[index] # 返回的是数据的元组

    def __len__(self): # 实现获取数据集的长度
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1) 
        self.sigmoid = torch.nn.Sigmoid() 

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__ == "__main__":
    dataset = DiabetesDataset('diabetes.csv.gz')
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=2 # 2个并行进程
        )

    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # enumerate编号，从0开始给train_loader取出的data编号
            # data是由x，y组成的元组
            # x是32*8的张量，y是32*1张量（最后一个batch按剩余sample数决定）
            inputs, labels = data
            # inputs和labels自动转换为张量
            
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()