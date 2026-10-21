import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_dataset():
    x,y,coef = make_regression(n_samples=100, 
                               n_features=1, 
                               noise=10,
                               bias=14.5, 
                               coef=True, 
                               random_state=24)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return x, y, coef


def train(x,y,coef):
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = nn.Linear(1,1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs , loss_list= 100,[]
    for epoch in range  (epochs):
        total_loss,total_sample=0.0,0
        for train_x,train_y in dataloader:
            y_pred = model(train_x)
            loss = criterion(y_pred,train_y)
            total_loss += loss.item()
            total_sample += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(total_loss/total_sample)

        print(f'epoch:{epoch+1}, Loss: {total_loss/total_sample:.4f}')
    print(f'平均损失：{loss_list}')
    print(f'权重：{model.weight}, 偏置：{model.bias}')

    plt.figure(1)
    plt.plot(range(epochs), loss_list, color='blue', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.legend()
    

    plt.figure(2)
    x_np = x.numpy()
    y_np = y.numpy()

    plt.scatter(x_np, y_np, color='gray')
    
    # 【修复1】预测值：直接把整个 x 丢进模型，并用 detach() 剥离梯度
    y_pred_numpy = model(x).detach().numpy()
    
    # 【修复2】真实值：因为 coef 是 numpy 数组，直接用 x_np 和它做矩阵乘加
    y_true_numpy = x_np * coef + 14.5
    
    plt.plot(x_np, y_pred_numpy, color='red', label='预测值(模型拟合)')
    plt.plot(x_np, y_true_numpy, color='green', label='真实值(理论直线)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('线性回归预测结果')
    plt.grid()
    plt.legend()
    plt.show()
 



if __name__ =='__main__':
    x,y,coef=create_dataset()
    #print(f'x:{x.shape}, y:{y.shape}, coef:{coef}')
    train(x,y,coef)




