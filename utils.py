import torch
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

###################### 画loss函数图象 ########################################
def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)# 定义大小
    plt.xlabel(x_label) #x 轴标签
    plt.ylabel(y_label) #y 轴标签
    plt.semilogy(x_vals, y_vals)# 画出 x,y成对的值 (x,y),可由列表给出数据
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

########################## end ###############################################


def relu(x):
    """激活函数 ReLU()"""
    return torch.max(input=x, other=torch.tensor(0.0))

def softmax(x):
    """softmax()函数 分类中使用求概率"""
    x_exp = x.exp()
    exp_sum = x_exp.sum(dim = 1,keepdims= True)
    return  x_exp / exp_sum

def cross_entropy(y_that,y):
    """cross_entropy 计算交叉熵w误差"""
    return -torch.log(y_that.gather(dim = 1,index = y.view(-1,1)))

'''square_loss 误差'''
def square_loss(y_that,y):
    return (y_that - y)**2


def load_MNINST():
    '''数据存储路径'''
    path_train_imgs = '../data/MNIST_figure_user/train_images.npy'
    path_train_labers = '../data/MNIST_figure_user/train_labels.npy'
    path_test_imgs = '../data/MNIST_figure_user/t10k_images.npy'
    path_test_labers = '../data/MNIST_figure_user/t10k_labels.npy'
    '''训练集数据'''
    imgs = torch.from_numpy(np.load(path_train_imgs))
    labels = torch.from_numpy(np.load(path_train_labers)).long()
    '''测试集数据'''
    test_imgs = torch.from_numpy(np.load(path_test_imgs))
    test_labels = torch.from_numpy(np.load(path_test_labers)).long()
    return [imgs,labels,test_imgs,test_labels]


'''模型评估函数 分类问题中使用'''
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0,0
    with torch.no_grad():
        for x,y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模型时关闭dropout
                acc_sum += (net(x).argmax(dim=1)==y).float().sum().item()
                net.train() # 改为训练模式
            else:           #自定义模型
                if 'is_train' in net.__code__.co_varnames:
                    #将 is_train 改为 false
                    acc_sum += (net(x,is_train = False).argmax(dim=1)==y).float().sum().item()
                else:
                    acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

'''梯度下降算法'''
'''BGD (Batch Gradient Descent), SGD(Stochastic), MBGD(Mini-Batch)'''
def sgd(params, lr,batch_size):
    for param in params:
        param.data -= param.grad * lr / batch_size


'''mnist data 测试'''
def train_ch3(net, train_iter, test_iter,all_train,all_test,loss, num_epochs, batch_size,
              params, lr=None, optimizer=None):
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        n, train_acc_sum, train_l_sum = 0, 0.0, 0.0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到 使用nn.Model时用到
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        train_ls.append(loss(net(all_train[0]), all_train[1]).sum().item() / all_train[1].shape[0])
        test_ls.append(loss(net(all_test[0]), all_test[1]).sum().item() / all_test[1].shape[0])
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, n %d'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, n))
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])


def train_ch5(net, train_iter, test_iter,loss, num_epochs, optimizer):
    for epoch in range(num_epochs):
        n, train_acc_sum, train_l_sum, batch_count = 0, 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, n %d'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, n))


'''函数参数意义：将以 p 的概率丢弃 x 中的元素'''
def dropout(x, p):
    x = x.float()
    assert 0<= p <=1
    e_p = 1 - p
    if e_p == 0:
        return torch.zeros_like(x)
    mask = (torch.rand(x.shape) < e_p).float()
    return mask * x / e_p



#K交叉验证 是用来模型选择，以的一种方法
#k交叉验证 数据准备
# X 代表数据features， y 代表数据labels
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k>1

    #每一小块的大小
    fold_size = X.shape[0] // k
    x_train, y_train =  None, None

    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        x_part, y_part = X[idx, :], y[idx]

        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), dim = 0)
            y_train = torch.cat((y_train, y_part), dim = 0)

    return x_train, y_train, x_valid, y_valid



