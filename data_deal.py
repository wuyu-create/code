import os
import struct
import numpy as np
import torchvision.transforms as transforms
import imageio

# #################### 处理MINST_figure数据 ####################
#加载数据 MNIST_figure 中的数据
def load_mnist(path='MNIST_figure', kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


#加载数据 MNIST_figure 中的数据
transform1 = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5])])
def process_data(path="train"):

    img, lab = load_mnist(kind=path)

    # numpy 修改矩阵维度方法
    p = img.reshape(-1, 1, 28, 28)
    d_t = np.transpose(p, (0, 2, 3, 1))
    d_t_trans = []
    for i in range(d_t.shape[0]):
        # transforms.Compose后类型就为 Tenors了，变换成矩阵方便后面计算
        d_t_trans.append(transform1(d_t[i]).view(-1).detach().numpy())
    np.save('MNIST_figure_user/%s_images.npy'%path, np.array(d_t_trans))
    np.save('MNIST_figure_user/%s_labels.npy'%path, lab)
'''处理时 process_data()'''

# ############################ end #############################

# ############################ CIFAR-10 ########################
'''
保存训练数据
'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 保存训练数据
def save_train_cifar_10(file = 'CIFAR-10/cifar-10-batches-py/data_batch_'):
    imgs = np.empty([0, 3072])
    labels = np.empty([0,])
    for i in range(5):
        file1 = file + str(i + 1)
        dict_data = unpickle(file1)
        labels = np.insert(labels, i * 10000, np.array(dict_data[b'labels']), axis=0)
        imgs = np.insert(imgs, i * 10000, dict_data[b'data'], axis=0)
    np.save('CIFAR-10_user/cifar_train_img.npy', imgs)
    np.save('CIFAR-10_user/cifar_train_label.npy', labels)

# 保存 test数据
def save_test_cifar_10(file='CIFAR-10/cifar-10-batches-py/test_batch'):
    dict_data = unpickle(file)
    imgs = dict_data[b'data']
    labels = np.array(dict_data[b'labels'])
    np.save('CIFAR-10_user/cifar_test_img.npy', imgs)
    np.save('CIFAR-10_user/cifar_test_label.npy', labels)


'''保存图片'''
def save_image(path = 'train'):
    imgs = np.load('CIFAR-10_user/cifar_%s_img.npy' % path).astype(np.uint8)
    labels = np.load('CIFAR-10_user/cifar_%s_label.npy' % path).astype(np.long)
    imgs = imgs.reshape(-1,3,32,32)
    for i in range(labels.shape[0]):
        picName = 'E:\\cifar-%s-pic\\'%path+str(i)+'_pic_'+str(labels[i])+'.jpg'
        imageio.imwrite(picName, imgs[i].transpose(1, 2, 0))




# ######################################## end ##################################

