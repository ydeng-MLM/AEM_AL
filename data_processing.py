import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from nano_generator import nano_generate, nano_simulate, toy_func, toy_initial

def normalize_np(x):
    for i in range(len(x[0])):
        x_max = np.max(x[:, i])
        x_min = np.min(x[:, i])
        x_range = (x_max - x_min)
        x[:, i] = (x[:, i] - x_min) / x_range
        x[:, i] = (x[:, i] - 0.5) / 0.5
    return x

def data_initial(n_sample, data=None):
    if data == 'toy':
        X_train, y_train = toy_initial(n_sample)
        X_test, y_test = toy_initial(n_sample)
        X_train = X_train/10
        X_test = X_test/10
    elif data == 'Nano':
        X_train, y_train = nano_generate(30, 70, num_samples=n_sample, num_layers=8)
        X_test, y_test = nano_generate(30, 70, num_samples=1000, num_layers=8)
        X_train = ((X_train - 50) / 20).astype('float32')
        X_test = ((X_test - 50) / 20).astype('float32')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

    print("The new train/validation set are")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    trainset = MetaMaterialDataSet(X_train, y_train, bool_train=True)
    testset = MetaMaterialDataSet(X_test, y_test, bool_train=False)
    # create data loaders
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
    testloader = DataLoader(testset, batch_size=1024, shuffle=True)
    return X_train, y_train, trainloader, testloader


def data_add(X_temp, y_temp, X_add, data=None):
    if data == 'toy':
        y_add = toy_func(toy_data_convert(X_add)).astype('float32')
    elif data == 'Nano':
        y_add = nano_simulate(nano_data_convert(X_add)).astype('float32')



    X_train = np.vstack((X_temp, X_add))
    y_train = np.vstack((y_temp, y_add))
    trainset = MetaMaterialDataSet(X_train, y_train, bool_train=True)
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)

    return X_train, y_train, trainloader

def toy_data_convert(x):
    ans = 10*x
    return ans

def nano_data_convert(x):
    ans = 50 + 20*x
    return ans

def random_X_gen(n_gen, n_feature):
    np.random.seed(np.random.randint(0, 1000000))
    xdata = np.random.uniform(-1, 1, [n_gen, n_feature]).astype('float32')

    return xdata

class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]



if __name__ == '__main__':
    X_temp, y_temp, trainloader, testloader = data_initial(5, 'toy')
    X_add = random_X_gen(2, 1)
    print(X_temp, y_temp)
    print(X_add.shape)
    X_temp, y_temp, trainloader = data_add(X_temp, y_temp, X_add, 'toy')
    print(X_temp, y_temp)