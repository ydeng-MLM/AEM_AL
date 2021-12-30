import torch
import numpy as np

from model_func import MLP_nano, MLP_toy
from data_processing import data_initial, data_add, random_X_gen

from sklearn.metrics import pairwise_distances


def query_GS(X_train, n_batch, n_pool):
  #Query term calculation, for greedy sampling, we are purely calculating the distance
  X_pool = random_X_gen(n_pool, X_train.shape[1])
  X_add = []
  for i in range(n_batch):
      dist = pairwise_distances(X_train, X_pool, metric='euclidean')
      D_min = np.min(dist, axis=0).reshape(-1, 1)
      X_add.append(X_pool[np.argmax(D_min), :])
      X_train = np.vstack((X_train, X_pool[np.argmax(D_min), :]))
      X_pool = np.delete(X_pool, np.argmax(D_min), 0)

  X_add = np.array(X_add).astype('float32')
  return X_add

def AL_train(mse_target, n_init, n_batch, n_pool, data_type='Nano'):
    X_temp, y_temp, trainloader, testloader = data_initial(n_sample=n_init, data=data_type)

    mse_GS = np.array([])
    mse = 100
    n_total = n_init
    counter = 0

    while mse > mse_target:
        dnn = MLP_nano(trainloader, testloader)
        dnn.load()
        dnn.train(verbose=False)
        mse = dnn.predict()
        print("At GS iteration " + str(counter) + f": mse = {mse:.5f}")
        mse_GS = np.append(mse_GS, mse)
        X_add = query_GS(X_temp, n_batch, n_pool)
        X_temp, y_temp, trainloader = data_add(X_temp, y_temp, X_add, data=data_type)
        n_total += n_batch
        counter += 1

    return mse_GS, n_total

if __name__ == '__main__':
    AL_train(mse_target=8e-3, n_init=1000, n_batch=1000, n_pool=10000, data_type='Nano')