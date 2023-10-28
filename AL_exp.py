import os
import numpy as np

from model_func import MLP_nano, MLP_nano2, MLP_toy, MLP_ADM, MLP_color
from data_processing import data_initial, data_add, random_X_gen
from uq360.PM_models import PM_model_selection
from nano_generator import toy_func, nano_simulate

from sklearn.metrics import pairwise_distances
from NA.predict import ADM_simulate
from Color.predict import color_simulate
from data_processing import color_data_convert, nano_data_convert\

def query_GS(X_train, n_batch, n_pool):
  #Query term calculation, for greedy sampling, we are purely calculating the distance
  #X_pool = random_X_gen(n_pool, X_train.shape[1])
  X_add = []
  for i in range(n_batch):
      X_pool = random_X_gen(n_pool, X_train.shape[1])
      dist = pairwise_distances(X_train, X_pool, metric='euclidean')
      D_min = np.min(dist, axis=0).reshape(-1, 1)
      X_add.append(X_pool[np.argmax(D_min), :])
      X_train = np.vstack((X_train, X_pool[np.argmax(D_min), :]))
      #X_pool = np.delete(X_pool, np.argmax(D_min), 0) #This is for batch GS mode

  X_add = np.array(X_add).astype('float32')
  return X_add, 0, np.max(D_min)


def query_qbc(X_train, n_batch, n_pool, n_com, trainloader, testloader, data_type=None, model_type=None):
    X_pool = random_X_gen(n_pool, X_train.shape[1])
    y_pred_qbc = []

    for i in range(n_com):
        dnn = model_selection(trainloader, testloader, data_type, model_type)
        dnn.train(verbose=False)
        y_pred = dnn.forward(X_pool)
        y_pred_qbc.append(y_pred)

    y_pred_qbc = np.array(y_pred_qbc)

    # Query term calculation, for QBC, we can simply use variance between the outputs of different models
    var = np.mean(np.var(y_pred_qbc, axis=0), axis=1)

    X_add = []
    for i in range(n_batch):
        X_add.append(X_pool[np.argmax(var), :])
        X_train = np.vstack((X_train, X_pool[np.argmax(var), :]))
        X_pool = np.delete(X_pool, np.argmax(var), 0)
        var = np.delete(var, np.argmax(var), 0)

    X_add = np.array(X_add).astype('float32')
    return X_add, np.max(var), 0

def query_PM(X_train, y_train, n_batch, n_pool, trainloader, testloader, data_type=None):
    X_pool = random_X_gen(n_pool, X_train.shape[1])
    dnn = model_selection(trainloader, testloader, data_type, model_type='PM')
    dnn.fit(X_train, y_train)
    res = dnn.predict(X_pool)
    var = np.mean((res.y_upper - res.y_lower), axis=1)

    X_add = []
    for i in range(n_batch):
        X_add.append(X_pool[np.argmax(var), :])
        X_train = np.vstack((X_train, X_pool[np.argmax(var), :]))
        X_pool = np.delete(X_pool, np.argmax(var), 0)
        var = np.delete(var, np.argmax(var), 0)

    X_add = np.array(X_add).astype('float32')
    return X_add

def query_optimal(X_train, n_batch, n_pool, n_com, trainloader, testloader, data_type=None, model_type=None):
    X_pool = random_X_gen(n_pool, X_train.shape[1])
    
    dnn = model_selection(trainloader, testloader, data_type, model_type)
    dnn.load()
    dnn.train(verbose=False)
    y_pred = np.array(dnn.forward(X_pool))
    
    if data_type == 'toy':
        y_truth = toy_func(10*X_pool)
        mse = np.square(y_pred-y_truth)
    elif data_type == 'ADM':
        y_truth = ADM_simulate(X_pool)
        mse = (np.mean(np.square(y_pred-y_truth), axis=1))
    elif data_type == 'color':
        y_truth = color_simulate(color_data_convert(X_pool))
        mse = (np.mean(np.square(y_pred-y_truth), axis=1))
    elif data_type == 'Nano':
        y_truth = nano_simulate(nano_data_convert(X_pool))
        mse = (np.mean(np.square(y_pred-y_truth), axis=1))
    



    y_pred_qbc = []

    for i in range(n_com):
        dnn = model_selection(trainloader, testloader, data_type, model_type)
        dnn.train(verbose=False)
        y_pred = dnn.forward(X_pool)
        y_pred_qbc.append(y_pred)

    y_pred_qbc = np.array(y_pred_qbc)

    # Query term calculation, for QBC, we can simply use variance between the outputs of different models
    var = np.mean(np.var(y_pred_qbc, axis=0), axis=1)
    
    mse_ans = np.copy(mse)
    print(mse_ans.shape, var.shape)


    X_add = []
    for i in range(n_batch):
        X_add.append(X_pool[np.argmax(mse), :])
        X_train = np.vstack((X_train, X_pool[np.argmax(mse), :]))
        X_pool = np.delete(X_pool, np.argmax(mse), 0)
        mse = np.delete(mse, np.argmax(mse), 0)

    X_add = np.array(X_add).astype('float32')
    return X_add, mse_ans, var




def query_gqbc(X_train, n_batch, n_pool, n_com, rd, trainloader, testloader, data_type=None, model_type=None):
    n_batch_qbc = int((1-rd)*n_batch)
    n_batch_GS = n_batch - n_batch_qbc
    X_qbc, avar, _ = query_qbc(X_train, n_batch_qbc, n_pool, n_com, trainloader, testloader, data_type, model_type)
    X_train = np.vstack((X_train, X_qbc))
    X_GS, _, dmins = query_GS(X_train, n_batch_GS, n_pool)
    X_add = np.vstack((X_qbc, X_GS))
    return X_add, avar, dmins

def query_bqbc(X_train, n_batch, n_pool, n_com, rd, trainloader, testloader, data_type=None, model_type=None):
    X_pool = random_X_gen(n_pool, X_train.shape[1])
    y_pred_qbc = []

    for i in range(n_com):
        dnn = model_selection(trainloader, testloader, data_type, model_type)
        dnn.train(verbose=False)
        y_pred = dnn.forward(X_pool)
        y_pred_qbc.append(y_pred)

    y_pred_qbc = np.array(y_pred_qbc)

    # Query term calculation, for QBC, we can simply use variance between the outputs of different models
    var = np.mean(np.var(y_pred_qbc, axis=0), axis=1)

    #Calculating the Euclidean distance for diverstiy consideration
    dist = pairwise_distances(X_train, X_pool, metric='euclidean')
    D_min = np.min(dist, axis=0).reshape(-1, 1)

    dvar = var + rd*D_min.squeeze() #var has shape (100,), D_min has shape (100,1), use squeeze to ensure dvar has 1 dimension shape

    X_add = []
    for i in range(n_batch):
        X_add.append(X_pool[np.argmax(dvar), :])
        X_train = np.vstack((X_train, X_pool[np.argmax(dvar), :]))
        X_pool = np.delete(X_pool, np.argmax(dvar), 0)
        dvar = np.delete(dvar, np.argmax(dvar), 0)

    X_add = np.array(X_add).astype('float32')
    return X_add, np.max(var), np.max(D_min)


def query_dqbc(X_train, n_batch, n_pool, n_com, rd, trainloader, testloader, data_type=None, model_type=None):
    X_pool = random_X_gen(n_pool, X_train.shape[1])
    y_pred_qbc = []
    qbc_mses = []
    trains = []
    tests = []

    for i in range(n_com):
        dnn = model_selection(trainloader, testloader, data_type, model_type)
        train, test = dnn.train(verbose=False)
        trains.append(train)
        tests.append(test)
        qbc_mses.append(dnn.predict())
        y_pred = dnn.forward(X_pool)
        y_pred_qbc.append(y_pred)

    y_pred_qbc = np.array(y_pred_qbc)

    # Query term calculation, for QBC, we can simply use variance between the outputs of different models
    var = np.mean(np.var(y_pred_qbc, axis=0), axis=1)

    #Calculating the Euclidean distance for diverstiy consideration
    dist = pairwise_distances(X_train, X_pool, metric='euclidean')
    D_min = np.min(dist, axis=0).reshape(-1, 1)
    #X_total = np.vstack((X_pool, X_train))
    #dist = pairwise_distances(X_total, X_pool, metric='euclidean')
    #D_min = np.zeros(var.shape)
    #for i in range(dist.shape[1]):
    #    D_min[i] = np.min(dist[dist[:, i] != np.min(dist[:, i]), i])

    #dvar = var + rd*D_min.squeeze() #var has shape (100,), D_min has shape (100,1), use squeeze to ensure dvar has 1 dimension shape

    X_add = []
    avars = []
    D_mins = []
    for i in range(n_batch):
        dvar = var + rd*D_min.squeeze()
        avars.append(var[np.argmax(dvar)])
        D_mins.append(D_min.squeeze()[np.argmax(dvar)])
        X_add.append(X_pool[np.argmax(dvar), :])
        X_train = np.vstack((X_train, X_pool[np.argmax(dvar), :]))
        X_pool = np.delete(X_pool, np.argmax(dvar), 0)
        #dvar = np.delete(dvar, np.argmax(dvar), 0)
        var = np.delete(var, np.argmax(dvar), 0)
        dist = pairwise_distances(X_train, X_pool, metric='euclidean')
        D_min = np.min(dist, axis=0).reshape(-1, 1)

    X_add = np.array(X_add).astype('float32')
    return X_add, avars, D_mins, qbc_mses, trains, tests

def model_selection(trainloader, testloader, data_type=None, model_type=None):
    if model_type == 'MLP':
        if data_type == 'toy':
            return MLP_toy(trainloader, testloader)
        elif data_type == 'Nano':
            return MLP_nano(trainloader, testloader)
        elif data_type == 'Nano_2':
            return MLP_nano2(trainloader, testloader)
        elif data_type == 'ADM':
            return MLP_ADM(trainloader, testloader)
        elif data_type == 'color':
            return MLP_color(trainloader, testloader)

    elif model_type == 'PM':
            return PM_model_selection(data_type)

def query(X_temp, y_temp, n_batch, n_pool, n_com, rd, trainloader, testloader, data_type=None, model_type=None, query_type=None):
    if query_type == 'qbc':
        return query_qbc(X_temp, n_batch, n_pool, n_com, trainloader, testloader, data_type, model_type)
    elif query_type == 'GS':
        return query_GS(X_temp, n_batch, n_pool)
    elif query_type == 'PM':
        return query_PM(X_temp, y_temp, n_batch, n_pool, trainloader, testloader, data_type)
    elif query_type == 'dqbc':
        return query_dqbc(X_temp, n_batch, n_pool, n_com, rd, trainloader, testloader, data_type, model_type)
    elif query_type == 'bqbc':
        return query_bqbc(X_temp, n_batch, n_pool, n_com, rd, trainloader, testloader, data_type, model_type)
    elif query_type == 'gqbc':
        return query_gqbc(X_temp, n_batch, n_pool, n_com, rd, trainloader, testloader, data_type, model_type)
    elif query_type == 'random':
        return random_X_gen(n_batch, X_temp.shape[1]), 0, 0
    elif query_type == 'optimal':
         return query_optimal(X_temp, n_batch, n_pool, n_com, trainloader, testloader, data_type, model_type)

def AL_train(mse_target, n_init, n_test, n_batch, n_pool, n_com, rd, data_type=None, model_type=None, query_type=None):
    X_temp, y_temp, X_test, y_test, trainloader, testloader = data_initial(n_sample=n_init, n_test=n_test, data=data_type)

    mse_GS = np.array([])
    mse = 100
    counter = 0

    if data_type == 'Nano':
        max_len = 15000
    elif data_type == 'Nano_2':
        max_len = 5000
    elif data_type == 'ADM':
        max_len = 30000
    elif data_type == 'color':
        max_len = 5000
    elif data_type == 'toy':
        max_len = 500
    
    avars = []
    D_mins = []

    while mse > mse_target and len(X_temp) <= max_len:
        dnn = model_selection(trainloader, testloader, data_type, model_type)
        dnn.load()
        dnn.train(verbose=False)
        mse= dnn.predict()
        print("At " + query_type + " iteration " + str(counter) + f": # of sample = {len(X_temp)}, test mse = {mse:.5f}")
        mse_GS = np.append(mse_GS, mse)
        X_add, var, D_min = query(X_temp, y_temp, n_batch, n_pool, n_com, rd, trainloader, testloader, data_type, model_type, query_type)
        avars.extend(var)
        D_mins.extend(D_min)
        X_temp, y_temp, trainloader = data_add(X_temp, y_temp, X_add, data=data_type)
        counter += 1
        #y_pred = dnn.forward(X_test)
        #np.savetxt(data_type+'_Xtest_'+query_type+'_iteration'+str(counter)+'.csv', X_test, delimiter=',')
        #np.savetxt(data_type+'_ypred_'+query_type+'_iteration'+str(counter)+'.csv', y_pred, delimiter=',')
        #np.savetxt(data_type+'_batch'+str(n_batch)+'_train_data_'+query_type+'_iteration'+str(counter)+'.csv', X_temp, delimiter=',')
        print(var.shape, D_min.shape)
        np.savetxt(data_type+'_batch'+str(n_batch)+'_pool_mse_'+query_type+'_iteration'+str(counter)+'.csv', var, delimiter=',')
        np.savetxt(data_type+'_batch'+str(n_batch)+'_pool_var_'+query_type+'_iteration'+str(counter)+'.csv', D_min, delimiter=',')
        #np.savetxt(data_type+'_batch'+str(n_batch)+'_qbc_mse'+query_type+'_iteration'+str(counter)+'.csv', qbc_mses, delimiter=',')
        #np.savetxt(data_type+'_batch'+str(n_batch)+'_train_curve'+query_type+'_iteration'+str(counter)+'.csv', train, delimiter=',')
        #np.savetxt(data_type+'_batch'+str(n_batch)+'_test_curve'+query_type+'_iteration'+str(counter)+'.csv', test, delimiter=',')



    return mse_GS, len(X_temp), avars, D_mins

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    for i in range(1):
        data_types = ['ADM']
        query_types = ['optimal']
        for data_type in data_types:
            for query_type in query_types:
                for n_batch in [1000]:
                    n_pool = 100
                    n_com = 2
                    rd = 1
                    if query_type == 'qbc':
                        n_pool = n_batch*10
                        n_com = 5
                    if query_type == 'optimal':
                        n_pool = n_batch*10
                        n_com = 5
                    if query_type == 'dqbc':
                        n_pool = n_batch*100
                        n_com = 5
                        rd = 1
                    mse_array, n_total, avars, dmins = AL_train(mse_target=1e-9, n_init=10000, n_test=1000, n_batch=n_batch, n_pool=n_pool, n_com=n_com, rd=rd, data_type=data_type, model_type='MLP', query_type=query_type)
                    #np.savetxt(data_type+'_batch'+str(n_batch)+'_mse_array_'+query_type+str(i)+'.csv', mse_array, delimiter=',')
                    #np.savetxt(data_type+'_batch'+str(n_batch)+query_type+'_vars'+str(i)+'.csv', avars, delimiter=',')
                    #np.savetxt(data_type+'_batch'+str(n_batch)+query_type+'_dmins'+str(i)+'.csv', dmins, delimiter=',')
    '''
    X_temp, y_temp, trainloader, testloader = data_initial(n_sample=10, n_test=10, data='ADM')
    X_pool = random_X_gen(10000, X_temp.shape[1])
    dnn = model_selection(trainloader, testloader, 'ADM', model_type='MLP')
    dnn.load()
    dnn.train(verbose=False)
    mse = dnn.predict()
    print(mse)
    X_add = query(X_temp, y_temp, 10, 10, trainloader, testloader, 'ADM', 'MLP', 'GS')
    print(X_add)
    X_temp, y_temp, trainloader = data_add(X_temp, y_temp, X_add, data='ADM')
    '''
