import numpy as np
from AL_central import AL_train


if __name__ == '__main__':   
    for i in range(10):
        data_types = ['color']
        query_types = ['PCA_GS']
        for data_type in data_types:
            for query_type in query_types:
                for n_init in [500]:
                    for pool_ratio in [500]:
                        n_pool = pool_ratio
                        n_batch = 100
                        n_com = 1
                        rd = 1
                        if query_type == 'qbc':
                            n_pool = n_batch*pool_ratio
                            n_com = 5
                        elif query_type == 'optimal':
                            n_pool = n_batch*10
                            n_com = 1
                        elif query_type == 'worst':
                            n_pool = n_batch*10
                            n_com = 1
                        elif query_type == 'dqbc':
                            n_pool = n_batch*10
                            n_com = 5
                            rd = 0.5
                        elif query_type == 'PM':
                            n_pool = n_batch*pool_ratio
                        print(n_pool, n_com, rd)
                        mse_array, n_total = AL_train(mse_target=1e-9, n_init=n_init, n_test=1000, n_batch=n_batch, n_pool=n_pool, n_com=n_com, rd=rd, data_type=data_type, model_type='MLP', query_type=query_type)
                        #np.savetxt('benchmark_summary/color_opt/062122/'+data_type+'_mse_array_n_init_'+str(n_init)+'_pool_ratio_'+str(pool_ratio)+'_'+query_type+str(i)+'.csv', mse_array, delimiter=',')
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
