from GPyOpt.methods import BayesianOptimization

from AL_central import AL_train

DATA = 'color'
MSE_T = 1e-3

def obj_func_dqbc(x):
    n_pool = 100*int(x[0, 0])
    n_com = int(x[0, 1])
    rd = x[0, 2]
    print(n_pool, n_com, rd)
    mse, n_train = AL_train(mse_target=MSE_T, n_init=1000, n_test=10000, n_batch=100, n_pool=n_pool, n_com=n_com, rd=rd, data_type=DATA, model_type='MLP', query_type='dqbc')
    return n_train

def bayesian_opt_dqbc():
    bounds = [{'name': 'n_pool', 'type': 'continuous', 'domain': (2, 10)},
              {'name': 'n_com', 'type': 'continuous', 'domain': (2, 10)},
              {'name': 'rd', 'type': 'discrete', 'domain': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

    BO = BayesianOptimization(obj_func_dqbc, domain=bounds)
    BO.run_optimization(max_iter=30)
    print("=" * 20)
    print("Value of (x,y) that minimises the objective:" + str(BO.x_opt))
    print("Minimum value of the objective: " + str(BO.fx_opt))
    print("=" * 20)

def obj_func_qbc(x):
    n_pool = 100*int(x[0, 0])
    n_com = int(x[0, 1])
    print(n_pool, n_com)
    mse, n_train = AL_train(mse_target=MSE_T, n_init=1000, n_test=1000, n_batch=100, n_pool=n_pool, n_com=n_com, rd=0, data_type=DATA, model_type='MLP', query_type='qbc')
    return n_train

def bayesian_opt_qbc():
    bounds = [{'name': 'n_pool', 'type': 'continuous', 'domain': (2, 10)},
              {'name': 'n_com', 'type': 'continuous', 'domain': (2, 10)}]

    BO = BayesianOptimization(obj_func_qbc, domain=bounds)
    BO.run_optimization(max_iter=30)
    print("=" * 20)
    print("Value of (x,y) that minimises the objective:" + str(BO.x_opt))
    print("Minimum value of the objective: " + str(BO.fx_opt))
    print("=" * 20)

def obj_func_gs(x):
    n_pool = 10*int(x[0, 0])
    print(n_pool)
    mse, n_train = AL_train(mse_target=MSE_T, n_init=1000, n_test=1000, n_batch=100, n_pool=n_pool, n_com=1, rd=0, data_type=DATA, model_type='MLP', query_type='GS')
    return n_train

def bayesian_opt_gs():
    bounds = [{'name': 'n_pool', 'type': 'continuous', 'domain': (1, 100)}]

    BO = BayesianOptimization(obj_func_gs, domain=bounds)
    BO.run_optimization(max_iter=30)
    print("=" * 20)
    print("Value of (x,y) that minimises the objective:" + str(BO.x_opt))
    print("Minimum value of the objective: " + str(BO.fx_opt))
    print("=" * 20)

def obj_func_pm(x):
    n_pool = 100*int(x[0, 0])
    print(n_pool)
    mse, n_train = AL_train(mse_target=MSE_T, n_init=1000, n_test=1000, n_batch=100, n_pool=n_pool, n_com=1, rd=0, data_type=DATA, model_type='MLP', query_type='PM')
    return n_train

def bayesian_opt_pm():
    bounds = [{'name': 'n_pool', 'type': 'continuous', 'domain': (2, 10)}]

    BO = BayesianOptimization(obj_func_pm, domain=bounds)
    BO.run_optimization(max_iter=30)
    print("=" * 20)
    print("Value of (x,y) that minimises the objective:" + str(BO.x_opt))
    print("Minimum value of the objective: " + str(BO.fx_opt))
    print("=" * 20)

if __name__ == '__main__':
    bayesian_opt_dqbc()
    #bayesian_opt_qbc()
    #bayesian_opt_gs()
    #bayesian_opt_pm()

