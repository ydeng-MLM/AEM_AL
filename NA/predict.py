"""
This file serves as a prediction interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
import time
# Torch

# Own
from NA.class_wrapper import Network
from NA.model_maker import NA
from NA.utils.helper_functions import load_flags
from NA.utils.evaluation_helper import plotMSELossDistrib
# Libs
import numpy as np

def predict_from_model(pre_trained_model, Xpred_file, no_plot=True, load_state_dict=None):
    """
    Predicting interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param Xpred_file: The Prediction file position
    :param no_plot: If True, do not plot (For multi_eval)
    :param load_state_dict: The new way to load the model for ensemble MM
    :return: None
    """
    # Retrieve the flag object
    print("This is doing the prediction for file", Xpred_file)
    print("Retrieving flag object for parameters")
    if (pre_trained_model.startswith("models")):
        eval_model = pre_trained_model[7:]
        print("after removing prefix models/, now model_dir is:", eval_model)
    
    flags = load_flags(pre_trained_model)                       # Get the pre-trained model
    flags.eval_model = pre_trained_model                    # Reset the eval mode
    flags.test_ratio = 0.2              #useless number  

    # Get the data, this part is useless in prediction but just for simplicity
    #train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(NA, flags, train_loader=None, test_loader=None, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # Evaluation process
    print("Start eval now:")
    
    if not no_plot:
        # Plot the MSE distribution
        pred_file, truth_file = ntwk.predict(Xpred_file, no_save=False, load_state_dict=load_state_dict)
        flags.eval_model = pred_file.replace('.','_') # To make the plot name different
        plotMSELossDistrib(pred_file, truth_file, flags)
    else:
        pred_file, truth_file = ntwk.predict(Xpred_file, no_save=True, load_state_dict=load_state_dict)
    
    print("Evaluation finished")

    return pred_file, truth_file, flags

def ensemble_predict(model_list, Xpred_file, model_dir=None, no_plot=True, remove_extra_files=True, state_dict=False):
    """
    This predicts the output from an ensemble of models
    :param model_list: The list of model names to aggregate
    :param Xpred_file: The Xpred_file that you want to predict
    :param model_dir: The directory to plot the plot
    :param no_plot: If True, do not plot (For multi_eval)
    :param remove_extra_files: Remove all the files generated except for the ensemble one
    :param state_dict: New way to load model using state_dict instead of load module
    :return: The prediction Ypred_file
    """
    print("this is doing ensemble prediction for models :", model_list)
    pred_list = []
    # Get the predictions into a list of np array
    for pre_trained_model in model_list:
        if state_dict is False:
            pred_file, truth_file, flags = predict_from_model(pre_trained_model, Xpred_file)
            # This line is to plot all histogram, make sure comment the pred_list.append line below as well for getting all the histograms
            #pred_file, truth_file, flags = predict_from_model(pre_trained_model, Xpred_file, no_plot=False)
        else:
            model_folder = os.path.join('..', 'Data', 'Yang_sim', 'model_param')
            pred_file, truth_file, flags = predict_from_model(model_folder, Xpred_file, load_state_dict=pre_trained_model)
        pred_list.append(np.copy(np.expand_dims(pred_file, axis=2)))
    # Take the mean of the predictions
    pred_all = np.concatenate(pred_list, axis=2)
    pred_mean = np.mean(pred_all, axis=2)
    save_name = Xpred_file.replace('Xpred', 'Ypred')
    np.savetxt(save_name, pred_mean)
    
    # If no_plot, then return
    if no_plot:
        return

    # saving the plot down
    flags.eval_model = 'ensemble_plot' + Xpred_file.replace('/', '')
    if model_dir is None:
        return plotMSELossDistrib(save_name, truth_file, flags)
    else:
        return plotMSELossDistrib(save_name, truth_file, flags, save_dir=model_dir)




def predict_all(models_dir="data"):
    """
    This function predict all the files in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if 'Xpred' in file and 'meta_material' in file:                     # Only meta material has this need currently
            print("predicting for file", file)
            predict_from_model("models/meta_materialreg0.0005trail_2_complexity_swipe_layer1000_num6", 
            os.path.join(models_dir,file))
    return None


def ensemble_predict_master(model_dir, Xpred_file, no_plot, plot_dir=None):
    print("entering folder to predict:", model_dir)
    model_list = []
    
    # patch for new way to load model using state_dict
    state_dict = False
    if 'state_dicts' in model_dir:
        state_dict = True

    # get the list of models to load
    for model in os.listdir(model_dir):
        print("entering:", model)
        if 'skip' in model or '.zip' in model :             # For skipping certain folders
            continue;
        if os.path.isdir(os.path.join(model_dir,model)) or '.pt' in model:
            model_list.append(os.path.join(model_dir, model))
    if plot_dir is None:
        return ensemble_predict(model_list, Xpred_file, model_dir, state_dict=state_dict, no_plot=no_plot)
    else:
        return ensemble_predict(model_list, Xpred_file, plot_dir, state_dict=state_dict, no_plot=no_plot)
        


def predict_ensemble_for_all(model_dir, Xpred_file_dirs, no_plot):
    for files in os.listdir(Xpred_file_dirs):
        if 'Xpred' in files and 'Yang' in files:
            # If this has already been predicted, skip this file!
            if os.path.isfile(Xpred_file_dirs.replace('Xpred','Ypred')):
                continue
            ensemble_predict_master(model_dir, os.path.join(Xpred_file_dirs, files), plot_dir=Xpred_file_dirs, no_plot=no_plot)

def creat_mm_dataset():
    """
    Function to create the meta-material dataset from the saved checkpoint files
    :return:
    """
    # Define model folder
    model_folder = os.path.join('.', 'Yang_sim', 'model_param')
    # Load the flags to construct the model
    flags = load_flags(model_folder)
    flags.eval_model = model_folder
    ntwk = Network(NA, flags, train_loader=None, test_loader=None, inference_mode=True, saved_model=flags.eval_model)
    # This is the full file version, which would take a while. Testing pls use the next line one
    geometry_points = os.path.join('.', 'Yang_sim', 'dataIn', 'data_x.csv')
    # Small version is for testing, the large file taks a while to be generated...
    #geometry_points = os.path.join('..', 'Simulated_DataSets', 'Meta_material_Neural_Simulator', 'dataIn', 'data_x_small.csv')
    Y_filename = geometry_points.replace('data_x', 'data_y')

    # Set up the list of prediction files
    pred_list = []
    num_models = 10

    # for each model saved, load the dictionary and do the inference
    for i in range(num_models):
        print('predicting for {}th model saved'.format(i+1))
        state_dict_file = os.path.join('.', 'Yang_sim', 'state_dicts', 'mm{}.pt'.format(i))
        pred_file, truth_file = ntwk.predict(Xpred_file=geometry_points, load_state_dict=state_dict_file, no_save=True)
        pred_list.append(pred_file)

    Y_ensemble = np.zeros(shape=(*np.shape(pred_file), num_models))
    # Combine the predictions by doing the average
    for i in range(num_models):
        Y_ensemble[:, : ,i] = pred_list[i]

    Y_ensemble = np.mean(Y_ensemble, axis=2)
    #X = pd.read_csv(geometry_points, header=None, sep=' ').values
    #MM_data = np.concatenate((X, Y_ensemble), axis=1)
    #MM_data_file = geometry_points.replace('data_x', 'dataIn/MM_data')
    np.savetxt(Y_filename, Y_ensemble)
    #np.savetxt(MM_data_file, MM_data)


def ADM_generate(n_gen=100):
    """
    Function to generate both geometry and spectrum for all-dielectric metasurface dataset
    """
    # Define model folder
    model_folder = os.path.join('NA', 'Yang_sim', 'model_param')
    # Load the flags to construct the model
    flags = load_flags(model_folder)
    flags.eval_model = model_folder
    ntwk = Network(NA, flags, train_loader=None, test_loader=None, inference_mode=True, saved_model=flags.eval_model)
    # This is the full file version, which would take a while. Testing pls use the next line one
    geometry_points = np.random.uniform(-1, 1, [n_gen, 14]).astype('float32')

    # Set up the list of prediction files
    pred_list = []
    num_models = 10

    # for each model saved, load the dictionary and do the inference
    for i in range(num_models):
        state_dict_file = os.path.join('NA', 'Yang_sim', 'state_dicts', 'mm{}.pt'.format(i))
        pred_file = ntwk.simulate(geometry_points, load_state_dict=state_dict_file)
        pred_list.append(pred_file)

    Y_ensemble = np.zeros(shape=(*np.shape(pred_file), num_models))
    # Combine the predictions by doing the average
    for i in range(num_models):
        Y_ensemble[:, : ,i] = pred_list[i]

    Y_ensemble = np.mean(Y_ensemble, axis=2)
    return geometry_points, Y_ensemble

def ADM_simulate(X):
    """
    Function to generate both geometry and spectrum for all-dielectric metasurface dataset
    """
    # Define model folder
    model_folder = os.path.join('NA', 'Yang_sim', 'model_param')
    # Load the flags to construct the model
    flags = load_flags(model_folder)
    flags.eval_model = model_folder
    ntwk = Network(NA, flags, train_loader=None, test_loader=None, inference_mode=True, saved_model=flags.eval_model)
    # This is the full file version, which would take a while. Testing pls use the next line one
    geometry_points = X

    # Set up the list of prediction files
    pred_list = []
    num_models = 10

    # for each model saved, load the dictionary and do the inference
    for i in range(num_models):
        state_dict_file = os.path.join('NA', 'Yang_sim', 'state_dicts', 'mm{}.pt'.format(i))
        pred_file = ntwk.simulate(geometry_points, load_state_dict=state_dict_file)
        pred_list.append(pred_file)

    Y_ensemble = np.zeros(shape=(*np.shape(pred_file), num_models))
    # Combine the predictions by doing the average
    for i in range(num_models):
        Y_ensemble[:, : ,i] = pred_list[i]

    Y_ensemble = np.mean(Y_ensemble, axis=2)
    return Y_ensemble

if __name__ == '__main__':
    # To create Meta-material dataset, use this line 
    start = time.time()
    #creat_mm_dataset()
    X, y = ADM_generate()
    print(X.shape, y.shape)
    print('Time is spend on producing MM dataset is {}'.format(time.time()-start))
    
    # multi evaluation 
    #method_list = ['Tandem','MDN','INN','cINN','VAE','GA','NA','NN']
    #for method in method_list:
       #predict_ensemble_for_all('../Data/Yang_sim/state_dicts/', '../mm_bench_multi_eval/' + method + '/Yang_sim/', no_plot=True)
    
    #predict_ensemble_for_all('../Data/Yang_sim/state_dicts/', '/home/sr365/MM_Bench/GA/temp-dat/GA1_chrome_gen_300/Yang_sim', no_plot=True)  
    
    #predict_from_model("models/Yang_sim_best_model", 'data/test_Xpred_Yang_best_model.csv', no_plot=False, load_state_dict=None)