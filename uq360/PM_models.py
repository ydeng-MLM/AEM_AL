import torch

from uq360.algorithms.ensemble_heteroscedastic_regression import EnsembleHeteroscedasticRegression

def PM_model_selection(data=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data == 'toy':
        config_HR = {"num_features": 1, "num_hidden": 64, "num_outputs": 1, "batch_size": 16, "num_epochs": 100,
                     "lr": 0.001}
        HR_kwargs = {"model_type": 'mlp',
                     "config": config_HR,
                     "device": device,
                     "verbose": False}
        # define config for ensemble
        config_ensemble = {"num_models": 5,
                           "batch_size": 16,
                           "model_kwargs": HR_kwargs, }
        return EnsembleHeteroscedasticRegression(model_type='ensembleheteroscedasticregression', config=config_ensemble,
                                                device=device, verbose=False)
    elif data == 'Nano':
        config_HR = {"num_features": 8, "num_hidden": 2600, "num_outputs": 201, "batch_size": 1000, "num_epochs": 500,
                     "lr": 0.001}
        HR_kwargs = {"model_type": 'mlp',
                     "config": config_HR,
                     "device": device,
                     "verbose": False}
        # define config for ensemble
        config_ensemble = {"num_models": 5,
                           "batch_size": 1000,
                           "model_kwargs": HR_kwargs, }
        return EnsembleHeteroscedasticRegression(model_type='ensembleheteroscedasticregression', config=config_ensemble,
                                                device=device, verbose=False)
    
    elif data == 'Nano_2':
        config_HR = {"num_features": 4, "num_hidden": 2000, "num_outputs": 201, "num_layer": 4, "batch_size": 100, "num_epochs": 500,
                     "lr": 0.001}
        HR_kwargs = {"model_type": 'mlp',
                     "config": config_HR,
                     "device": device,
                     "verbose": False}
        # define config for ensemble
        config_ensemble = {"num_models": 5,
                           "batch_size": 100,
                           "model_kwargs": HR_kwargs, }
        return EnsembleHeteroscedasticRegression(model_type='ensembleheteroscedasticregression', config=config_ensemble,
                                                device=device, verbose=False)


    elif data == 'ADM':
        config_HR = {"num_features": 14, "num_hidden": 2000, "num_outputs": 2000, "num_layer": 10, "batch_size": 1000, "num_epochs": 500,
                     "lr": 0.001}
        HR_kwargs = {"model_type": 'mlp',
                     "config": config_HR,
                     "device": device,
                     "verbose": False}
        # define config for ensemble
        config_ensemble = {"num_models": 5,
                           "batch_size": 1000,
                           "model_kwargs": HR_kwargs, }
        return EnsembleHeteroscedasticRegression(model_type='ensembleheteroscedasticregression', config=config_ensemble,
                                                 device=device, verbose=False)

    elif data == 'color':
        config_HR = {"num_features": 3, "num_hidden": 2000, "num_outputs": 3,  "num_layer": 9, "batch_size": 100, "num_epochs": 500,
                     "lr": 0.001}
        HR_kwargs = {"model_type": 'mlp',
                     "config": config_HR,
                     "device": device,
                     "verbose": False}
        # define config for ensemble
        config_ensemble = {"num_models": 5,
                           "batch_size": 100,
                           "model_kwargs": HR_kwargs, }
        return EnsembleHeteroscedasticRegression(model_type='ensembleheteroscedasticregression', config=config_ensemble,
                                                 device=device, verbose=False)

    else:
        return NotImplementedError
