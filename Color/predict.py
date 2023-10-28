import os
import torch
import numpy as np

from Color import MLP_MIXER

def color_generate(n_gen):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mixer = torch.load(os.path.join('Color/saved_model/', 'Color'+'.pth'), map_location=device)

    X = np.random.uniform(0, 1, [n_gen, 3]).astype('float32')
    X[:, 0] = 0.05*X[:, 0]
    X[:, 2] = 0.05 * X[:, 2]

    if n_gen > 5000:
        y = np.zeros((n_gen, 3))
        for i in range(int(n_gen/1000)):
            X_tensor = torch.from_numpy(X[1000*i:1000*(i+1), :]).to(torch.float).cuda()
            y[1000*i:1000*(i+1), :] = mixer.forward(X_tensor).detach().cpu().numpy()
    else:
        X_tensor = torch.from_numpy(X).to(torch.float).cuda()
        y = mixer.forward(X_tensor).detach().cpu().numpy()
    return X, y

def color_simulate(X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mixer = torch.load(os.path.join('Color/saved_model/', 'Color' + '.pth'), map_location=device)

    X_tensor = torch.from_numpy(X).to(torch.float).cuda()
    y = mixer.forward(X_tensor).detach().cpu().numpy()
    return y

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    X, y = color_generate(10000)
    print(X.shape, y.shape)




