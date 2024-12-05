#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from numpy.random import binomial
import torch
import random
import botorch
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from sklearn import preprocessing
from scipy.stats import multivariate_normal
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood


# In[9]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 31
torch.manual_seed(seed)
np.random.seed(seed)
# device


# In[10]:


large = 25; med = 19; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': 20,
          'figure.figsize': (27, 8),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': med}
plt.rcParams.update(params)


# # Load Data:

# In[94]:


x_all = np.load("../data/x_all.npy")
x_val = np.load("../data/x_val.npy")
x_test = np.load("../data/x_test.npy")
y_all = np.load("../data/y_all.npy")
y_val = np.load("../data/y_val.npy")
y_test = np.load("../data/y_test.npy")
# print(y_test)



# In[15]:


#reference: https://chrisorm.github.io/NGP.html
class REncoder(torch.nn.Module):
    """Encodes inputs of the form (x_i,y_i) into representations, r_i."""
    
    def __init__(self, in_dim, out_dim, init_func = torch.nn.init.normal_):
        super(REncoder, self).__init__()
        self.l1_size = 64 #16
        self.l2_size = 32 #8
        self.l3_size = 16 #DNE
        
        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, self.l3_size)
        self.l4 = torch.nn.Linear(self.l3_size, out_dim)
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()
        self.a3 = torch.nn.Sigmoid()
        
        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)
            init_func(self.l4.weight)
        
    def forward(self, inputs):
        return self.l4(self.a3(self.l3(self.a2(self.l2(self.a1(self.l1(inputs)))))))

class ZEncoder(torch.nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the 
    normally distributed function encoding, z."""
    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(ZEncoder, self).__init__()
        self.m1_size = out_dim
        self.logvar1_size = out_dim
        
        self.m1 = torch.nn.Linear(in_dim, self.m1_size)
        self.logvar1 = torch.nn.Linear(in_dim, self.m1_size)

        if init_func is not None:
            init_func(self.m1.weight)
            init_func(self.logvar1.weight)
        
    def forward(self, inputs):

        return self.m1(inputs), self.logvar1(inputs)


def MAE(pred, target):
#     print(target.unsqueeze(2).shape)
    loss = torch.abs(pred-(target.unsqueeze(2)[:,1:,...]))
    return loss.mean()


# In[16]:


conv_outDim = 64
init_channels = 4
image_channels_in_encoder = 4
image_channels_in_decoder = 2
kernel_size = 3
lstm_hidden_size = 128
decoder_init = torch.unsqueeze(torch.from_numpy(np.load("../data/initial_pic.npy")).float().to(device), 0)

class ConvEncoder(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            #input_shape = (2,30,30)
            nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
#             nn.Dropout(p = .2),
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.output = nn.Sequential(
            #start pushing back
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(p = .2),
            nn.Linear(32, conv_outDim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
#         print("x before reshape", x.shape)
        x = x.view(x.size(0), -1)
#         print("x after reshape: ", x.shape)
        output = self.output(x)
#         print("shape of encoder output: ", output.shape)
        
        return output


# In[17]:


class ConvDecoder(nn.Module):
    def __init__(self, in_dim, out_dim) :
        super().__init__()
        self.input = nn.Sequential(
            #start pushing back
            nn.Linear(in_dim, conv_outDim),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=conv_outDim, out_channels=init_channels*8, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
#             nn.Dropout(p = .2),
#             nn.MaxPool2d(kernel_size = 3)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
#             nn.MaxPool2d(kernel_size = 3)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels_in_decoder, kernel_size = kernel_size,stride = 2, output_padding = 1),
            nn.ReLU()
#             nn.MaxPool2d(kernel_size = 3)
        )

        
    def forward(self, x_pred):
        """x_pred: No. of data points, by x_dim
        z: No. of samples, by z_dim
        """
        x = self.input(x_pred)
        x = x.view(-1, conv_outDim, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.conv4(x)
#         print("predicted image shape", output.shape)
        
        return output


# In[18]:


class DCRNNModel(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, init_func=torch.nn.init.normal_):
        super().__init__()
        self.conv_encoder = ConvEncoder(image_channels_in_encoder)
        self.conv_encoder_in_decoder = ConvEncoder(image_channels_in_decoder)
        self.deconv = ConvDecoder(lstm_hidden_size, y_dim) 
        self.encoder_lstm = nn.LSTM(input_size = conv_outDim+x_dim, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size = conv_outDim+x_dim+z_dim, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
#         self.repr_encoder = REncoder(x_dim+conv_outDim, r_dim) # (x,y)->r
        self.z_encoder = ZEncoder(lstm_hidden_size, z_dim) # r-> mu, logvar
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.zdim = z_dim
        self.xdim = x_dim
        self.y_init = decoder_init

    def stack_y(self, y):
        seq1 = y[:-1]
        seq2 = y[1:]
        seq3 = torch.cat((seq1, seq2), 1)
        return seq3
    
    def data_to_z_params(self, x, y):
        """Helper to batch together some steps of the process."""
        rs_all = None
        for i,theta_seq in enumerate(y):
            y_stacked = self.stack_y(theta_seq)
            y_conv_c = self.conv_encoder(y_stacked)
            encode_hidden_state = None
            xy = torch.cat([y_conv_c, x[i].repeat(len(y_stacked)).reshape(-1,x_dim)], dim=1)
#             print("shape of xy: ", xy.shape)
            rs , encode_hidden_state = self.encoder_lstm(xy, encode_hidden_state)
#             self.lstm_linear(rs)
            rs = rs.unsqueeze(0)
#             print("shape of rs: ", rs.shape)
            if rs_all is None:
                rs_all = rs
            else:
                rs_all = torch.vstack((rs_all, rs))
            
        r_agg = rs_all.mean(dim=0) 
        return self.z_encoder(r_agg) # Get mean and variance for q(z|...)
    
    def sample_z(self, mu, logvar,n=1):
        """Reparameterisation trick."""
        if n == 1:
            eps = torch.autograd.Variable(logvar.data.new(z_dim).normal_()).to(device)
        else:
            eps = torch.autograd.Variable(logvar.data.new(n,z_dim).normal_()).to(device)
        
        # std = torch.exp(0.5 * logvar)
        std = 0.1+ 0.9*torch.sigmoid(logvar)
#         print(mu + std * eps)
        return mu + std * eps
    
    def data_to_z(self, x, y, num_z_samples=1):
        mu, logvar = self.data_to_z_params(x, y)
        return self.sample_z(mu, logvar, num_z_samples)

    def xy_to_y_pred(self, x, y, x_target, num_z_samples=1):
        batch_size = x.size(0)
        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(x, y)
        z_samples = self.data_to_z(x, y, num_z_samples).view(-1, self.zdim)
        z_samples = z_samples.unsqueeze(1)
        h = torch.zeros(batch_size, 1, lstm_hidden_size, dtype=torch.float32).to(device)
        c = torch.zeros(batch_size, 1, lstm_hidden_size, dtype=torch.float32).to(device)
        y_pred_all = None
        for k in range(1):
            x_tiled = x_target.unsqueeze(0)
            x_z_pairs = torch.cat([x_tiled, z_samples], dim=2)
            _, (h, c) = self.decoder_lstm(x_z_pairs, (h, c))
            x_conv_c = self.conv_encoder_in_decoder(y)
            h_pred = torch.cat([x_conv_c, x_tiled], dim=2)
            h_pred = h_pred.view(h_pred.size(0) * h_pred.size(1), h_pred.size(2))
            h_pred = torch.cat([h_pred, h[-1:,:,:]], dim=1)
            y_pred = self.deconv(h_pred)
            y_pred = y_pred.view(batch_size, -1, y_pred.size(1), y_pred.size(2), y_pred.size(3))
            if y_pred_all is None:
                y_pred_all = y_pred
            else:
                y_pred_all = torch.cat((y_pred_all, y_pred), dim=1)
        return y_pred_all

    def KLD_gaussian(self):
        """Analytical KLD between 2 Gaussians."""
        mu_q, logvar_q, mu_p, logvar_p = self.z_mu_all, self.z_logvar_all, self.z_mu_context, self.z_logvar_context

        std_q = 0.1+ 0.9*torch.sigmoid(logvar_q)
        std_p = 0.1+ 0.9*torch.sigmoid(logvar_p)
        p = torch.distributions.Normal(mu_p, std_p)
        q = torch.distributions.Normal(mu_q, std_q)
        return torch.distributions.kl_divergence(q, p).sum()
    
    def decoder(self, theta, z_mu, z_log, seq_len=5, prev_pred = None):
        if (prev_pred is None):
            prev_pred = self.y_init
       
        outputs = None
        encoded_states = None
        deconv_encoder_hidden_state = None

        
        for i in range(seq_len):
            convEncoded = self.conv_encoder_in_decoder(prev_pred)
            tempTensor = torch.empty(conv_outDim+x_dim+z_dim).to(device)
            tempTensor[:conv_outDim] = convEncoded
            tempTensor[conv_outDim:-x_dim] = self.sample_z(z_mu[i], z_log[i])
            tempTensor[-x_dim:] = theta
            
            if encoded_states is None:
                encoded_states = tempTensor
                convEncoded = torch.unsqueeze(tempTensor, 0)
            else:
                encoded_states = torch.vstack((encoded_states, tempTensor))
                convEncoded = encoded_states            
        
    #         print(convEncoded.shape)
            # 4+z_dim+x_dim


            output, deconv_encoder_hidden_state = self.decoder_lstm(convEncoded, deconv_encoder_hidden_state)
    #             output = self.fc_conv_de_to_hidden(output[-1])
            # end of convlstm in decoder
    #         print("shape of output: ", output.shape)



            #start of deconv
    #             output = self.fc_deconv_de_to_hidden(output)
            # final image predicted
            outputs = self.deconv(output)
            outputs = outputs.unsqueeze(1)            
            # update prev_pred to the prediction
            prev_pred = outputs[-1]
            
        return outputs
        

    def forward(self, x_t, x_c, y_c, x_ct, y_ct):
        """
        """

        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(x_ct, y_ct)
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
#         print("shape of z_mu_all: ", self.z_mu_all.shape)
        outputs = []
        for target in x_t:
            output = self.decoder(target, self.z_mu_all, self.z_logvar_all)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
#         self.zs = self.sample_z(self.z_mu_all, self.z_logvar_all)
#         print("shape of zs: ", self.zs.shape)
        return outputs
    


# In[19]:


# all good, no additional modification needed
def random_split_context_target(x,y, n_context):
    """Helper function to split randomly into context and target"""
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)

def sample_z(mu, logvar,n=1):
    """Reparameterisation trick."""
    if n == 1:
        eps = torch.autograd.Variable(logvar.data.new(z_dim).normal_())
    else:
        eps = torch.autograd.Variable(logvar.data.new(n,z_dim).normal_())
    
    std = 0.1+ 0.9*torch.sigmoid(logvar)
#     print(mu + std * eps)
    return mu + std * eps

def data_to_z_params(x, y, calc_score = False):
    """Helper to batch together some steps of the process."""
    rs_all = None
    for i,theta_seq in enumerate(y):
        if calc_score:
            theta_seq = torch.cat([decoder_init, theta_seq])
        y_stacked = dcrnn.stack_y(theta_seq)
        y_conv_c = dcrnn.conv_encoder(y_stacked)
        encode_hidden_state = None
#             print(y_conv_c.shape)
#         print("shape of y after conv layer: ", y_conv_c.shape)
        # corresponding theta to current y: x[i]
        xy = torch.cat([y_conv_c, x[i].repeat(len(y_stacked)).reshape(-1,x_dim)], dim=1)
#             print("shape of xy: ", xy.shape)
        rs , encode_hidden_state = dcrnn.encoder_lstm(xy, encode_hidden_state)
#             self.lstm_linear(rs)
        rs = rs.unsqueeze(0)
#             print("shape of rs: ", rs.shape)
        if rs_all is None:
            rs_all = rs
        else:
            rs_all = torch.vstack((rs_all, rs))

#         print("shape of rs_all: ", rs_all.shape)
    r_agg = rs_all.mean(dim=0) # Average over samples
#         print("shape of r_agg: ", r_agg.shape)
    return dcrnn.z_encoder(r_agg) # Get mean and variance for q(z|...)

def test(x_train, y_train, x_test):
    with torch.no_grad():
        z_mu, z_logvar = data_to_z_params(x_train.to(device),y_train.to(device))
      
        output_list = None
        for i in range (len(x_test)):
        #           zsamples = sample_z(z_mu, z_logvar) 
            output = dcrnn.decoder(x_test[i:i+1].to(device), z_mu, z_logvar).cpu().unsqueeze(0)
            if output_list is None:
                output_list = output.detach()
            else:
                output_list = torch.vstack((output_list, output.detach()))
    
    return output_list.numpy()


# In[84]:


def train(n_epochs, x_train, y_train, x_val, y_val, x_test, y_test, n_display=500, patience = 5000): #7000, 1000
    train_losses = []
    mae_losses = []
    kld_losses = []
    val_losses = []
    test_losses = []

    means_test = []
    stds_test = []
    min_loss = 0. # for early stopping
    wait = 0
    min_loss = float('inf')
    dcrnn.train()
    print(int(n_epochs / 1000))
    for t in range(int(n_epochs / 1000)): 
        opt.zero_grad()
        #Generate data and process
        x_context, y_context, x_target, y_target = random_split_context_target(
                                x_train, y_train, int(len(y_train)*0.2))

        x_c = torch.from_numpy(x_context).float().to(device)
        x_t = torch.from_numpy(x_target).float().to(device)
        y_c = torch.from_numpy(y_context).float().to(device)
        y_t = torch.from_numpy(y_target).float().to(device)

        x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
        y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

        y_pred = dcrnn(x_t, x_c, y_c, x_ct, y_ct)
        
#         print("shape of y_pred: ", y_pred.shape)
#         print("shape of y_t: ", y_t.shape)

        train_loss = MAE(y_pred, y_t) + dcrnn.KLD_gaussian()
        mae_loss = MAE(y_pred, y_t)
        kld_loss = dcrnn.KLD_gaussian()
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(dcrnn.parameters(), 5) #10
        opt.step()
        
        #val loss
        y_val_pred = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_val).float())
#         print("shape of y_val_pred: ", y_val_pred.shape)
#         print("shape of y_val: ", y_val.shape)
        val_loss = MAE(torch.from_numpy(y_val_pred).float(),torch.from_numpy(y_val).float())
        #test loss
        y_test_pred = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_test).float())
        test_loss = MAE(torch.from_numpy(y_test_pred).float(),torch.from_numpy(y_test).float())

        if t % n_display ==0:
            print('train loss:', train_loss.item(), 'mae:', mae_loss.item(), 'kld:', kld_loss.item(), flush=True)
            print('val loss:', val_loss.item(), 'test loss:', test_loss.item(), flush=True)
            ypred_allset.append(y_pred)
#             print(y_train)

        if t % (n_display/10) ==0:
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            test_losses.append(test_loss.item())
#             mae_losses.append(mae_loss.item())
#             kld_losses.append(kld_loss.item())
        
#         if train_loss.item() < 10:
#             return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all
        
#         #early stopping
        if val_loss < min_loss:
            wait = 0
            min_loss = val_loss
            
        elif val_loss >= min_loss:
            wait += 1
            if wait == patience:
                print('Early stopping at epoch: %d' % t)
                return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all
        
    return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all

def train_botorch(n_epochs, x_train, y_train, x_val, y_val, x_test, y_test, n_display=500, patience=5000, bo_iter=10):
    train_losses = []
    mae_losses = []
    kld_losses = []
    val_losses = []
    test_losses = []

    means_test = []
    stds_test = []
    min_loss = float('inf')
    wait = 0
    dcrnn.train()

    # Set up bounds for learning rate optimization
    lr_bounds = torch.tensor([[1e-4], [1e-2]], dtype=torch.float64)
    train_x = []
    train_y = []

    print(f"Bayesian Optimization Iterations: {bo_iter}")

    for bo_itr in range(bo_iter):
        print(f"BoTorch iteration {bo_itr + 1}/{bo_iter}")

        # First BO iteration switch: Random initialization
        if bo_itr == 0:
            current_lr = random.uniform(lr_bounds[0].item(), lr_bounds[1].item())
        else:
            # Train GP model with BoTorch
            gp_train_x = torch.tensor(train_x, dtype=torch.float64)
            gp_train_y = torch.tensor(train_y, dtype=torch.float64).unsqueeze(-1)

            x_mean = gp_train_x.mean(0)
            x_std = gp_train_x.std(0) if gp_train_x.size(0) > 1 else torch.ones_like(x_mean)

            y_mean = gp_train_y.mean()
            y_std = gp_train_y.std() if gp_train_y.size(0) > 1 else torch.ones_like(y_mean)

            # Standardize
            gp_train_x = (gp_train_x - x_mean) / x_std
            gp_train_y = (gp_train_y - y_mean) / y_std

            gp = SingleTaskGP(gp_train_x, gp_train_y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll.train()

            # Optimizer
            optimizer_gp = torch.optim.Adam(gp.parameters(), lr=0.1)
            optimizer_gp.zero_grad()
            output = gp(gp_train_x)
            loss = -mll(output, gp_train_y)
            loss.backward(torch.ones_like(loss))
            optimizer_gp.step()

            ei = ExpectedImprovement(model=gp, best_f=gp_train_y.max())
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=lr_bounds,
                q=1,
                num_restarts=5,
                raw_samples=20
            )
            current_lr = candidate.item()

        # Update optimizer with the BoTorch learning rate
        opt = torch.optim.Adam(dcrnn.parameters(), lr=current_lr)

        # Adjust the number of epochs per BO iteration
        bo_epochs = int(n_epochs / bo_iter)
        print(f"Training for {bo_epochs} epochs with learning rate {current_lr:.6f}")

        for t in range(bo_epochs):
            opt.zero_grad()

            # Generate data and process
            x_context, y_context, x_target, y_target = random_split_context_target(
                x_train, y_train, int(len(y_train) * 0.2)
            )

            x_c = torch.from_numpy(x_context).float().to(device)
            x_t = torch.from_numpy(x_target).float().to(device)
            y_c = torch.from_numpy(y_context).float().to(device)
            y_t = torch.from_numpy(y_target).float().to(device)
            x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
            y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

            y_pred = dcrnn(x_t, x_c, y_c, x_ct, y_ct)

            # Compute losses
            train_loss = MAE(y_pred, y_t) + dcrnn.KLD_gaussian()
            mae_loss = MAE(y_pred, y_t)
            kld_loss = dcrnn.KLD_gaussian()

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(dcrnn.parameters(), 5)
            opt.step()

            # Validation loss
            y_val_pred = test(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).float(),
                torch.from_numpy(x_val).float()
            )
            val_loss = MAE(torch.from_numpy(y_val_pred).float(), torch.from_numpy(y_val).float())

            # Test loss
            y_test_pred = test(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).float(),
                torch.from_numpy(x_test).float()
            )
            test_loss = MAE(torch.from_numpy(y_test_pred).float(), torch.from_numpy(y_test).float())

            # Logging
            if t % n_display == 0:
                print('train loss:', train_loss.item(), 'mae:', mae_loss.item(), 'kld:', kld_loss.item(), flush=True)
                print('val loss:', val_loss.item(), 'test loss:', test_loss.item(), flush=True)

            if t % (n_display / 10) == 0:
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                test_losses.append(test_loss.item())
                mae_losses.append(mae_loss.item())
                kld_losses.append(kld_loss.item())

            # Early stopping
            if val_loss < min_loss:
                wait = 0
                min_loss = val_loss
            elif val_loss >= min_loss:
                wait += 1
                if wait == patience:
                    print('Early stopping at epoch: %d' % t)
                    return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all

        # Log the results for BO
        train_x.append([current_lr])
        train_y.append(val_loss.item())

    return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all

def generate_batch(x,y, batch_size):
    """Helper function to split randomly into context and target"""
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=batch_size, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)


# In[86]:


# initialize search_data_x to the dataset with x_init removed
def calculate_score(x_train, y_train, x_search):
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_search = torch.from_numpy(x_search).float()
    dcrnn.eval()

    # query z_mu, z_var of the current training data
    with torch.no_grad():
        z_mu, z_logvar = data_to_z_params(x_train.to(device),y_train.to(device))
        
        output_list = []
        for theta in x_search:
            output = dcrnn.decoder(theta, z_mu, z_logvar)
#             print("shape of output in calculating score: ", output.shape)
            output_list.append(output)
        outputs = torch.stack(output_list, dim=0)

        y_search = outputs.squeeze(2)

        x_search_all = torch.cat([x_train.to(device),x_search.to(device)],dim=0)
        y_search_all = torch.cat([y_train[:,1:,...].to(device),y_search],dim=0)
        
#         print("shape of y_search_all: ", y_search_all.shape)
#         print("shape of x_search_all: ", x_search_all.shape)

        # generate z_mu_search, z_var_search
        z_mu_search, z_logvar_search = data_to_z_params(x_search_all.to(device),y_search_all.to(device), calc_score = True)

        # calculate and save kld
        mu_q, var_q, mu_p, var_p = z_mu_search,  0.1+ 0.9*torch.sigmoid(z_logvar_search), z_mu, 0.1+ 0.9*torch.sigmoid(z_logvar)

        std_q = torch.sqrt(var_q)
        std_p = torch.sqrt(var_p)

        p = torch.distributions.Normal(mu_p, std_p)
        q = torch.distributions.Normal(mu_q, std_q)
        score = torch.distributions.kl_divergence(q, p).sum()


    return score




# BO search:

# In[87]:


# TODO: replace np.linespace with our correct ones for reaction diffusion data
def mae_plot(mae, selected_mask):
    epsilon, beta  = np.meshgrid(np.linspace(0.25, 0.7, 10), np.linspace(1.1, 4.1, 31))
    selected_mask = selected_mask.reshape(30,9)
    mae_min, mae_max = 0, 1200

    fig, ax = plt.subplots(figsize=(16, 5))
    # f, (y1_ax) = plt.subplots(1, 1, figsize=(16, 10))

    c = ax.pcolormesh(beta-0.05, epsilon-0.025, mae, cmap='binary', vmin=mae_min, vmax=mae_max)
    ax.set_title('MAE Mesh')
    # set the limits of the plot to the limits of the data
    ax.axis([beta.min()-0.05, beta.max()-0.05, epsilon.min()-0.025, epsilon.max()-0.025])
    x,y = np.where(selected_mask==1)
    x = x*0.1+1.1
    y = y*0.05+0.25
    ax.plot(x, y, 'r*', markersize=15)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Epsilon')
    plt.show()

def score_plot(score, selected_mask):
    epsilon, beta  = np.meshgrid(np.linspace(0.25, 0.7, 10), np.linspace(1.1, 4.1, 31))
    score_min, score_max = 0, 1
    selected_mask = selected_mask.reshape(30,9)
    score = score.reshape(30,9)
    fig, ax = plt.subplots(figsize=(16, 5))
    # f, (y1_ax) = plt.subplots(1, 1, figsize=(16, 10))

    c = ax.pcolormesh(beta-0.05, epsilon-0.025, score, cmap='binary', vmin=score_min, vmax=score_max)
    ax.set_title('Score Mesh')
    # set the limits of the plot to the limits of the data
    ax.axis([beta.min()-0.05, beta.max()-0.05, epsilon.min()-0.025, epsilon.max()-0.025])
    x,y = np.where(selected_mask==1)
    x = x*0.1+1.1
    y = y*0.05+0.25
    ax.plot(x, y, 'r*', markersize=15)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Epsilon')
    plt.show()




# In[88]:


def MAE_MX(y_pred, y_test):
    y_pred = y_pred.reshape(160, 5, 2, 32, 32)
    y_test = y_test[:,1:,...].reshape(160, 5, 2, 32, 32)
    mae_matrix = np.mean(np.abs(y_pred - y_test),axis=(2,3,4))
    mae = np.mean(np.abs(y_pred - y_test))
    return mae_matrix, mae



# In[90]:


r_dim = 64
z_dim = 64 #8
x_dim = 2 #
y_dim = 2 


ypred_allset = []
ypred_testset = []
mae_allset = []
maemetrix_allset = []
mae_testset = []
score_set = []
# save the value for all y_train
yall_set = np.array(y_all)
print(yall_set.shape)

decoder_init = torch.unsqueeze(torch.from_numpy(np.load("../data/initial_pic.npy")).float().to(device), 0)
dcrnn = DCRNNModel(x_dim, y_dim, r_dim, z_dim).to(device)
opt = torch.optim.Adam(dcrnn.parameters(), 1e-3) #1e-3

y_pred_test_list = []
y_pred_all_list = []
all_mae_matrix_list = []
all_mae_list = []
test_mae_list = []
score_list = []
all_rewards = []
# get initial choices of data
# batch size is 8
x_train,y_train, search_data_x, search_data_y = generate_batch(x_all, y_all, 5)
print(x_train)
np_model = DCRNNModel(x_dim=x_dim, y_dim=y_dim, r_dim=r_dim, z_dim=z_dim).to(device)
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
initial_scenarios = len(search_data_x)
data_percentage = []


for i in range(10): #8
    dcrnn.train()
    print('training data shape:', x_train.shape, y_train.shape, flush=True)

    train_losses, val_losses, test_losses, z_mu, z_logvar = train_botorch(20,x_train,y_train,x_val, y_val, x_test, y_test,500, 1500) #originally 20000, 5000
    y_pred_test = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_test).float())
#     print("y_pred_shape: ", y_pred_test.shape)
    y_pred_test_list.append(y_pred_test)


    test_mae = MAE(torch.from_numpy(y_pred_test).float(),torch.from_numpy(y_test).float())
    test_mae_list.append(test_mae.item())
    print('Test MAE:',test_mae.item(), flush=True)

    y_pred_all = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_all).float())
#     print("shape of y_pred_all: ", y_pred_all.shape)
    y_pred_all_list.append(y_pred_all)
    mae_matrix, mae = MAE_MX(y_pred_all, y_all)


    all_mae_matrix_list.append(mae_matrix)
    all_mae_list.append(mae)
    print('All MAE:',mae, flush=True)

    
    reward_list = []
    index_list = []
    for k in range(int(len(search_data_x))):
        index = np.random.choice(len(search_data_x), 5, replace=False).tolist()
        index_list.append(index)
        search_data_x_batch = np.stack([search_data_x[i] for i in index],0)
#         print("shape of search_data_y_all: ", search_data_y_all.shape)
        reward = calculate_score(x_train, y_train, search_data_x_batch)
        reward_list.append(reward.item())

#     print('reward_list:',reward_list)
    selected_ind = np.argmax(np.array(reward_list))
    x_train = np.concatenate((x_train, [search_data_x[i] for i in index_list[selected_ind]]), axis=0)
    y_train = np.concatenate((y_train, [search_data_y[i] for i in index_list[selected_ind]]), axis=0)
    
    search_data_x = [e for i, e in enumerate(search_data_x) if i not in index_list[selected_ind]]
    search_data_y = [e for i, e in enumerate(search_data_y) if i not in index_list[selected_ind]]
    remaining_scenarios = len(search_data_x)
    data_used = (1 - remaining_scenarios / initial_scenarios) * 100
    data_percentage.append(data_used)

    print(f'Remaining scenarios: {remaining_scenarios}, Data used: {data_used:.2f}%', flush=True) 
    max_reward = max(reward_list)
    all_rewards.append(max_reward) 



y_pred_all_arr = np.stack(y_pred_all_list,0)
y_pred_test_arr = np.stack(y_pred_test_list,0)
all_mae_matrix_arr = np.stack(all_mae_matrix_list,0)
all_mae_arr = np.stack(all_mae_list,0)
test_mae_arr = np.stack(test_mae_list,0)
# score_arr = np.stack(score_list,0)

ypred_allset.append(y_pred_all_arr)
ypred_testset.append(y_pred_test_arr)
maemetrix_allset.append(all_mae_matrix_arr)
mae_allset.append(all_mae_arr)
mae_testset.append(test_mae_arr)
np.save("data/final_ypred_testset_final_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.npy" % seed, np.array(ypred_testset))
np.save("data/final_maemetrix_allset_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.npy" % seed, np.array(maemetrix_allset))
np.save("data/final_mae_allset_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.npy" % seed, np.array(mae_allset))
np.save("data/final_mae_testset_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.npy" % seed, np.array(mae_testset))
torch.save(dcrnn.state_dict(), "data/final_5_seq_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.pt" % seed)

np.save('data/test_mae_list.npy', np.array(test_mae_list))
np.save('data/all_rewards.npy', np.array(all_rewards))
np.save('data/all_mae_list.npy', np.array(all_mae_list))

plt.figure()
plt.plot(range(len(test_mae_list)), test_mae_list, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Test MAE')
plt.title('Test MAE over Iterations')
plt.savefig('graphs/test_mae_over_iterations.png')
plt.close()

plt.figure()
plt.plot(range(len(all_rewards)), all_rewards, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Max Reward')
plt.title('Max Reward over Iterations')
plt.savefig('graphs/max_reward_over_iterations.png')
plt.close()

plt.figure()
plt.plot(range(len(all_mae_list)), all_mae_list, marker='o')
plt.xlabel('Iteration')
plt.ylabel('All MAE')
plt.title('All MAE over Iterations')
plt.savefig('graphs/all_mae_over_iterations.png')
plt.close()

plt.figure()
plt.plot(data_percentage, test_mae_list, marker='o')
plt.xlabel('Percentage of Data')
plt.ylabel('Test MAE')
plt.title('Test MAE over Percentage of Data')
plt.savefig('graphs/test_mae_over_percentage_data.png')
plt.close()

plt.figure()
plt.plot(data_percentage, all_rewards, marker='o')
plt.xlabel('Percentage of Data')
plt.ylabel('Max Reward')
plt.title('Max Reward over Percentage of Data')
plt.savefig('graphs/max_reward_over_percentage_data.png')
plt.close()

plt.figure()
plt.plot(data_percentage, all_mae_list, marker='o')
plt.xlabel('Percentage of Data')
plt.ylabel('All MAE')
plt.title('All MAE over Percentage of Data')
plt.savefig('graphs/all_mae_over_percentage_data.png')
plt.close()

print('training finished, dicts saved')
