



import sympy
import torch
import argparse
import yaml
import random
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

from lib import utils
from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    print(torch.__version__)
    print(torch.cuda.is_available())  
    print(torch.version.cuda)  
    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        
        i = 1
        np.random.seed(i)
        random.seed(i)
        torch.manual_seed(i)
        
        max_itr = 12
        data, search_data_x, search_data_y = utils.load_dataset(**supervisor_config.get('data'))
        supervisor = DCRNNSupervisor(random_seed=i, iteration=0, max_itr=max_itr, 
                                     adj_mx=adj_mx, device=device, **supervisor_config)

        if not os.path.exists('seed%d/reward_list' % (i)):
            os.makedirs('seed%d/reward_list' % (i))
        if not os.path.exists('seed%d/index_list' % (i)):
            os.makedirs('seed%d/index_list' % (i))

        for itr in range(max_itr):
            supervisor.iteration = itr
            supervisor._data = data
            supervisor.train()

            reward_list = []
            index_list = []
            for k in range(int(len(search_data_x))):
                index = np.random.choice(len(search_data_x), 8, replace=False).tolist()
                index_list.append(index)
                search_data_x_all = np.concatenate([search_data_x[i] for i in index], 0)
                search_data_y_all = np.concatenate([search_data_y[i] for i in index], 0)

                # Reshape and scale data
                n_samples = search_data_x_all.shape[0]
                n_features = np.prod(search_data_x_all.shape[1:])

                # Flatten search_data_x_all for scaling
                search_data_x_all_flat = search_data_x_all.reshape(n_samples, n_features)

                # MinMax scaling
                scaler_x = MinMaxScaler()
                search_data_x_all_scaled = scaler_x.fit_transform(search_data_x_all_flat)

                # Standard scaling
                standard_scaler = StandardScaler()
                search_data_x_all_standardized = standard_scaler.fit_transform(search_data_x_all_scaled)

                # Convert to tensors
                train_x = torch.tensor(search_data_x_all_standardized, dtype=torch.float64).to(device)
                train_y = torch.tensor(search_data_y_all, dtype=torch.float64).unsqueeze(-1).reshape(-1, 1).to(device)

                # Debugging shapes
                print(f"Shape of train_x: {train_x.shape}")
                print(f"Shape of train_y: {train_y.shape}")

                # BoTorch model/acquisition function
                gp_model = SingleTaskGP(train_x, train_y).to(device)
                mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(device)

                # Adam Optimization
                optimizer = Adam([{'params': gp_model.parameters()}], lr=0.1)
                
                def train_model(num_steps=100):
                    gp_model.train()
                    mll.train()
                    
                    for i in range(num_steps):
                        optimizer.zero_grad()
                        output = gp_model(train_x)
                        loss = -mll(output, train_y)
                        
                        # Ensure the loss is a scalar before backward
                        loss = loss.mean()  # Mean reduction
                        loss.backward()
                        
                        optimizer.step()
                        print(f"Step {i+1}/{num_steps}, Loss: {loss.item()}")

                train_model()

                ei = ExpectedImprovement(gp_model, best_f=train_y.max())
                bounds = torch.stack([torch.zeros(train_x.shape[-1]), torch.ones(train_x.shape[-1])]).to(device)
                
                candidate, acq_value = optimize_acqf(
                    acq_function=ei,
                    bounds=bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
                )
                
                reward = acq_value.item()
                reward_list.append(reward)

            np.save('seed%d/reward_list/itr%d.npy' % (i, itr+1), np.array(reward_list))
            np.save('seed%d/index_list/itr%d.npy' % (i, itr+1), np.stack(index_list))

            selected_ind = np.argmax(np.array(reward_list))
            selected_data_x = [search_data_x[i] for i in index_list[selected_ind]]
            selected_data_y = [search_data_y[i] for i in index_list[selected_ind]]

            selected_data = {'x': selected_data_x, 'y': selected_data_y}
            search_config = supervisor_config.get('data').copy()
            search_config['selected_data'] = selected_data
            search_config['previous_data'] = data

            data = utils.generate_new_trainset(**search_config)

            search_data_x = [e for i, e in enumerate(search_data_x) if i not in index_list[selected_ind]]
            search_data_y = [e for i, e in enumerate(search_data_y) if i not in index_list[selected_ind]]
            print('remained scenarios:', len(search_data_x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_cov.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)







