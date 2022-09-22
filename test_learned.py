from mb_agg import *
from agent_utils import *
import torch
import argparse
from Params import configs
import time
import numpy as np


device = torch.device(configs.device)

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
parser.add_argument('--Pn_j', type=int, default=6, help='Number of jobs of instances to test')
parser.add_argument('--Pn_m', type=int, default=6, help='Number of machines instances to test')
parser.add_argument('--Nn_j', type=int, default=6, help='Number of jobs on which to be loaded net are trained')
parser.add_argument('--Nn_m', type=int, default=6, help='Number of machines on which to be loaded net are trained')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--lt_low', type=int, default=1, help='LB of lead time')
parser.add_argument('--lt_high', type=int, default=99, help='UB of lead time')
parser.add_argument('--run_type', type=str, default="L2D-LeadTime", help='Problem instance type that we run')
parser.add_argument('--shuffle_machines', type=bool, default=True, help='Toggle for permute_rows in machine matrix')
parser.add_argument('--n_test', type=int, default=100, help='Number of instances to test the model on')
parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
params = parser.parse_args()

N_JOBS_P = params.Pn_j
N_MACHINES_P = params.Pn_m
LOW = params.low
HIGH = params.high
LT_LOW = params.lt_low
LT_HIGH = params.lt_high
SHUFFLE_MACHINES = params.shuffle_machines
SEED = params.seed
N_JOBS_N = params.Nn_j
N_MACHINES_N = params.Nn_m
N_TEST = params.n_test


from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO
env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)

ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
          n_j=N_JOBS_P,
          n_m=N_MACHINES_P,
          num_layers=configs.num_layers,
          neighbor_pooling_type=configs.neighbor_pooling_type,
          input_dim=configs.input_dim,
          hidden_dim=configs.hidden_dim,
          num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
          num_mlp_layers_actor=configs.num_mlp_layers_actor,
          hidden_dim_actor=configs.hidden_dim_actor,
          num_mlp_layers_critic=configs.num_mlp_layers_critic,
          hidden_dim_critic=configs.hidden_dim_critic)
path = './SavedNetworkNew/{}.pth'.format(str(params.run_type) + '_' + str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_'
                                         + str(LOW) + '_' + str(HIGH) + '_' + str(LT_LOW) + '_' + str(LT_HIGH))
# ppo.policy.load_state_dict(torch.load(path))
ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
# ppo.policy.eval()
g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                         batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                         n_nodes=env.number_of_tasks,
                         device=device)

from uniform_instance_gen import uni_instance_gen
np.random.seed(SEED)

dataLoaded = np.load('./DataGen/generatedData_' + str(configs.run_type) + '_' + str(configs.n_j) + '_'
                     + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
dataset = []

for i in range(dataLoaded.shape[0]):
    dataset.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2]))

dataset = [uni_instance_gen(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=LOW, high=HIGH, lt_low=LT_LOW, lt_high=LT_HIGH,
                            shuffle_machines=SHUFFLE_MACHINES) for _ in range(N_TEST)]


def test(dataset):
    result = []
    # torch.cuda.synchronize()
    t1 = time.time()
    for i, data in enumerate(dataset):
        adj, fea, candidate, mask = env.reset(data)
        ep_reward = - env.max_endTime
        # delta_t = []
        # t5 = time.time()
        while True:
            # t3 = time.time()
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            # t4 = time.time()
            # delta_t.append(t4 - t3)

            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                   graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)
                action = greedy_select_action(pi, candidate)

            adj, fea, reward, done, candidate, mask = env.step(action)
            ep_reward += reward

            if done:
                break
        # t6 = time.time()
        # print(t6 - t5)
        # print(max(env.end_time))
        print('Instance' + str(i + 1) + ' makespan:', -ep_reward + env.posRewards)
        result.append(-ep_reward + env.posRewards)
        # print(sum(delta_t))
    # torch.cuda.synchronize()
    t2 = time.time()
    print(t2 - t1)
    file_writing_obj = open('./' + 'drltime_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '.txt', 'w')
    file_writing_obj.write(str((t2 - t1)/len(dataset)))

    # print(result)
    # print(np.array(result, dtype=np.single).mean())
    np.save('drlResult_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '_Seed' + str(SEED), np.array(result, dtype=np.single))
    # print(np.array(result, dtype=np.single).mean())


if __name__ == '__main__':
    import cProfile

    cProfile.run('test(dataset)', filename='restats')