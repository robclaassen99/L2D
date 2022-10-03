from mb_agg import *
from agent_utils import *
import torch
import argparse
from Params import configs
from test_dispatching_rules import test_dd_tightness, generate_deadline_data
import numpy as np
import pickle


def test(data_set, deadline_data_set):
    results_per_env = {}
    avg_results = {}

    results_per_env['c_max'] = []
    results_per_env['t_frac'] = []
    results_per_env['t_max'] = []
    results_per_env['t_tot'] = []

    for i, data in enumerate(data_set):
        adj, fea, candidate, mask = env.reset(data)
        ep_reward = - env.max_endTime

        g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                 batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                                 n_nodes=env.number_of_tasks,
                                 device=device)
        while True:
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)

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

        finish_time = np.zeros_like(env.action_matrix)
        finish_time[np.nonzero(env.action_matrix)] = env.finish_time

        job_finish_times = finish_time[np.arange(finish_time.shape[0]),
                                       (finish_time != 0).cumsum(1).argmax(1)].astype(np.int32)
        tardiness = (deadline_data_set[i] - job_finish_times)[:env.number_of_jobs]

        results_per_env['c_max'].append(abs(ep_reward - env.posRewards))
        results_per_env['t_frac'].append(round(np.sum(tardiness < 0).item() / tardiness.shape[0], 2))
        results_per_env['t_max'].append(round(abs(np.amin(tardiness).item()), 2))
        results_per_env['t_tot'].append(round(abs(np.sum(tardiness[tardiness < 0]).item()), 2))

    avg_results['c_max'] = round(sum(results_per_env['c_max']) / len(results_per_env['c_max']), 2)
    avg_results['t_frac'] = round(sum(results_per_env['t_frac']) / len(results_per_env['t_frac']), 2)
    avg_results['t_max'] = round(sum(results_per_env['t_max']) / len(results_per_env['t_max']), 2)
    avg_results['t_tot'] = round(sum(results_per_env['t_tot']) / len(results_per_env['t_tot']), 2)

    print(avg_results)

    return results_per_env, avg_results


if __name__ == '__main__':
    device = torch.device(configs.device)

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=30, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=10, help='Number of machines instances to test')
    parser.add_argument('--Pn_t', type=int, default=20, help='Number of trucks instances to test')
    parser.add_argument('--Nn_j', type=int, default=30, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=10, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--Nn_t', type=int, default=20, help='Number of trucks on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=1, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--lt_low', type=int, default=1, help='LB of lead time')
    parser.add_argument('--lt_high', type=int, default=99, help='UB of lead time')
    parser.add_argument('--run_type', type=str, default='L2D-LeadTime_Loading_VRL',
                        help='Problem instance type that we run')
    parser.add_argument('--shuffle_machines', type=bool, default=True, help='Toggle for permute_rows in machine matrix')
    parser.add_argument('--n_test', type=int, default=100, help='Number of instances to test the model on')
    parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
    params = parser.parse_args()

    N_JOBS_P = params.Pn_j
    N_MACHINES_P = params.Pn_m
    N_TRUCKS_P = params.Pn_t
    LOW = params.low
    HIGH = params.high
    LT_LOW = params.lt_low
    LT_HIGH = params.lt_high
    SEED = params.seed
    N_JOBS_N = params.Nn_j
    N_MACHINES_N = params.Nn_m
    N_TRUCKS_N = params.Nn_t

    from JSSP_Env import SJSSP
    from PPO_jssp_multiInstances import PPO

    env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P, n_t=N_TRUCKS_P)

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_P,
              n_m=N_MACHINES_P,
              n_t=N_TRUCKS_P,
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
                                             + str(N_TRUCKS_N) + '_' + str(LOW) + '_' + str(HIGH) + '_' + str(LT_LOW)
                                             + '_' + str(LT_HIGH))
    # ppo.policy.load_state_dict(torch.load(path))
    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # ppo.policy.eval()

    np.random.seed(SEED)

    dataLoaded = np.load(
        './DataGen/generatedData_' + str(params.run_type) + '_' + str(N_JOBS_P) + '_' + str(N_MACHINES_P) + '_'
        + str(N_TRUCKS_P) + '_Seed' + str(SEED) + '.npy')
    arrayLoaded = np.load(
        './DataGen/generatedArray_' + str(params.run_type) + '_' + str(N_JOBS_P) + '_' + str(N_MACHINES_P) + '_'
        + str(N_TRUCKS_P) + '_Seed' + str(SEED) + '.npy')
    dataset = []
    for i in range(dataLoaded.shape[0]):
        dataset.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2], dataLoaded[i][3], arrayLoaded[i]))

    experiment_c = test_dd_tightness(dataset, N_JOBS_P, N_MACHINES_P, N_TRUCKS_P)
    best_c = min(experiment_c, key=experiment_c.get)  # select c with minimum difference to 0.5

    deadline_data = generate_deadline_data(dataset, best_c, N_JOBS_P, N_MACHINES_P, N_TRUCKS_P)

    print(f'({N_JOBS_P} x {N_MACHINES_P} x {N_TRUCKS_P})')
    performance_per_env, avg_performance = test(dataset, deadline_data)

    # writing results to picke file, used for dictionary storage
    with open('./agent_results/' + str(params.run_type) + '_' + str(N_JOBS_P) + '_' + str(N_MACHINES_P) + '_'
              + str(N_TRUCKS_P) + '_Seed' + str(SEED) + '_' + str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_'
              + str(N_TRUCKS_N) + '.pkl', 'wb') as f:
        pickle.dump(performance_per_env, f)