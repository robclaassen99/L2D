from dispatching_rules import *
from uniform_instance_gen import generate_deadlines
from JSSP_Env import SJSSP
import pickle


def test_dd_tightness(data_set, n_j, n_m, n_t):
    env = SJSSP(n_j=n_j, n_m=n_m, n_t=n_t)

    tardiness_per_c = {}
    tightness = 5.0
    while True:
        tightness = round(tightness, 1)
        print(tightness)
        total_num_jobs = 0
        total_tardy_jobs = 0
        for data in data_set:
            # reset JSSP environment
            adj, node_fea, candidates, mask = env.reset(data)
            # compute job deadlines
            job_deadlines = generate_deadlines(dur_flat=env.dur, lt_flat=env.lt, truck_array=env.truck_array,
                                               action_matrix=env.action_matrix, deadline_tightness=tightness)

            while True:
                action = shortest_processing_time(candidates, mask, env.dur)
                adj, fea, reward, done, candidates, mask = env.step(action.item())
                if done:
                    break

            finish_time = np.zeros_like(env.action_matrix)
            finish_time[np.nonzero(env.action_matrix)] = env.finish_time

            job_finish_times = finish_time[np.arange(finish_time.shape[0]),
                                           (finish_time != 0).cumsum(1).argmax(1)].astype(np.int32)
            tardiness = (job_deadlines - job_finish_times)[:env.number_of_jobs]
            total_num_jobs += env.number_of_jobs
            total_tardy_jobs += np.sum(tardiness < 0).item()

        tardiness_per_c[tightness] = abs(0.5 - total_tardy_jobs / total_num_jobs)
        print(total_tardy_jobs / total_num_jobs)
        tightness += 0.1
        if total_tardy_jobs / total_num_jobs < 0.5:
            break

    return tardiness_per_c


def generate_deadline_data(data_set, c, n_j, n_m, n_t):
    env = SJSSP(n_j=n_j, n_m=n_m, n_t=n_t)
    deadline_data_set = []
    for data in data_set:
        _ = env.reset(data)
        job_deadlines = generate_deadlines(dur_flat=env.dur, lt_flat=env.lt, truck_array=env.truck_array,
                                           action_matrix=env.action_matrix, deadline_tightness=c)
        deadline_data_set.append(job_deadlines)
    return np.array(deadline_data_set)


def baseline_performance(data_set, deadline_data_set, n_j, n_m, n_t):
    env = SJSSP(n_j=n_j, n_m=n_m, n_t=n_t)

    results_per_env = {}
    avg_results = {}

    rule_set = ['rand', 'mtr', 'mor', 'mwr', 'lpt', 'spt', 'edd']

    # rollout episode using current model
    for rule in rule_set:
        results_per_env[(rule, 'c_max')] = []
        results_per_env[(rule, 't_frac')] = []
        results_per_env[(rule, 't_max')] = []
        results_per_env[(rule, 't_tot')] = []

        for i, data in enumerate(data_set):
            # reset JSSP environment
            adj, node_fea, candidates, mask = env.reset(data)
            rewards = - env.initQuality
            while True:
                if rule == 'spt':
                    action = shortest_processing_time(candidates, mask, env.dur)
                elif rule == 'lpt':
                    action = largest_processing_time(candidates, mask, env.dur)
                elif rule == 'mwr':
                    action = most_work_remaining(candidates, mask, env)
                elif rule == 'mtr':
                    action = most_time_remaining(candidates, mask, env)
                elif rule == 'mor':
                    action = most_operations_remaining(candidates, mask, env)
                elif rule == 'rand':
                    action = random_selection(candidates, mask)
                elif rule == 'edd':
                    action = earliest_due_date(candidates, mask, env, deadline_data_set[i])
                adj, fea, reward, done, candidates, mask = env.step(action.item())
                rewards += reward

                if done:
                    break

            finish_time = np.zeros_like(env.action_matrix)
            finish_time[np.nonzero(env.action_matrix)] = env.finish_time

            job_finish_times = finish_time[np.arange(finish_time.shape[0]),
                                           (finish_time != 0).cumsum(1).argmax(1)].astype(np.int32)
            tardiness = (deadline_data_set[i] - job_finish_times)[:env.number_of_jobs]

            results_per_env[(rule, 'c_max')].append(abs(rewards - env.posRewards))
            results_per_env[(rule, 't_frac')].append(round(np.sum(tardiness < 0).item() / tardiness.shape[0], 2))
            results_per_env[(rule, 't_max')].append(round(abs(np.amin(tardiness).item()), 2))
            results_per_env[(rule, 't_tot')].append(round(abs(np.sum(tardiness[tardiness < 0]).item()), 2))

        avg_results[(rule, 'c_max')] = round(
            sum(results_per_env[(rule, 'c_max')]) / len(results_per_env[(rule, 'c_max')]), 2)
        avg_results[(rule, 't_frac')] = round(
            sum(results_per_env[(rule, 't_frac')]) / len(results_per_env[(rule, 't_frac')]), 2)
        avg_results[(rule, 't_max')] = round(
            sum(results_per_env[(rule, 't_max')]) / len(results_per_env[(rule, 't_max')]), 2)
        avg_results[(rule, 't_tot')] = round(
            sum(results_per_env[(rule, 't_tot')]) / len(results_per_env[(rule, 't_tot')]), 2)

    return results_per_env, avg_results


if __name__ == '__main__':
    n_j = 200
    n_m = 10
    n_t = 150
    run_type = 'L2D-LeadTime_Loading_VRL'
    np_seed_val = 200
    compute_results = True

    if compute_results:
        dataLoaded = np.load(
            './DataGen/generatedData_' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t) + '_Seed'
            + str(np_seed_val) + '.npy')
        arrayLoaded = np.load(
            './DataGen/generatedArray_' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t) + '_Seed'
            + str(np_seed_val) + '.npy')
        test_data = []
        for i in range(dataLoaded.shape[0]):
            test_data.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2], dataLoaded[i][3], arrayLoaded[i]))

        experiment_c = test_dd_tightness(test_data, n_j, n_m, n_t)
        best_c = min(experiment_c, key=experiment_c.get)  # select c with minimum difference to 0.5

        deadline_data = generate_deadline_data(test_data, best_c, n_j, n_m, n_t)

        performance_per_env, avg_performance = baseline_performance(test_data, deadline_data, n_j,
                                                                    n_m, n_t)  # [vali_data[0]]

        print(avg_performance)

        # writing results to picke file, used for dictionary storage
        with open('./dispatching_rule_results/' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t)
                  + '_Seed' + str(np_seed_val) + '.pkl', 'wb') as f:
            pickle.dump(performance_per_env, f)
    else:
        with open('./dispatching_rule_results/' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t)
                  + '_Seed' + str(np_seed_val) + '.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
            print(loaded_dict)

    temp = 0