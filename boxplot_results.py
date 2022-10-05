import matplotlib.pyplot as plt
import numpy as np
import pickle


def boxplot_makespan(results_rules, results_agent, rule_set, test, j, m):
    # loading dataset
    data = []
    for rule in rule_set:
        data.append(np.array(results_rules[(rule, 'c_max')]))
    data.append(np.array(results_agent['c_max']))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.boxplot(data, notch=True, vert=False)
    labels = rule_set + ['agent']
    ax.set_yticklabels(labels)
    plt.xlabel("Makespan")
    if test:
        plt.title(f"Box plot of makespan on 1000 instances of size {j}x{m}")
    else:
        plt.title(f"Box plot of makespan on 100 instances of size {j}x{m}")
    plt.show()


if __name__ == '__main__':
    n_j = 15
    n_m = 15
    run_type = 'L2D'
    np_seed_val = 200
    np_seed_test = 100
    test_set = True
    rules = ['mor', 'mwr', 'edd', 'rand']

    if test_set:
        with open('./dispatching_rule_results/test_set_' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_Seed' +
                  str(np_seed_test) + '.pkl', 'rb') as f:
            loaded_dict_rules = pickle.load(f)
            # print(loaded_dict_rules)
        with open('./agent_results/test_set_' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_Seed' +
                  str(np_seed_test) + '.pkl', 'rb') as f:
            loaded_dict_agent = pickle.load(f)
            print(loaded_dict_agent)
    else:
        with open('./dispatching_rule_results/' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_Seed' +
                  str(np_seed_val) + '.pkl', 'rb') as f:
            loaded_dict_rules = pickle.load(f)
            print(loaded_dict_rules)
        with open('./agent_results/' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_Seed' +
                  str(np_seed_val) + '.pkl', 'rb') as f:
            loaded_dict_agent = pickle.load(f)
            print(loaded_dict_agent)

    boxplot_makespan(loaded_dict_rules, loaded_dict_agent, rules, test_set, n_j, n_m)
