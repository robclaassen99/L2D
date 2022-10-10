import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


def boxplot_makespan(results_rules, results_agent, rule_set, test, j, m, t):
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
        plt.title(f"Box plot of makespan on 1000 instances of size {j}x{m}x{t}")
    else:
        plt.title(f"Box plot of makespan on instances of size {j}x{m}x{t}")

    if save:
        plt.savefig(
            './experiment_plots/boxplot_makespan_{}_{}_{}_{}_{}_{}_{}.png'.format(str(j), str(m), str(t), str(low),
                                                                                  str(high), str(lt_low), str(lt_high)))
    if show:
        plt.show()


def boxplot_diff_to_opt(results_rules, results_agent, opt_results, rule_set, test, j, m, t):
    data = []
    for rule in rule_set:
        rel_rule_result = results_rules[(rule, 'c_max')][:len(opt_results)]
        diff = np.array(rel_rule_result) - opt_results
        data.append(diff)
    rel_agent_result = results_agent['c_max'][:len(opt_results)]
    diff = np.array(rel_agent_result) - opt_results
    data.append(diff)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.boxplot(data, notch=True, vert=False)
    labels = rule_set + ['agent']
    ax.set_yticklabels(labels)
    plt.xlabel("Absolute difference with optimal makespan")
    if test:
        plt.title(f"Box plot of difference with optimal makespan on 1000 instances of size {j}x{m}x{t}")
    else:
        plt.title(f"Box plot of difference with optimal makespan on instances of size {j}x{m}x{t}")

    if save:
        plt.savefig('./experiment_plots/boxplot_diff_to_opt_{}_{}_{}_{}_{}_{}_{}.png'.format(str(j), str(m), str(t),
                                                                                             str(low), str(high),
                                                                                             str(lt_low), str(lt_high)))
    if show:
        plt.show()


def boxplot_gap_to_opt(results_rules, results_agent, opt_results, rule_set, test, j, m, t):
    data = []
    for rule in rule_set:
        rel_rule_result = results_rules[(rule, 'c_max')][:len(opt_results)]
        gap = (np.array(rel_rule_result) - opt_results) / opt_results
        data.append(gap)
    rel_agent_result = results_agent['c_max'][:len(opt_results)]
    gap = (np.array(rel_agent_result) - opt_results) / opt_results
    data.append(gap)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.boxplot(data, notch=True, vert=False)
    labels = rule_set + ['agent']
    ax.set_yticklabels(labels)
    ax.set_xlim(left=-0.05)
    plt.xlabel("Gap to optimal makespan")
    if test:
        plt.title(f"Box plot of gap to optimal makespan on 1000 instances of size {j}x{m}x{t}")
    else:
        plt.title(f"Box plot of gap to optimal makespan on instances of size {j}x{m}x{t}")

    if save:
        plt.savefig('./experiment_plots/boxplot_gap_to_opt_{}_{}_{}_{}_{}_{}_{}.png'.format(str(j), str(m), str(t),
                                                                                            str(low), str(high),
                                                                                            str(lt_low), str(lt_high)))
    if show:
        plt.show()


if __name__ == '__main__':
    show = False
    save = True

    n_j = 30
    n_m = 10
    n_t = 20
    low = 1
    high = 99
    lt_low = 1
    lt_high = 99
    run_type = 'L2D-LeadTime_Loading_VRL'
    np_seed_val = 200
    np_seed_test = 100
    test_set = False
    rules = ['mtr', 'mwr', 'edd', 'rand']

    sns.set_style("darkgrid", {'axes.grid': True,
                               'axes.edgecolor': 'black',
                               'grid.color': '.6',
                               'grid.linestyle': ':'
                               })

    if test_set:
        with open('./dispatching_rule_results/test_set_' + str(run_type) + '_' + str(n_j) + '_' + str(n_m)
                  + '_' + str(n_t) + '_Seed' + str(np_seed_test) + '.pkl', 'rb') as f:
            loaded_dict_rules = pickle.load(f)
            # print(loaded_dict_rules)
        with open('./agent_results/test_set_' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t)
                  + '_Seed' + str(np_seed_test) + '.pkl', 'rb') as f:
            loaded_dict_agent = pickle.load(f)
            # print(loaded_dict_agent)
        with open(f'./optimal_results/test_set_{run_type}_optimal_{n_j}_{n_m}_{n_t}_{low}_{high}_{lt_low}_{lt_high}.txt') as f:
            lines = f.readline()
            loaded_results = lines.strip('][').split(', ')
            opt_results = np.array([float(res) for res in loaded_results])
        print(np.sum(opt_results) / opt_results.shape[0])

    else:
        with open('./dispatching_rule_results/' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t)
                  + '_Seed' + str(np_seed_val) + '.pkl', 'rb') as f:
            loaded_dict_rules = pickle.load(f)
            # print(loaded_dict_rules)
        with open('./agent_results/' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t) + '_Seed' +
                  str(np_seed_val) + '.pkl', 'rb') as f:
            loaded_dict_agent = pickle.load(f)
            # print(loaded_dict_agent)

        with open(f'./optimal_results/{run_type}_optimal_{n_j}_{n_m}_{n_t}_{low}_{high}_{lt_low}_{lt_high}.txt') as f:
            lines = f.readline()
            loaded_results = lines.strip('][').split(', ')
            opt_results = np.array([float(res) for res in loaded_results])

    boxplot_makespan(loaded_dict_rules, loaded_dict_agent, rules, test_set, n_j, n_m, n_t)
    boxplot_diff_to_opt(loaded_dict_rules, loaded_dict_agent, opt_results, rules, test_set, n_j, n_m, n_t)
    boxplot_gap_to_opt(loaded_dict_rules, loaded_dict_agent, opt_results, rules, test_set, n_j, n_m, n_t)