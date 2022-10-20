import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd


def plot_log(input_data, n_j, n_m, n_t, smooth_factor, save, show):
    data_list = [- float(s) for s in re.findall(r'-?\d+\.?\d*', input_data)[1::2]][:]
    sns.set_style("darkgrid", {'axes.grid': True,
                               'axes.edgecolor': 'black',
                               'grid.color': '.6',
                               'grid.linestyle': ':'
                               })
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    data_smooth = pd.Series(data_list).rolling(smooth_factor, min_periods=1).mean()
    sns.lineplot(data=data_list, ci=95, label='Raw makespan', alpha=0.2)
    sns.lineplot(data=data_smooth, label='Smoothened makespan')
    plt.xlim([-100, len(data_list) + 100])
    plt.ylim([data_smooth.min() - 10, data_smooth.max() + 10])
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Makespan', fontsize=12)
    lgd = plt.legend(frameon=True, fancybox=True, prop={'weight': 'bold', 'size': 10}, loc="best")
    plt.title(f'Average makespan during training ({n_j} x {n_m} x {n_t})', fontsize=13)
    ax = plt.gca()

    plt.setp(ax.get_xticklabels(), fontsize=11)
    plt.setp(ax.get_yticklabels(), fontsize=11)
    sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig(
            './experiment_plots/{}_{}_{}_{}_{}_{}_{}_{}_{}.png'.format(str(run_type), str(datatype), str(n_j), str(n_m),
                                                                       str(n_t), str(l), str(h), str(lt_l), str(lt_h)))
    if show:
        plt.show()


def plot_val(input_data, n_j, n_m, n_t, save, show):
    data_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', input_data)][:]
    sns.set_style("darkgrid", {'axes.grid': True,
                               'axes.edgecolor': 'black',
                               'grid.color': '.6',
                               'grid.linestyle': ':'
                               })
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    sns.lineplot(data=data_list, ci=95, label='Average makespan')
    plt.xlim([-5, len(data_list) + 5])
    plt.ylim([min(data_list) - 15, max(data_list) + 15])
    plt.xlabel('Validation checkpoint', fontsize=12)
    plt.ylabel('Makespan', fontsize=12)
    lgd = plt.legend(frameon=True, fancybox=True, prop={'weight': 'bold', 'size': 10}, loc="best")
    plt.title(f'Average makespan on validation set ({n_j} x {n_m} x {n_t})', fontsize=13)
    ax = plt.gca()

    plt.setp(ax.get_xticklabels(), fontsize=11)
    plt.setp(ax.get_yticklabels(), fontsize=11)
    sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig(
            './experiment_plots/{}_{}_{}_{}_{}_{}_{}_{}_{}.png'.format(str(run_type), str(datatype), str(n_j), str(n_m),
                                                                       str(n_t), str(l), str(h), str(lt_l), str(lt_h)))
    if show:
        plt.show()


if __name__ == '__main__':
    # plot parameters
    show = True
    save = False
    # problem params
    n_j = 20
    n_m = 10
    n_t = 15
    l = 1
    h = 99
    lt_l = 1
    lt_h = 99
    smooth_factor = 300
    run_type = "L2D-LeadTime_Loading_VRL_deepPPO"
    datatype = 'vali'  # 'vali', 'log'

    f = open('./run_results/{}s/{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(datatype, run_type, datatype, n_j, n_m, n_t, l,
                                                                       h, lt_l, lt_h),
             'r').readline()
    if datatype == 'log':
        plot_log(f, n_j, n_m, n_t, smooth_factor, save, show)
    elif datatype == 'vali':
        plot_val(f, n_j, n_m, n_t, save, show)


