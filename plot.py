import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
import re
from random import sample
import pandas as pd


def plot_log(input_data, n_j, n_m, smooth_factor, save, show):
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
    plt.ylim([data_smooth.min() - 5, data_smooth.max() + 5])
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Makespan', fontsize=12)
    lgd = plt.legend(frameon=True, fancybox=True, prop={'weight': 'bold', 'size': 10}, loc="best")
    plt.title(f'Average makespan during training ({n_j} x {n_m})', fontsize=13)
    ax = plt.gca()

    plt.setp(ax.get_xticklabels(), fontsize=11)
    plt.setp(ax.get_yticklabels(), fontsize=11)
    sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig('./experiment_plots/{}_{}_{}_{}_{}_{}.png'.format(str(run_type), str(datatype), str(n_m), str(n_j),
                                                                      str(l), str(h)))
    if show:
        plt.show()


def plot_val(input_data, n_j, n_m, save, show):
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
    plt.title(f'Average makespan on validation set ({n_j} x {n_m})', fontsize=13)
    ax = plt.gca()

    plt.setp(ax.get_xticklabels(), fontsize=11)
    plt.setp(ax.get_yticklabels(), fontsize=11)
    sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig('./experiment_plots/{}_{}_{}_{}_{}_{}.png'.format(str(run_type), str(datatype), str(n_m), str(n_j),
                                                                      str(l), str(h)))
    if show:
        plt.show()


if __name__ == '__main__':
    # plot parameters
    show = False
    save = True
    # problem params
    n_j = 15
    n_m = 15
    l = 1
    h = 99
    smooth_factor = 200
    run_type = "L2D"
    datatype = 'vali'  # 'vali', 'log'

    f = open('./run_results/{}s/{}_{}_{}_{}_{}_{}.txt'.format(datatype, run_type, datatype, n_j, n_m, l, h),
             'r').readline()
    if datatype == 'log':
        plot_log(f, n_j, n_m, smooth_factor, save, show)
    elif datatype == 'vali':
        plot_val(f, n_j, n_m, save, show)


def plot_schedule(instance_number, n_j, n_m, start_times, op_id_on_machine, durMat, actMat):
    """

    :param instance_number: id of the scheduling instance for which we plot the result
    :param n_j: number of jobs in the scheduling instance
    :param n_m: number of machines in the scheduling instance (excluding loading machine)
    :param start_times: numpy array of shape [n_m, n_m + 1], which shows the start times of jobs on each machine
    :param op_id_on_machine: numpy array of shape [n_m, n_m + 1], which shows the ID of operations at each index in each
                             machine's schedule
    :param durMat: numpy array of shape [n_j + n_t, n_m + 1], which shows the duration of every operation
    :param actMat: numpy array of shape [n_j + n_t, n_m + 1], which shows the operation ID + 1 of every operation at
                   the corresponding job-machine intersect
    :return: None
    """

    all_bolors = [k for k, v in pltc.cnames.items()]
    all_colors = list(pltc.CSS4_COLORS.keys())

    fig, sched = plt.subplots()
    sched.set_title(f'Final schedule for instance {instance_number}')
    # sched.set_ylim(0, 10*n_j)
    # sched.set_xlim(0, 450)
    sched.set_xlabel('Time')
    sched.set_ylabel('Machine')
    sched.set_yticks(np.arange(start=5, stop=15 + 10 * (n_m - 1), step=10))
    yticklabels = ['Machine ' + str(i) for i in range(1, n_m + 1)]
    # yticklabels = ['Pers 4', 'Pers 5', 'MC', 'TD', 'TA', 'TO', 'Edelstanz', 'Buigbank', 'Eindverpakken', 'IADP',
    #                'Laden']
    yticklabels.reverse()
    sched.set_yticklabels(yticklabels)
    sched.grid(True)
    colors = sample(all_colors, n_j)
    lower_yaxis = 1
    lbl_pool = []
    for mach in reversed(range(n_m)):
        st = start_times[mach]
        st = st[st >= 0]
        ops = op_id_on_machine[mach]
        ops = ops[ops >= 0]
        jobs = [np.where(actMat == op + 1)[0][0] for op in ops]
        job_names = [f'Job {job}' for job in jobs]
        labels = []
        for job_name in job_names:
            if job_name not in lbl_pool:
                labels.append(job_name)
                lbl_pool.append(job_name)
            else:
                labels.append('_' + job_name)
        facecolors = [colors[job] for job in jobs]
        durs = durMat.flatten()[np.nonzero(durMat.flatten())][ops]
        bars = list(zip(st, durs))
        for bar in range(len(bars)):
            sched.broken_barh([bars[bar]], (lower_yaxis, 9), facecolors=(facecolors[bar]), label=labels[bar])
        lower_yaxis += 10

    sched.legend()

    plt.show()

"""
    if datatype == 'vali':
        obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f)])[:]
        # obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f)[1::2]])[:]
        idx = np.arange(obj.shape[0])
        # plotting...
        plt.title('Makespan on unseen data')
        plt.xlabel('Checkpoint', {'size': x_label_scale})
        plt.ylabel('MakeSpan', {'size': y_label_scale})
        plt.grid()
        plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
        plt.tight_layout()
        plt.legend(fontsize=anchor_text_size)
        if save:
            plt.savefig('./{}{}'.format('message-passing_time', save_file_type))
        if show:
            plt.show()
    elif datatype == 'log':
        obj = numpy.array([- float(s) for s in re.findall(r'-?\d+\.?\d*', f)[1::2]])[:].reshape(-1, stride).mean(axis=-1)
        idx = np.arange(obj.shape[0])
        # plotting...
        plt.title('Makespan during training')
        plt.xlabel('Train step (x50)', {'size': x_label_scale})
        plt.ylabel('Makespan', {'size': y_label_scale})
        plt.grid()
        plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
        plt.tight_layout()
        plt.legend(fontsize=anchor_text_size)
        if save:
            plt.savefig('./{}{}'.format('message-passing_time', save_file_type))
        if show:
            plt.show()
    else:
        print('Wrong datatype.')

"""