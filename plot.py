import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import re
from random import sample

# plot parameters
x_label_scale = 15
y_label_scale = 15
anchor_text_size = 15
show = True
save = False
save_file_type = '.pdf'
# problem params
n_j = 15
n_m = 15
l = 1
h = 99
stride = 50
datatype = 'vali'  # 'vali', 'log'

if __name__ == '__main__':
    f = open('./run_results/{}s/{}_{}_{}_{}_{}.txt'.format(datatype, datatype, n_j, n_m, l, h), 'r').readline()
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

