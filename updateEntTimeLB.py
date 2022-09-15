import numpy as np


def calEndTimeLB(finish_time, durMat, ltMat):
    lb_completion = np.copy(finish_time)
    for job in range(finish_time.shape[0]):
        for op in range(finish_time.shape[1]):
            if op != 0 and finish_time[job, op] == 0:  # not the first operation, unscheduled
                lb_completion[job, op] = lb_completion[job, op - 1] + ltMat[job, op - 1] + durMat[job, op]
            elif op == 0 and finish_time[job, op] == 0:  # first operation, unscheduled
                lb_completion[job, op] = durMat[job, op]

    return lb_completion


if __name__ == '__main__':
    dur = np.array([[1, 2], [3, 4]])
    lt = np.array([[5, 2], [4, 7]])
    finish_time = np.zeros_like(dur)

    ret = calEndTimeLB(finish_time, dur, lt)