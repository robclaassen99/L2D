import numpy as np


def calEndTimeLB(finish_time, dur_flat, lt_flat, actMat, truck_array, n_j, n_t, nbo, last_col, pss):
    lb_completion = np.copy(finish_time)
    # loop over jobs, excluding loading operations
    for job in range(n_j):
        # get the ID of all operations belonging to job
        # -1, because indexing in actMat starts from 1 while ID starts from 0
        ops = actMat[job, np.nonzero(actMat[job])[0]] - 1
        unscheduled_ops = [op for op in ops if op not in pss]
        for op in unscheduled_ops:
            lb_completion[op] = lb_completion[op - 1] + lt_flat[op - 1] + dur_flat[op] if op - 1 in ops else dur_flat[op]

    # get the ID of all loading operations
    loading_ops = actMat[n_j:][np.nonzero(actMat[n_j:])] - 1
    # take durations of loading operations from the duration vector, and store them in order of the regular jobs
    loading_durations_per_job = [dur_flat[nbo:][truck_num] for truck_num in truck_array]
    # select the lower bound on completion of the last operation of every job, and add lead time + loading duration
    possible_lbs_loading = lb_completion[last_col[:n_j]] + lt_flat[last_col[:n_j]] + loading_durations_per_job
    # take maximum of the possible lower bounds for each truck
    lb_load = [np.amax(possible_lbs_loading[np.where(truck_array == truck_num)]) for truck_num in range(n_t)]

    for op_idx, op in enumerate(loading_ops):
        if op not in pss:
            lb_completion[op] = lb_load[op_idx]

    return lb_completion


if __name__ == '__main__':
    dur = np.array([[1, 2], [3, 4]])
    lt = np.array([[5, 2], [4, 7]])
    finish_time = np.zeros_like(dur)
