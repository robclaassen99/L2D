import numpy as np


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def extend_matrix(input_mat: np.ndarray, n_j: int, n_m: int, n_t: int, low: int, high: int):
    """
    Function that can be used to extend the machine matrix and duration matrix to implement the loading operation
    for loading operations

    :param input_mat: version of any matrix without loading operations
    :param n_j: number of jobs
    :param n_m: number of machines
    :param n_t: number of trucks (job batches)
    :param low: lower bound for extra values in output matrix
    :param high: upper bound for extra values in output matrix

    :return: input matrix extended with loading machine and loading operations
    """
    fill_values = np.random.randint(low=low, high=high, size=n_t, dtype=int)
    load_values = np.concatenate([np.zeros(n_j, dtype=int), np.full(shape=n_t, fill_value=fill_values, dtype=int)])
    out_mat = np.zeros((n_j + n_t, n_m), dtype=int)
    out_mat[:, -1] = load_values
    out_mat[:-n_t, :-1] = input_mat
    return out_mat


def uni_instance_gen(n_j, n_m, n_t, low, high, lt_low, lt_high, shuffle_machines):
    assert n_j >= n_t, 'Number of trucks must be leq number of jobs'

    # n_m - 1 because last machine is specifically for loading
    times = np.random.randint(low=low, high=high, size=(n_j, n_m - 1))
    lead_times = np.random.randint(low=lt_low, high=lt_high, size=(n_j, n_m - 1))
    machines = np.expand_dims(np.arange(1, n_m), axis=0).repeat(repeats=n_j, axis=0)
    if shuffle_machines:
        machines = permute_rows(machines)

    # loading operations only have >0 duration for loading machine
    # basic operations have 0 duration for loading machine, >0 all other machines
    times = extend_matrix(input_mat=times, n_j=n_j, n_m=n_m, n_t=n_t, low=low, high=high)
    # loading operations always take place on last machine (i.e. machine number n_m)
    # basic operations are never processed by last machine
    machines = extend_matrix(input_mat=machines, n_j=n_j, n_m=n_m, n_t=n_t, low=n_m, high=n_m+1)
    # loading operations have no lead time, because they have no successor operation
    lead_times = extend_matrix(input_mat=lead_times, n_j=n_j, n_m=n_m, n_t=n_t, low=0, high=1)

    guarantee_truck = np.arange(n_t)  # make sure each truck ID is included at least once in truck_array
    rest_trucks = np.random.randint(low=0, high=n_t, size=(n_j - n_t))  # assign remaining jobs to random trucks
    truck_array = np.concatenate([guarantee_truck, rest_trucks], axis=None, dtype=int)
    np.random.shuffle(truck_array)  # randomize truck allocation

    return times, lead_times, machines, truck_array


def generate_deadlines(times, lead_times, truck_array, deadline_tightness):
    n_j = truck_array.shape[0]
    n_t = times.shape[0] - n_j
    total_time = times + lead_times
    tpt = np.sum(total_time, axis=1)
    load_dur_per_job = [tpt[n_j + truck_array[j]] for j in range(n_j)]
    tpt[:n_j] += load_dur_per_job
    max_tpt_per_truck = np.array([np.amax(tpt[j]) for t in range(n_t) for j in np.where(truck_array == t)])
    truck_deadlines = max_tpt_per_truck * deadline_tightness
    deadlines = np.concatenate((np.array([truck_deadlines[truck_array[j]] for j in range(n_j)]), truck_deadlines))
    deadlines[:n_j] -= load_dur_per_job
    return np.rint(deadlines).astype(np.int32)


def override(fn):
    """
    override decorator
    """
    return fn


if __name__ == '__main__':
    n_j = 3
    n_m = 3
    n_t = 2
    low = 1
    high = 10
    lt_low = 1
    lt_high = 10
    shuffle_machines = True
    dur, lt, m, truck_list = uni_instance_gen(n_j, n_m, n_t, low, high, lt_low, lt_high, shuffle_machines)