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


def binary_mask_random(arr):
    n_zeros = np.random.randint(0, arr.shape[0])
    if n_zeros > 0:
        # NOTE: changed this to + 1, so validation set and training instance of 10x10 are only reproducible of we
        # remove this + 1 again
        for i in range(np.random.randint(0, n_zeros + 1)):
            x = np.random.randint(0, arr.shape[0])
            arr[x] = 0
    return arr


def uni_instance_gen(n_j, n_m, n_t, low, high, lt_low, lt_high):
    assert n_j >= n_t, 'Number of trucks must be leq number of jobs'

    # n_m - 1 because last machine is specifically for loading
    times = np.random.randint(low=low, high=high, size=(n_j, n_m - 1))
    lead_times = np.random.randint(low=lt_low, high=lt_high, size=(n_j, n_m - 1))
    machines = np.expand_dims(np.arange(1, n_m), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)

    mask = np.ones(shape=(n_j, n_m - 1))
    np.apply_along_axis(func1d=binary_mask_random, axis=1, arr=mask)

    # loading operations only have >0 duration for loading machine
    # basic operations have 0 duration for loading machine, >0 all other machines
    times = extend_matrix(input_mat=times, n_j=n_j, n_m=n_m, n_t=n_t, low=low, high=high)
    # loading operations always take place on last machine (i.e. machine number n_m)
    # basic operations are never processed by last machine
    machines = extend_matrix(input_mat=machines, n_j=n_j, n_m=n_m, n_t=n_t, low=n_m, high=n_m+1)
    # loading operations have no lead time, because they have no successor operation
    lead_times = extend_matrix(input_mat=lead_times, n_j=n_j, n_m=n_m, n_t=n_t, low=0, high=1)
    # loading operations have mask = 1 for last machine, 0s everywhere else
    # basic operations have only 0s in the loading machine column
    mask = extend_matrix(input_mat=mask, n_j=n_j, n_m=n_m, n_t=n_t, low=1, high=2)

    guarantee_truck = np.arange(n_t)  # make sure each truck ID is included at least once in truck_array
    rest_trucks = np.random.randint(low=0, high=n_t, size=(n_j - n_t))  # assign remaining jobs to random trucks
    truck_array = np.concatenate([guarantee_truck, rest_trucks], axis=None, dtype=int)
    np.random.shuffle(truck_array)  # randomize truck allocation

    return times, lead_times, machines, mask, truck_array


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
    dur, lt, m, truck_array, mask = uni_instance_gen(n_j, n_m, n_t, low, high, lt_low, lt_high)