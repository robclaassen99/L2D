import numpy as np


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high, lt_low, lt_high, shuffle_machines):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    lead_times = np.random.randint(low=lt_low, high=lt_high, size=(n_j, n_m))
    lead_times[:, -1] = np.zeros(shape=n_j)  # no downstream lead time for last operation
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    if shuffle_machines:
        machines = permute_rows(machines)
    return times, lead_times, machines


def override(fn):
    """
    override decorator
    """
    return fn


