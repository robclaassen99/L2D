import numpy as np


def get_cols(m: np.ndarray):
    """
    Function that selects the IDs of the first and last operation of each job, including loading operations
    If a job only has one operation (which is always true for loading 'jobs'), the operation is included in both arrays

    :param m: machine matrix

    :return: tuple of shape 2 containing a vector of first operations and a vector of last operations
    """

    num_ops = np.count_nonzero(m, axis=1)  # vector of number of operations of every job, including loading
    first_idxs_temp = num_ops.cumsum()
    first_idxs = np.concatenate([[0], first_idxs_temp[:-1]])
    last_idxs = first_idxs + (num_ops - 1)

    return first_idxs, last_idxs


def check_loading_predecessors(truck_array: np.ndarray, job_a: int, mask: np.ndarray):
    """
    Function that determines for a given truck whether all jobs allocated to it are finished

    :param truck_array: truck vector of shape [n_j], storing on which truck each job is loaded
    :param job_a: ID of the job to which the operation added to the schedule in this step belongs
    :param mask: boolean vector of shape [n_j + n_t] of finished jobs

    :return: tuple of shape 2 containing:
        job_remaining: boolean that indicates if all jobs in the truck of job_a are finished (False) or not (True)
        truck_number: ID of the truck on which job_a should be loaded
    """
    # select the ID of the truck on which job_a should be loaded
    truck_number = truck_array[job_a]
    # select IDs of all jobs allocated to truck of job_a
    jobs_in_truck = np.where(truck_array == truck_number)
    # check if all jobs allocated to truck are done by checking whether all their values in mask are True
    # job_remaining is set to False if all jobs in the truck are finished, else it is set to True
    job_remaining = not np.all([mask[job] for job in jobs_in_truck])

    return job_remaining, truck_number
