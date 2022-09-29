from itertools import compress
import numpy as np

def shortest_processing_time(candidates, mask, dur):
    # perform masking
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    durations = np.take(dur, masked_candidates)
    shortest_op = np.argmin(durations)
    action = masked_candidates[shortest_op]
    return action


def largest_processing_time(candidates, mask, dur):
    # perform masking
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    durations = np.take(dur, masked_candidates)
    longest_op = np.argmax(durations)
    action = masked_candidates[longest_op]
    return action


def most_work_remaining(candidates, mask, env):
    # only collect information for unscheduled operations
    finished_mark = np.zeros_like(env.action_matrix)
    finished_mark[np.nonzero(env.action_matrix)] = env.finished_mark
    unfinished_mark = np.zeros_like(finished_mark)
    where_0 = np.where(finished_mark == 0)
    unfinished_mark[where_0] = 1
    unfinished_duration_matrix = env.dur * unfinished_mark
    # sum up remaining processing time
    job_tpt = np.sum(unfinished_duration_matrix, axis=1)
    # add corresponding loading duration to job tpt
    truck_dur_per_job = np.array([job_tpt[env.number_of_jobs + env.truck_array[job]] for job in range(env.number_of_jobs)])
    job_tpt[:env.number_of_jobs] += truck_dur_per_job
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    idx = np.arange(env.number_of_jobs + env.number_of_trucks)
    masked_idx = np.array(list(compress(idx, inverted_mask)))
    # remove illegal jobs from the tpt array
    mwr_candidates = job_tpt[masked_idx]
    # take job index with highest mwr
    longest_op = np.argmax(mwr_candidates)
    # action = operation in masked_candidates at the job index
    action = masked_candidates[longest_op]
    return action


def most_time_remaining(candidates, mask, env):
    # only collect information for unscheduled operations
    finished_mark = np.zeros_like(env.action_matrix)
    finished_mark[np.nonzero(env.action_matrix)] = env.finished_mark
    unfinished_mark = np.zeros_like(finished_mark)
    where_0 = np.where(finished_mark == 0)
    unfinished_mark[where_0] = 1
    unfinished_duration_matrix = env.dur * unfinished_mark
    unfinished_lead_time_matrix = env.lt * unfinished_mark
    # sum up remaining processing time + lead time
    job_tpt = np.sum((unfinished_duration_matrix + unfinished_lead_time_matrix), axis=1)
    # add corresponding loading duration to job tpt
    truck_dur_per_job = np.array([job_tpt[env.number_of_jobs + env.truck_array[job]] for job in range(env.number_of_jobs)])
    job_tpt[:env.number_of_jobs] += truck_dur_per_job
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    idx = np.arange(env.number_of_jobs + env.number_of_trucks)
    masked_idx = np.array(list(compress(idx, inverted_mask)))
    # remove illegal jobs from the tpt array
    mtr_candidates = job_tpt[masked_idx]
    # take job index with highest mwr
    longest_op = np.argmax(mtr_candidates)
    # action = operation in masked_candidates at the job index
    action = masked_candidates[longest_op]
    return action


def most_operations_remaining(candidates, mask, env):
    finished_mark = np.zeros_like(env.action_matrix)
    finished_mark[np.nonzero(env.action_matrix)] = env.finished_mark
    unfinished_mark = np.zeros_like(finished_mark)
    where_0 = np.where(finished_mark == 0)
    unfinished_mark[where_0] = 1
    ops_remaining = np.sum(unfinished_mark, axis=1)
    ops_remaining[env.number_of_jobs:] = 1
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    idx = np.arange(env.number_of_jobs + env.number_of_trucks)
    masked_idx = np.array(list(compress(idx, inverted_mask)))
    mor_candidates = ops_remaining[masked_idx]
    most_ops_remaining = np.argmax(mor_candidates)
    action = masked_candidates[most_ops_remaining]
    return action


def random_selection(candidates, mask):
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    action = np.random.choice(masked_candidates)
    return action


def earliest_due_date(candidates, mask, env, due_dates):
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    idx = np.arange(env.number_of_jobs + env.number_of_trucks)
    masked_idx = np.array(list(compress(idx, inverted_mask)))
    # remove illegal jobs from the due date array
    due_dates_candidates = due_dates[masked_idx]
    earliest_op = np.argmin(due_dates_candidates)
    action = masked_candidates[earliest_op]
    return action
