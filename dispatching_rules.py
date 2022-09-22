from itertools import compress
import numpy as np
from Params import configs
from JSSP_Env import SJSSP


def shortest_processing_time(candidates, mask, dur):
    # perform masking
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    durations = dur[masked_candidates]
    shortest_op = np.argmin(durations)
    action = masked_candidates[shortest_op]
    return action


def largest_processing_time(candidates, mask, dur):
    # perform masking
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    durations = dur[masked_candidates]
    longest_op = np.argmax(durations)
    action = masked_candidates[longest_op]
    return action


def most_work_remaining(candidates, mask, env):
    # convert flat arrays to matrix representations
    duration_matrix = np.zeros_like(env.action_matrix)
    duration_matrix[np.nonzero(env.action_matrix)] = env.dur
    lead_time_matrix = np.zeros_like(env.action_matrix)
    lead_time_matrix[np.nonzero(env.action_matrix)] = env.lt
    # only collect information for unscheduled operations
    unfinished_mark = np.zeros_like(env.finished_mark)
    where_0 = np.where(env.finished_mark == 0)
    unfinished_mark[where_0] = 1
    unfinished_mark_matrix = np.zeros_like(env.action_matrix)
    unfinished_mark_matrix[np.nonzero(env.action_matrix)] = unfinished_mark
    unfinished_duration_matrix = duration_matrix * unfinished_mark_matrix
    unfinished_lead_time_matrix = lead_time_matrix * unfinished_mark_matrix
    # sum up remaining processing time + lead time
    job_tpt = np.sum((unfinished_duration_matrix + unfinished_lead_time_matrix), axis=1)
    # add corresponding loading duration to job tpt
    truck_dur_per_job = np.array([job_tpt[env.number_of_jobs + env.truck_array[job]] for job in range(env.number_of_jobs)])
    job_tpt[:env.number_of_jobs] += truck_dur_per_job
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    job_idx = np.arange(env.number_of_jobs + env.number_of_trucks)
    masked_jobs = np.array(list(compress(job_idx, inverted_mask)))
    # remove illegal jobs from the tpt array
    mwr_candidates = job_tpt[masked_jobs]
    # take job index with highest mwr
    longest_op = np.argmax(mwr_candidates)
    # action = operation in masked_candidates at the job index
    action = masked_candidates[longest_op]
    return action


def most_operations_remaining(candidates, mask, env):
    unfinished_mark = np.zeros_like(env.finished_mark)
    where_0 = np.where(env.finished_mark == 0)
    unfinished_mark[where_0] = 1
    unfinished_mark_matrix = np.zeros_like(env.action_matrix)
    unfinished_mark_matrix[np.nonzero(env.action_matrix)] = unfinished_mark
    ops_remaining = np.sum(unfinished_mark_matrix, axis=1)
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    job_idx = np.arange(env.number_of_jobs + env.number_of_trucks)
    masked_jobs = np.array(list(compress(job_idx, inverted_mask)))
    mor_candidates = ops_remaining[masked_jobs]
    most_ops_remaining = np.argmax(mor_candidates)
    action = masked_candidates[most_ops_remaining]
    return action


def random_selection(candidates, mask):
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    action = np.random.choice(masked_candidates)
    return action


def earliest_due_date(candidates, mask, env, due_dates):
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    job_idx = np.arange(env.number_of_jobs + env.number_of_trucks)
    masked_jobs = np.array(list(compress(job_idx, inverted_mask)))
    # remove illegal jobs from the due date array
    due_dates_candidates = due_dates[masked_jobs]
    earliest_op = np.argmin(due_dates_candidates)
    action = masked_candidates[earliest_op]
    return action


def baseline_performance(data_set, n_j, n_m, n_t, rule_set, verbose=False):
    env = SJSSP(n_j=n_j, n_m=n_m, n_t=n_t)

    avg_make_spans = {}
    make_spans = {}

    # rollout episode using current model
    for rule in rule_set:
        make_spans[rule] = []
        for i, data in enumerate(data_set):

            # reset JSSP environment
            adj, node_fea, candidates, mask = env.reset(data)
            rewards = - env.initQuality
            while True:
                if rule == 'spt':
                    action = shortest_processing_time(candidates, mask, env.dur)
                elif rule == 'lpt':
                    action = largest_processing_time(candidates, mask, env.dur)
                elif rule == 'mwr':
                    action = most_work_remaining(candidates, mask, env)
                elif rule == 'mor':
                    action = most_operations_remaining(candidates, mask, env)
                elif rule == 'rand':
                    action = random_selection(candidates, mask)
                elif rule == 'edd':
                    # TODO: change due_date
                    due_dates = [0]
                    action = earliest_due_date(candidates, mask, env, due_dates)
                adj, fea, reward, done, candidates, mask = env.step(action.item())
                rewards += reward

                if done:
                    break

            make_spans[rule].append(rewards - env.posRewards)

        avg_make_spans[rule] = sum(make_spans[rule]) / len(make_spans[rule])

    return make_spans, avg_make_spans


if __name__ == '__main__':
    n_j = 15
    n_m = 10
    n_t = 10
    run_type = "L2D-LeadTime_Loading_VRL"
    seed_val = 200

    dataLoaded = np.load(
        './DataGen/generatedData_' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t)
        + '_Seed' + str(seed_val) + '.npy')
    arrayLoaded = np.load(
        './DataGen/generatedArray_' + str(run_type) + '_' + str(n_j) + '_' + str(n_m) + '_' + str(n_t)
        + '_Seed' + str(seed_val) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2], dataLoaded[i][3], arrayLoaded[i]))

    rule_set = ['rand', 'mor', 'mwr', 'lpt', 'spt']
    makespans, avg_makespans = baseline_performance(vali_data, n_j, n_m, n_t, rule_set)  # [vali_data[0]]
    temp = 0