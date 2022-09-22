from itertools import compress
import numpy as np
from Params import configs
from JSSP_Env import SJSSP


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
    unfinished_mark = np.zeros_like(env.finished_mark)
    where_0 = np.where(env.finished_mark == 0)
    unfinished_mark[where_0] = 1
    unfinished_duration_matrix = env.dur * unfinished_mark
    # sum up remaining processing time
    job_tpt = np.sum(unfinished_duration_matrix, axis=1)
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    job_idx = np.arange(env.number_of_jobs)
    masked_jobs = np.array(list(compress(job_idx, inverted_mask)))
    # remove illegal jobs from the tpt array
    mwr_candidates = job_tpt[masked_jobs]
    # take job index with highest mwr
    longest_op = np.argmax(mwr_candidates)
    # action = operation in masked_candidates at the job index
    action = masked_candidates[longest_op]
    return action


def most_time_remaining(candidates, mask, env):
    # only collect information for unscheduled operations
    unfinished_mark = np.zeros_like(env.finished_mark)
    where_0 = np.where(env.finished_mark == 0)
    unfinished_mark[where_0] = 1
    unfinished_duration_matrix = env.dur * unfinished_mark
    unfinished_lead_time_matrix = env.lt * unfinished_mark
    # sum up remaining processing time + lead time
    job_tpt = np.sum((unfinished_duration_matrix + unfinished_lead_time_matrix), axis=1)
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    job_idx = np.arange(env.number_of_jobs)
    masked_jobs = np.array(list(compress(job_idx, inverted_mask)))
    # remove illegal jobs from the tpt array
    mtr_candidates = job_tpt[masked_jobs]
    # take job index with highest mwr
    longest_op = np.argmax(mtr_candidates)
    # action = operation in masked_candidates at the job index
    action = masked_candidates[longest_op]
    return action


def most_operations_remaining(candidates, mask, env):
    unfinished_mark = np.zeros_like(env.finished_mark)
    where_0 = np.where(env.finished_mark == 0)
    unfinished_mark[where_0] = 1
    ops_remaining = np.sum(unfinished_mark, axis=1)
    # perform masking for jobs and candidates
    inverted_mask = [not elem for elem in mask]
    masked_candidates = np.array(list(compress(candidates, inverted_mask)))
    job_idx = np.arange(env.number_of_jobs)
    masked_jobs = np.array(list(compress(job_idx, inverted_mask)))
    mor_candidates = ops_remaining[masked_jobs]
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
    job_idx = np.arange(env.number_of_jobs)
    masked_jobs = np.array(list(compress(job_idx, inverted_mask)))
    # remove illegal jobs from the due date array
    due_dates_candidates = due_dates[masked_jobs]
    earliest_op = np.argmin(due_dates_candidates)
    action = masked_candidates[earliest_op]
    return action


def baseline_performance(data_set, n_j, n_m):
    env = SJSSP(n_j=n_j, n_m=n_m)

    avg_make_spans = {}
    make_spans = {}
    # rule_set = ['spt', 'lpt', 'mwr', 'mtr', 'mor']
    rule_set = ['rand', 'mtr', 'mor', 'mwr', 'lpt', 'spt']

    # rollout episode using current model
    for rule in rule_set:
        make_spans[rule] = []
        for data in data_set:

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
                elif rule == 'mtr':
                    action = most_time_remaining(candidates, mask, env)
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

    dataLoaded = np.load(
        './DataGen/generatedData_' + str(configs.run_type) + '_' + str(configs.n_j) + '_' + str(
            configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2]))

    results = baseline_performance(vali_data, configs.n_j, configs.n_m)  # [vali_data[0]]
