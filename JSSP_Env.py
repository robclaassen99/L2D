import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
from utils import get_cols, check_loading_predecessors


class SJSSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m,
                 n_t):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_trucks = n_t
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        self.getCols = get_cols

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect
        if action not in self.partial_sol_sequeence:

            # UPDATE BASIC INFO:
            idx = np.where(self.action_matrix == (action + 1))
            row = idx[0][0]
            self.step_count += 1
            self.finished_mark[action] = 1
            dur_a = self.dur[action]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissibleLeftShift(a=action, job_a=row, mch_flat=self.m, dur_flat=self.dur,
                                                     lt_flat=self.lt, truck_array=self.truck_array,
                                                     first_col=self.first_col, last_col=self.last_col, nbo=self.nbo,
                                                     mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            self.finish_time[action] = startTime_a + dur_a

            # update omega or mask
            if action not in self.last_col:
                self.omega[row] += 1
            else:
                self.mask[row] = 1
                if action < self.nbo:  # action is basic operation
                    # check whether this job was the last job of a truck to finish
                    # job_remaining equals True if it was not the last job, and False if it was
                    job_remaining, truck_num = check_loading_predecessors(self.truck_array, row, self.mask)
                    self.mask[self.number_of_jobs + truck_num] = job_remaining

            self.LBs = self.getEndTimeLB(self.finish_time, self.dur, self.lt, self.action_matrix, self.truck_array,
                                         self.number_of_jobs, self.number_of_trucks, self.nbo, self.last_col,
                                         self.partial_sol_sequeence)

            # adj matrix
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            if action >= self.nbo:  # action is loading operation
                truck_number = row - self.number_of_jobs
                # select IDs of all jobs allocated to truck
                jobs_in_truck = np.where(self.truck_array == truck_number)
                # select last operation of all jobs allocated to truck
                last_ops_truck = self.last_col[jobs_in_truck]
                for op in last_ops_truck:
                    self.adj[action, op] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0

        # prepare for return
        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        reward = - (self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    @override
    def reset(self, data):
        # reset scheduled operations
        self.step_count = 0
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # load data
        self.truck_array = data[-1]
        self.action_mask = data[3].astype(np.int32)
        flat_mask = self.action_mask.flatten()
        # all matrices are flattened by default now!
        self.dur = (data[0] * self.action_mask).flatten()
        self.dur = self.dur[flat_mask > 0].astype(np.single)
        self.lt = (data[1] * self.action_mask).flatten()
        self.lt = self.lt[flat_mask > 0].astype(np.single)
        self.m = (data[2] * self.action_mask).flatten()
        self.m = self.m[flat_mask > 0].astype(np.int32)

        # basic instance information
        self.number_of_tasks = self.m.shape[0]
        self.nbo = self.number_of_tasks - self.number_of_trucks
        self.action_matrix = np.zeros_like(self.action_mask, dtype=np.int32)
        # note actions start from integer 1, not 0!
        self.action_matrix[np.nonzero(self.action_mask)] = np.arange(1, self.number_of_tasks + 1)

        # action matrix to index actions, as all data matrices now contain zero values
        # indexing of actions starts from 1, not 0
        self.action_matrix = np.zeros_like(self.action_mask, dtype=np.int32)
        self.action_matrix[np.nonzero(self.action_mask)] = np.arange(1, self.number_of_tasks + 1)

        # get first col and last col
        self.first_col, self.last_col = self.getCols(self.action_mask)

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream
        # add directed arcs between last operation of job and corresponding loading operation
        truck_y = self.last_col[:self.number_of_jobs]
        truck_x = [self.last_col[self.number_of_jobs:][truck_num] for truck_num in self.truck_array]
        self.adj[truck_x, truck_y] = 1

        self.finish_time = np.zeros(self.number_of_tasks, dtype=np.single)

        # initialize features
        self.LBs = self.getEndTimeLB(self.finish_time, self.dur, self.lt, self.action_matrix, self.truck_array,
                                     self.number_of_jobs, self.number_of_trucks, self.nbo, self.last_col,
                                     self.partial_sol_sequeence)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros(self.number_of_tasks, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)

        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)
        # initialize mask
        # value is False for all regular jobs, because none of them are finished at the start of an episode
        # value is True for all loading 'jobs', because they should be disabled at first
        self.mask = np.concatenate([np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool),
                                    np.full(shape=self.number_of_trucks, fill_value=1, dtype=bool)])

        # start time of operations on machines
        self.mchsStartTimes = -configs.high * np.ones_like(self.action_mask.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.action_mask.transpose(), dtype=np.int32)

        return self.adj, fea, self.omega, self.mask
