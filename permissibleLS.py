from Params import configs
import numpy as np


def permissibleLeftShift(a, job_a, mchMat, durMat, ltMat, actMat, truck_array, first_col, last_col, nbo,
                         mchsStartTimes, opIDsOnMchs):
    mch_flat = mchMat[np.nonzero(actMat)].flatten()
    dur_flat = durMat[np.nonzero(actMat)].flatten()
    lt_flat = ltMat[np.nonzero(actMat)].flatten()
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a, job_a, mch_flat, dur_flat, lt_flat, truck_array,
                                                        first_col, last_col, nbo, mchsStartTimes, opIDsOnMchs)
    dur_a = dur_flat[a]
    mch_a = mch_flat[a] - 1
    startTimesForMchOfa = mchsStartTimes[mch_a]
    opsIDsForMchOfa = opIDsOnMchs[mch_a]
    flag = False

    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
    # print('possiblePos:', possiblePos)
    if len(possiblePos) == 0:
        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a, jobRdyTime_a, dur_flat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
        else:
            flag = True
            startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa)
    return startTime_a, flag


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')
    index = np.where(startTimesForMchOfa == -configs.high)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)
    startTimesForMchOfa[index] = startTime_a
    opsIDsForMchOfa[index] = a
    return startTime_a


def calLegalPos(dur_a, jobRdyTime_a, dur_flat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
    durOfPossiblePos = dur_flat[opsIDsForMchOfa[possiblePos]]
    startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + dur_flat[[opsIDsForMchOfa[possiblePos[0]-1]]])
    endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a <= possibleGaps)[0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]
    return startTime_a


def calJobAndMchRdyTimeOfa(a, job_a, mch_flat, dur_flat, lt_flat, truck_array, first_col, last_col, nbo,
                           mchsStartTimes, opIDsOnMchs):
    mch_a = mch_flat[a] - 1

    if a < nbo:  # a is basic operation
        jobPredecessor = a - 1 if a not in first_col else None
        if jobPredecessor is not None:
            durJobPredecessor = dur_flat[jobPredecessor]
            mchJobPredecessor = mch_flat[jobPredecessor] - 1
            ltJobPredecessor = lt_flat[jobPredecessor]
            jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)]
                            + durJobPredecessor + ltJobPredecessor).item()
        else:
            jobRdyTime_a = 0
    else:  # a is loading operation
        jobs_in_truck = np.where(truck_array == (job_a - truck_array.shape[0]))[0]
        load_predecessors = last_col[jobs_in_truck]
        mch_load_predecessors = mch_flat[load_predecessors] - 1
        dur_load_predecessors = dur_flat[load_predecessors]
        lt_load_predecessors = lt_flat[load_predecessors]
        jobRdyTime_a = max([(mchsStartTimes[mch_load_predecessors[j]][np.where(opIDsOnMchs[mch_load_predecessors[j]]
                                                                               == load_predecessors[j])]
                             + dur_load_predecessors[j] + lt_load_predecessors[j])
                            for j in range(load_predecessors.shape[0])])
    # cal mchRdyTime_a
    mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(np.where(opIDsOnMchs[mch_a]
                                                                                               >= 0)[0]) != 0 else None
    if mchPredecessor is not None:
        durMchPredecessor = dur_flat[mchPredecessor]
        mchRdyTime_a = (mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1] + durMchPredecessor).item()
    else:
        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a


if __name__ == "__main__":
    from JSSP_Env import SJSSP
    from uniform_instance_gen import uni_instance_gen
    import time

    n_j = 3
    n_m = 3
    low = 1
    high = 99
    lt_low = 1
    lt_high = 99
    SEED = 10
    np.random.seed(SEED)
    env = SJSSP(n_j=n_j, n_m=n_m)

    '''arr = np.ones(3)
    idces = np.where(arr == -1)
    print(len(idces[0]))'''

    # rollout env random action
    t1 = time.time()
    data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high, lt_low=lt_low, lt_high=lt_high)
    print('Dur')
    print(data[0])
    print('LT')
    print(data[1])
    print('Mach')
    print(data[-1])
    print()

    # start time of operations on machines
    mchsStartTimes = -configs.high * np.ones_like(data[0].transpose(), dtype=np.int32)
    # Ops ID on machines
    opIDsOnMchs = -n_j * np.ones_like(data[0].transpose(), dtype=np.int32)

    # random rollout to test
    # count = 0
    _, _, omega, mask = env.reset(data)
    rewards = []
    flags = []
    # ts = []
    while True:
        action = np.random.choice(omega[np.where(mask == 0)])
        print(action)
        mch_a = np.take(data[-1], action) - 1
        # print(mch_a)
        # print('action:', action)
        # t3 = time.time()
        adj, _, reward, done, omega, mask = env.step(action)
        # t4 = time.time()
        # ts.append(t4 - t3)
        # jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a=action, mchMat=data[-1], durMat=data[0], mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs)
        # print('mchRdyTime_a:', mchRdyTime_a)
        startTime_a, flag = permissibleLeftShift(a=action, durMat=data[0].astype(np.single),
                                                 ltMat=data[1].astype(np.single), mchMat=data[-1], mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs)
        flags.append(flag)
        # print('startTime_a:', startTime_a)
        # print('mchsStartTimes\n', mchsStartTimes)
        # print('NOOOOOOOOOOOOO' if not np.array_equal(env.mchsStartTimes, mchsStartTimes) else '\n')
        print('opIDsOnMchs\n', opIDsOnMchs)
        # print('LBs\n', env.LBs)
        rewards.append(reward)
        # print('ET after action:\n', env.LBs)
        print()
        if env.done():
            break
    t2 = time.time()
    print(t2 - t1)
    # print(sum(ts))
    # print(np.sum(opIDsOnMchs // n_m, axis=1))
    # print(np.where(mchsStartTimes == mchsStartTimes.max()))
    # print(opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())])
    print(mchsStartTimes.max() + np.take(data[0], opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())]))
    # np.save('sol', opIDsOnMchs // n_m)
    # np.save('jobSequence', opIDsOnMchs)
    # np.save('testData', data)
    # print(mchsStartTimes)
    durAlongMchs = np.take(data[0], opIDsOnMchs)
    mchsEndTimes = mchsStartTimes + durAlongMchs
    print(mchsStartTimes)
    print(mchsEndTimes)
    print()
    print(env.opIDsOnMchs)
    print(env.adj)
    # print(sum(flags))
    # data = np.load('data.npy')

    # print(len(np.where(np.array(rewards) == 0)[0]))
    # print(rewards)
