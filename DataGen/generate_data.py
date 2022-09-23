import numpy as np
from uniform_instance_gen import uni_instance_gen

j = 30
m = 10
l = 1
h = 99
lt_l = 1
lt_h = 99
shuffle_machines = True
run_type = 'L2D-LeadTime'
batch_size = 100
seed = 200

np.random.seed(seed)

data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h, lt_low=lt_l, lt_high=lt_h, shuffle_machines=shuffle_machines)
                 for _ in range(batch_size)])
print(data.shape)
np.save('generatedData_{}_{}_{}_Seed{}.npy'.format(run_type, j, m, seed), data)
