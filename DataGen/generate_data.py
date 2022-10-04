import numpy as np
from uniform_instance_gen import uni_instance_gen

j = 6
m = 6
l = 1
h = 99
shuffle_machines = True
vali = False
run_type = 'L2D'
batch_size = 1000
seed = 100

np.random.seed(seed)

data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h, shuffle_machines=shuffle_machines) for _ in range(batch_size)])
print(data.shape)
if vali:
    np.save('generatedData_{}_{}_{}_Seed{}.npy'.format(run_type, j, m, seed), data)
else:
    np.save('./Test/generatedTestData_{}_{}_{}_Seed{}.npy'.format(run_type, j, m, seed), data)