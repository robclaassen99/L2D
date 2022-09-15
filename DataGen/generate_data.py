import numpy as np
from uniform_instance_gen import uni_instance_gen

j = 6
m = 6
t = 2
l = 1
h = 50
lt_l = 1
lt_h = 50
batch_size = 100
seed = 200

np.random.seed(seed)

data = np.array([uni_instance_gen(n_j=j, n_m=m, n_t=t, low=l, high=h, lt_low=lt_l, lt_high=lt_h)[:-1]
                 for _ in range(batch_size)])
t_array = np.array([uni_instance_gen(n_j=j, n_m=m, n_t=t, low=l, high=h, lt_low=lt_l, lt_high=lt_h)[-1]
                    for _ in range(batch_size)])
print(data.shape)
print(t_array.shape)
np.save('generatedDataLTTruck{}_{}_{}_Seed{}.npy'.format(j, m, t, seed), data)
np.save('generatedArrayLTTruck{}_{}_{}_Seed{}.npy'.format(j, m, t, seed), t_array)
