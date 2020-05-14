from systems import Oscillator, LSProcess
from integrator import Integrator
from utils import set_seed
from kf import KF

from typing import Callable
import numpy as np

import matplotlib.pyplot as plt

set_seed(4001)

dt = 0.001
n = 25000
z = Oscillator(dt, 0.0, 1.0)
sigma = 0.05
eta = np.random.normal(0.0, sigma, (2, 2))
F_hat = lambda t: z.F(t) + eta
print(F_hat(0))
f = KF(z.x0, F_hat, z.H, z.Q, z.R, dt)
hist_t = []
hist_z = []
hist_x = []
hist_err = []
for _ in range(n):
	z_t = z()
	x_t, err_t = f(z_t)
	hist_z.append(z_t)
	hist_t.append(z.t)
	hist_x.append(x_t) 
	hist_err.append(err_t)
hist_t = np.array(hist_t)
hist_z = np.array(hist_z)
hist_x = np.array(hist_x)
hist_err = np.array(hist_err)
# fig, axs = plt.subplots(1, 1, figsize=(10, 10))
# fig.suptitle('KF')
plt.plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
plt.plot(hist_x[:,0], hist_x[:,1], color='orange', label='est')
plt.legend()
plt.title(f'Estimating a partially known system (sigma={sigma})')
# axs[1].plot(hist_t, hist_z[:,0], color='blue', label='obs')
# axs[1].plot(hist_t, hist_x[:,0], color='orange', label='est')
# axs[2].plot(hist_t, hist_z[:,1], color='blue', label='obs')
# axs[2].plot(hist_t, hist_x[:,1], color='orange', label='est')
# axs[3].plot(hist_t, hist_err[:,0])
# axs[3].set_title('Axis 1 error')
# axs[4].plot(hist_t, hist_err[:,1])
# axs[4].set_title('Axis 2 error')
plt.show()