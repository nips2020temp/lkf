''' Learning Kalman-Bucy filter
'''

from systems import *
from integrator import Integrator
from utils import set_seed
from lkf import LKF
from kf import KF

from typing import Callable
import numpy as np
import pandas as pd
import pdb
import scipy.stats as stats

import matplotlib.pyplot as plt

set_seed(9001)

dt = 1e-4
T = 100.

""" Noisy LTI example """ 
z = Oscillator(dt, 0.0, 1.0)
eta_mu, eta_var = 0., 0.03
eta0 = np.random.normal(eta_mu, eta_var, (2, 2))
eta = lambda t: eta0
F_hat = lambda t: z.F(t) + eta(t)

print(F_hat(0))
f1 = KF(z.x0, F_hat, z.H, z.Q, z.R, dt)
f2 = LKF(z.x0, F_hat, z.H, z.Q, z.R, dt, tau=0.25, eps=1e-3, gamma=0.25)

max_err = 2.
max_eta_err = 100
max_zz = 100. 

hist_t = []
hist_z = []
hist_f1_x = []
hist_f2_x = []
hist_f1_err = []
hist_f2_err = []
hist_eta = []
while z.t <= T:
	z_t = z()
	x1_t, err1_t = f1(z_t)
	x2_t, err2_t = f2(z_t)
	hist_z.append(z_t)
	hist_t.append(z.t)
	hist_f1_x.append(x1_t) 
	hist_f2_x.append(x2_t) 
	hist_f1_err.append(err1_t)
	hist_f2_err.append(err2_t)
	hist_eta.append(f2.eta_t.copy())

	# Error condition 1
	if np.linalg.norm(err2_t) > max_err:
		print('Error overflowed!')
		break

	# Error condition 2
	if np.linalg.norm(f2.eta_t - eta(z.t)) > max_eta_err:
		print('Variation error overflowed!')
		break

	# Error condition 3
	if np.linalg.norm(f2.e_zz_t) > max_zz:
		print('d_zz overflowed!')
		break

# start, end = None, 20000 # for case analysis
start, end = None, None # for case analysis
every = 1

hist_t = np.array(hist_t)[start:end:every]
hist_z = np.array(hist_z)[start:end:every]
hist_f1_x = np.array(hist_f1_x)[start:end:every]
hist_f2_x = np.array(hist_f2_x)[start:end:every]
hist_f1_err = np.array(hist_f1_err)[start:end:every]
hist_f2_err = np.array(hist_f2_err)[start:end:every]
hist_eta = np.array(hist_eta)[start:end:every]

# pdb.set_trace()

fig, axs = plt.subplots(2, 5, figsize=(20, 20))

axs[0,0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
axs[0,0].plot(hist_f1_x[:,0], hist_f1_x[:,1], color='orange', label='est')
axs[0,0].legend()
axs[0,0].set_title('System')

axs[1,0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
axs[1,0].plot(hist_f2_x[:,0], hist_f2_x[:,1], color='orange', label='est')
axs[1,0].legend()

axs[0,1].plot(hist_t, hist_z[:,0], color='blue', label='obs')
axs[0,1].plot(hist_t, hist_f1_x[:,0], color='orange', label='est')
axs[0,1].set_title('Axis 1')

axs[1,1].plot(hist_t, hist_z[:,0], color='blue', label='obs')
axs[1,1].plot(hist_t, hist_f2_x[:,0], color='orange', label='est')

axs[0,2].plot(hist_t, hist_z[:,1], color='blue', label='obs')
axs[0,2].plot(hist_t, hist_f1_x[:,1], color='orange', label='est')
axs[0,2].set_title('Axis 2')

axs[1,2].plot(hist_t, hist_z[:,1], color='blue', label='obs')
axs[1,2].plot(hist_t, hist_f2_x[:,1], color='orange', label='est')

axs[0,3].plot(hist_t, hist_f1_err[:,0])
axs[0,3].set_title('Axis 1 error')

axs[1,3].plot(hist_t, hist_f2_err[:,0])

axs[0,4].plot(hist_t, hist_f1_err[:,1])
axs[0,4].set_title('Axis 2 error')

axs[1,4].plot(hist_t, hist_f2_err[:,1])

# var_err_rast = var_err.reshape((var_err.shape[0], 4))
# axs[1,3].plot(hist_t, var_err_rast[:,0])
# axs[1,3].plot(hist_t, var_err_rast[:,1])
# axs[1,3].plot(hist_t, var_err_rast[:,2])
# axs[1,3].plot(hist_t, var_err_rast[:,3])
# axs[1,3].set_title('Variation error (rasterized)')

plt.setp(axs[0, 0], ylabel='KF')
plt.setp(axs[1, 0], ylabel='LKF')

plt.show()
