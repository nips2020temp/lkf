''' Learning Kalman-Bucy filter
'''

from systems import *
from integrator import Integrator
from utils import *
from pool_executor import pool_execute
from lkf import LKF
from systems import *

from typing import Callable
import numpy as np
import pandas as pd
import pdb
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

seed = 9001
set_seed(seed)

def gen_system(arg: dict):
	z = Oscillator(arg['dt'], 0.0, 1.0)
	eta0 = np.random.normal(0., arg['eta_var'], (2, 2))
	eta = lambda t: eta0
	F_hat = lambda t: z.F(t) + eta(t)
	f = LKF(z.x0, F_hat, z.H, z.Q, z.R, arg['dt'], tau=arg['tau'], eps=arg['eps'], gamma=arg['gamma'])
	return z, f

gamma_rng = np.linspace(1., 0.05, 30)
eps_power_rng = np.linspace(-5., -0.5, 30)
eps_rng = np.power(10, eps_power_rng)
dt = 1e-3
T = 40
tau = 0.25
eta_var = 0.03

xx, yy = np.meshgrid(eps_rng, gamma_rng)
xy = np.stack((xx, yy), axis=2)
xy = xy.reshape((xy.shape[0]*xy.shape[1], xy.shape[2]))

args = [
	{'dt': dt, 'T': T, 'tau': tau, 'eps': elem[0], 'gamma': elem[1], 'eta_var': eta_var} for elem in xy
]

def reduce_result(result):
	if result['hist_t'][-1] < T: # didn't complete
		return np.nan
	else:
		return np.linalg.norm(rms(result['hist_err']))

results = pool_execute(args, gen_system, seed=seed, reduce_result=reduce_result)
results = np.array(results).reshape((xx.shape[0], xx.shape[1]))

# pdb.set_trace()

fig, ax = plt.subplots()
sns.heatmap(results, xticklabels=np.around(eps_power_rng, 2), yticklabels=np.around(gamma_rng, 2), ax=ax)
ax.set_xlabel('eps (powers of 10)')
ax.set_ylabel('gamma')
plt.show()

