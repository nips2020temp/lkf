''' Learning Kalman-Bucy filter
'''

from systems import *
from integrator import Integrator
from utils import set_seed
from pool_executor import pool_execute
from lkf import LKF
from systems import *

from typing import Callable
import numpy as np
import pandas as pd
import pdb
import scipy.stats as stats
import matplotlib.pyplot as plt

seed = 9001
set_seed(seed)

def gen_system(arg: dict):
	z = Oscillator(arg['dt'], 0.0, 1.0)
	eta0 = np.random.normal(0., arg['eta_var'], (2, 2))
	eta = lambda t: eta0
	F_hat = lambda t: z.F(t) + eta(t)
	f = LKF(z.x0, F_hat, z.H, z.Q, z.R, arg['dt'], tau=arg['tau'], eps=arg['eps'], gamma=arg['gamma'])
	return z, f

args = [
	# {'dt': 1e-4, 'T': 40., 'tau': 0.25, 'eps': 1e-3, 'gamma': 0.25, 'eta_var': 0.03, 'title': 'Tuned'},
	{'dt': 1e-3, 'T': 40., 'tau': 0.25, 'eps': 1e-3, 'gamma': 0.25, 'eta_var': 0.03, 'title': 'dt'},
	{'dt': 1e-3, 'T': 40., 'tau': 0.25, 'eps': 1e-4, 'gamma': 0.25, 'eta_var': 0.03, 'title': 'eps'},
	{'dt': 1e-3, 'T': 40., 'tau': 0.25, 'eps': 1e-3, 'gamma': 1.0, 'eta_var': 0.03, 'title': 'gamma'},
]

results = pool_execute(args, gen_system, seed=seed)

fig, axs = plt.subplots(1, len(args), figsize=(20, 20))
if type(axs) != np.ndarray:
	axs = [axs]

for (i, arg) in enumerate(args):
	result = results[i]
	axs[i].plot(result['hist_t'], result['hist_z'][:,0], color='blue', label='obs')
	axs[i].plot(result['hist_t'], result['hist_x'][:,0], color='orange', label='est')
	axs[i].set_title(arg['title'])

plt.show()
