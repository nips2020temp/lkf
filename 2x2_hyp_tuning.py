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
	{'dt': 1e-3, 'T': 40., 'tau': 0.25, 'eps': 1e-3, 'gamma': 0.25, 'eta_var': 0.03, 'title': 'Tuned'},
	{'dt': 1e-3, 'T': 40., 'tau': 0.1, 'eps': 1e-3, 'gamma': 0.25, 'eta_var': 0.03, 'title': 'Low tau'},
	{'dt': 1e-3, 'T': 40., 'tau': 0.25, 'eps': 1e-5, 'gamma': 0.25, 'eta_var': 0.03, 'title': 'Low eps'},
	{'dt': 1e-3, 'T': 40., 'tau': 0.25, 'eps': 1e-3, 'gamma': 1.0, 'eta_var': 0.03, 'title': 'High gamma'},
]

results = pool_execute(args, gen_system, seed=seed)

fig, axs = plt.subplots(2, len(args), figsize=(20, 20), gridspec_kw={'height_ratios': [4,1]})

for (i, arg) in enumerate(args):
	result = results[i]
	axs[0,i].plot(result['hist_t'], result['hist_z'][:,0], color='blue', label='obs')
	axs[0,i].plot(result['hist_t'], result['hist_x'][:,0], color='orange', label='est')
	axs[0,i].set_title(arg['title'])
	axs[0,i].set_ylim((-4.5, 4.5))
	axs[1,i].plot(result['hist_t'], np.linalg.norm(result['hist_err'], axis=1), color='blue', label='obs')
	axs[1,i].set_ylim((0., 2.0))
	axs[1,i].axis('off')

plt.setp(axs[1, 0], ylabel='Error (norm)')

fig.tight_layout()
plt.show()
