from systems import Oscillator, LSProcess
from integrator import Integrator
from utils import set_seed, rmse
from lkf import LKF

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

set_seed(5001)

dt = 0.001
n = 25000
eta = np.random.normal(0.0, 0.01, (2, 2))
z = Oscillator(dt, 0.0, 1.0)
F_hat = lambda t: z.F(t) + eta
print(F_hat(0))

tau_rng = np.linspace(0.01, 1, 40)
rmse_hist = []
for tau in tau_rng:

	z = Oscillator(dt, 0.0, 1.0)
	f = LKF(z.x0, F_hat, z.H, z.Q, z.R, dt, tau=tau)
	hist_err = []
	for _ in range(n):
		z_t = z()
		x_t, err_t, eta_t = f(z_t)
		hist_err.append(err_t)
	rmse_hist.append(rmse(np.array(hist_err)))

plt.plot(tau_rng, rmse_hist)
plt.xlabel('tau')
plt.ylabel('RMSE')

plt.show()