''' Stochastic linear systems
'''

from typing import Callable
import numpy as np
import scipy.integrate as scint

class LSProcess:
	''' Linear stochastic hidden proocess ''' 
	def __init__(self, x0: np.ndarray, F: callable, H: np.ndarray, dt: float, var_w: float, var_v: float):
		assert x0.shape[0] == H.shape[0] == F(0).shape[0]
		self.d = x0.shape[0]
		self.x0 = x0
		self.F = F
		self.H = H
		self.dt = dt
		var_w = np.ones((x0.shape[0], 1)) * var_w
		var_v = np.ones((x0.shape[0], 1)) * var_v
		self.Q = var_w@var_w.T
		self.R = var_v@var_v.T

		def system(t, x_t, F_t):
			w_t = np.random.multivariate_normal(np.zeros(self.d), self.Q)[:, np.newaxis]
			print(x_t)
			return F_t@x_t + w_t

		self.r = scint.ode(system).set_integrator('dop853')
		print(x0)
		self.r.set_initial_value(x0)

	def __call__(self):
		''' Observe process '''
		self.r.set_f_params(self.F(self.t))
		self.r.integrate(self.t + self.dt)
		x_t = self.r.y
		v_t = np.random.multivariate_normal(np.zeros(self.d), self.R)[:, np.newaxis]
		z_t = self.H@x_t + v_t
		print(z_t)
		return z_t 

	@property
	def t(self):
		return self.r.t


class Oscillator(LSProcess):
	def __init__(self, dt: float, var_w: float, var_v: float):
		F = lambda t: np.array([[-1.05,-3.60],[1.10, 1.05]])
		H = np.eye(2)
		x0 = np.array([[-1.], [-1.]])
		super().__init__(x0, F, H, dt, var_w, var_v)

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	z = Oscillator(0.01, 0.02, 0.02)
	hist_t = []
	hist_z = []
	for _ in range(1000):
		hist_z.append(z())
		hist_t.append(z.t)
	plt.plot(hist_t, hist_z)
