''' Discretization of Kalman-Bucy filter
'''

from systems import Oscillator, LSProcess
from integrator import Integrator

from typing import Callable
import numpy as np

class KF(LSProcess):
	def __init__(self, x0: np.ndarray, F: Callable, H: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float):
		self.F = F
		self.H = H
		self.Q = Q
		self.R = R
		self.dt = dt
		self.ndim = x0.shape[0]
		rep_ndim = self.ndim*(self.ndim+1) # representational dimension

		def f(t, state, z_t, F_t):
			state = state.reshape((self.ndim, self.ndim+1))
			x_t, P_t = state[:, :1], state[:, 1:]
			z_t = z_t[:, np.newaxis]
			K_t = P_t@self.H@np.linalg.inv(self.R)
			d_x = F_t@x_t + K_t@(z_t - self.H@x_t)
			d_P = F_t@P_t + P_t@F_t.T + self.Q - K_t@self.R@K_t.T
			d_state = np.concatenate((d_x, d_P), axis=1)
			return d_state.ravel() # Flatten for integrate.ode

		def g(t, state):
			return np.zeros((rep_ndim, rep_ndim))

		x0 = x0[:, np.newaxis]
		P0 = x0@x0.T
		iv = np.concatenate((x0, P0), axis=1).ravel() # Flatten for integrate.ode
		self.r = Integrator(f, g, rep_ndim)
		self.r.set_initial_value(iv, 0.)

	def __call__(self, z_t: np.ndarray):
		''' Observe through filter ''' 
		self.r.set_f_params(z_t, self.F(self.t))
		self.r.integrate(self.t + self.dt)
		x_t = np.squeeze(self.r.y.reshape((self.ndim, self.ndim+1))[:, :1])
		err_t = z_t - x_t@self.H.T
		return x_t.copy(), err_t # x_t variable gets reused somewhere...

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	dt = 0.001
	n = 10000
	z = Oscillator(dt, 0.0, 1.0)
	f = KF(z.x0, z.F, z.H, z.Q, z.R, dt)
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
	fig, axs = plt.subplots(1, 3, figsize=(15, 5))
	axs[0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
	axs[0].plot(hist_x[:,0], hist_x[:,1], color='orange', label='est')
	axs[0].legend()
	axs[1].plot(hist_t, hist_err[:,0])
	axs[2].plot(hist_t, hist_err[:,1])
	plt.show()

