''' Learning Kalman-Bucy filter
'''

from systems import Oscillator, LSProcess
from integrator import Integrator
from utils import set_seed

from typing import Callable
import numpy as np
import pandas as pd
import pdb

class LKF(LSProcess):
	def __init__(self, x0: np.ndarray, F: Callable, H: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float, tau: float = float('inf')):
		self.F = F
		self.H = H
		self.Q = Q
		self.R = R
		self.dt = dt
		self.tau = tau
		self.window = 300
		self.acc = 1
		self.ndim = x0.shape[0]
		self.err_hist = []
		self.eta_t = np.zeros((self.ndim, self.ndim)) # temp var..
		self.eta_t_hist = [np.zeros((self.ndim, self.ndim))] # temp var..
		self.eta_t_avg = np.zeros((self.ndim, self.ndim)) # temp var..
		self.e_zz_t = np.zeros((self.ndim, self.ndim)) # temp var..
		self.p_inv_t = np.zeros((self.ndim, self.ndim)) # temp var..
		self.p_t = np.zeros((self.ndim, self.ndim)) # temp var..

		def f(t, state, z_t, err_hist, F_t):
			state = state.reshape(self.ode_shape)
			x_t, P_t, eta_t = state[:, :1], state[:, 1:3], state[:, 3:]
			z_t = z_t[:, np.newaxis]
			K_t = P_t@self.H@np.linalg.inv(self.R)
			self.p_t = P_t
			if t > self.tau: 
				H_inv = np.linalg.inv(self.H)
				P_inv = np.linalg.inv(P_t)
				self.p_inv_t = P_inv
				tau_t = np.random.normal(tau, tau / 10)
				tau_n = min(int(tau_t / self.dt), len(err_hist))
				err_t, err_tau = err_hist[-1][:,np.newaxis], err_hist[-tau_n][:,np.newaxis]
				d_zz = (err_t@err_t.T - err_tau@err_tau.T) / tau_t
				E_z = sum(err_hist)[:,np.newaxis] / tau_n
				d_uu = (((err_t - err_tau)/tau_t)@E_z.T + E_z@((err_t - err_tau)/tau_t).T) / tau_t
				self.e_zz_t = d_zz
				# d_P = F_t@P_t + P_t@F_t.T + self.Q - K_t@self.R@K_t.T
				eta_t = H_inv@d_zz@H_inv.T@P_inv / 2
				# eta_t = H_inv@(d_zz - d_uu)@H_inv.T@P_inv / 2
				# eta_t = H_inv@d_zz@H_inv.T / 2
				# eta_t = np.clip(eta_t, a_min=-1, a_max=1)
				self.eta_t = eta_t # TODO fix hack
				self.eta_t_hist.append(eta_t)
				if self.acc + 1 > self.window:
					del self.eta_t_hist[0]
				else:
					self.acc += 1
				self.eta_t_avg = sum(self.eta_t_hist) / self.acc
			F_est = F_t - self.eta_t_avg
			d_x = F_est@x_t + K_t@(z_t - self.H@x_t)
			d_P = F_est@P_t + P_t@F_est.T + self.Q - K_t@self.R@K_t.T
			d_eta = np.zeros((self.ndim, self.ndim)) # H_inv@(err_t@err_t.T - err_tau@err_tau.T)@H_inv.T@P_inv / (2*tau)
			d_state = np.concatenate((d_x, d_P, d_eta), axis=1)
			return d_state.ravel() # Flatten for integrator

		def g(t, state):
			return np.zeros((self.ode_ndim, self.ode_ndim))

		# state
		x0 = x0[:, np.newaxis]
		# covariance
		P0 = np.eye(self.ndim)
		# model variation
		eta0 = np.zeros((self.ndim, self.ndim))
		iv = np.concatenate((x0, P0, eta0), axis=1).ravel() # Flatten for integrator
		self.r = Integrator(f, g, self.ode_ndim)
		self.r.set_initial_value(iv, 0.)

	def __call__(self, z_t: np.ndarray):
		''' Observe through filter ''' 
		self.r.set_f_params(z_t, self.err_hist, self.F(self.t))
		self.r.integrate(self.t + self.dt)
		x_t = np.squeeze(self.r.y.reshape(self.ode_shape)[:, :1])
		err_t = z_t - x_t@self.H.T
		self.err_hist.append(err_t)
		if self.t > 2 * self.tau:
			self.err_hist = self.err_hist[1:]
		return x_t.copy(), err_t # x_t variable gets reused somewhere...

	@property
	def ode_shape(self):
		return (self.ndim, 1 + 2*self.ndim) # representational dimension: x_t, P_t, eta_t

	@property
	def ode_ndim(self):
		return self.ode_shape[0] * self.ode_shape[1] # for raveled representation

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	set_seed(5001)

	dt = 0.001
	n = 50000
	z = Oscillator(dt, 0.0, 1.0)
	eta = np.random.normal(0.0, 0.01, (2, 2))
	F_hat = lambda t: z.F(t) + eta
	print(F_hat(0))
	f = LKF(z.x0, F_hat, z.H, z.Q, z.R, dt, tau=0.25)

	max_err = .3

	hist_t = []
	hist_z = []
	hist_x = []
	hist_err = []
	hist_eta = []
	hist_ezz = []
	hist_pin = []
	hist_p = []
	for _ in range(n):
		z_t = z()
		x_t, err_t = f(z_t)
		hist_z.append(z_t)
		hist_t.append(z.t)
		hist_x.append(x_t) 
		hist_err.append(err_t)
		hist_eta.append(f.eta_t_avg.copy()) # variation
		hist_ezz.append(f.e_zz_t.copy())
		hist_pin.append(f.p_inv_t.copy())
		hist_p.append(f.p_t.copy())
		if np.linalg.norm(err_t) > max_err:
			print('Error overflowed!')
			break

	hist_t = np.array(hist_t)
	hist_z = np.array(hist_z)
	hist_x = np.array(hist_x)
	hist_err = np.array(hist_err)
	hist_eta = np.array(hist_eta)
	hist_ezz = np.array(hist_ezz)
	hist_pin = np.array(hist_pin)
	hist_p = np.array(hist_p)

	# pdb.set_trace()

	fig, axs = plt.subplots(3, 4, figsize=(20, 20))
	fig.suptitle('LKF')
	
	axs[0,0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
	axs[0,0].plot(hist_x[:,0], hist_x[:,1], color='orange', label='est')
	axs[0,0].legend()
	axs[0,0].set_title('System')

	axs[0,1].plot(hist_t, hist_z[:,0], color='blue', label='obs')
	axs[0,1].plot(hist_t, hist_x[:,0], color='orange', label='est')
	axs[0,1].set_title('Axis 1')

	axs[0,2].plot(hist_t, hist_z[:,1], color='blue', label='obs')
	axs[0,2].plot(hist_t, hist_x[:,1], color='orange', label='est')
	axs[0,2].set_title('Axis 2')

	var_err = eta - hist_eta
	axs[1,0].plot(hist_t, np.linalg.norm(var_err, axis=1))
	axs[1,0].set_title('Variation error (norm)')

	axs[1,1].plot(hist_t, hist_err[:,0])
	axs[1,1].set_title('Axis 1 error')

	axs[1,2].plot(hist_t, hist_err[:,1])
	axs[1,2].set_title('Axis 2 error')

	var_err_rast = var_err.reshape((var_err.shape[0], 4))
	# axs[1,1].plot(hist_t, var_err_rast[:,0])
	# axs[1,1].plot(hist_t, var_err_rast[:,1])
	# axs[1,1].plot(hist_t, var_err_rast[:,2])
	axs[1,3].plot(hist_t, var_err_rast[:,3])
	axs[1,3].set_title('Variation error (rasterized)')

	err_avg = pd.Series(index=hist_t, data=var_err_rast[:,3]).rolling(500).mean()
	axs[2,0].plot(err_avg.index, err_avg)
	axs[2,0].set_title('Variation error (rasterized, rolling mean)')

	axs[2,1].plot(hist_t, np.linalg.norm(hist_ezz, axis=(1,2)))
	axs[2,1].set_title('d_zz')

	p_inv_rast = hist_pin.reshape((hist_pin.shape[0], 4))
	axs[2,2].plot(hist_t, p_inv_rast[:,0])
	axs[2,2].plot(hist_t, p_inv_rast[:,1])
	axs[2,2].plot(hist_t, p_inv_rast[:,2])
	axs[2,2].plot(hist_t, p_inv_rast[:,3])
	axs[2,2].set_title('P_inv (rasterized)')

	p_rast = hist_p.reshape((hist_p.shape[0], 4))
	axs[2,3].plot(hist_t, p_rast[:,0])
	axs[2,3].plot(hist_t, p_rast[:,1])
	axs[2,3].plot(hist_t, p_rast[:,2])
	axs[2,3].plot(hist_t, p_rast[:,3])
	axs[2,3].set_title('P (rasterized)')


	# axs[3].plot(hist_t, hist_eta)

	# compact plot
	# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
	# axs[0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
	# axs[0].plot(hist_x[:,0], hist_x[:,1], color='orange', label='est')
	# axs[0].legend()
	# axs[0].set_title('System')
	# axs[1].plot(hist_t, hist_err[:,0])
	# axs[1].set_title('Axis 1 error')

	plt.show()
