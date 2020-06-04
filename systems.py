''' Stochastic linear systems
'''

from integrator import *

from typing import Callable
import numpy as np
import scipy.integrate as scint

class LSProcess:
	''' Linear stochastic hidden proocess ''' 
	def __init__(self, x0: np.ndarray, F: callable, H: np.ndarray, dt: float, var_w: float, var_v: float, w_bnd: tuple = (-float('inf'), float('inf')), v_bnd: tuple = (-float('inf'), float('inf'))):
		assert x0.shape[0] == H.shape[0] == F(0).shape[0]
		self.ndim = x0.shape[0]
		self.x0 = x0
		self.F = F
		self.H = H
		self.dt = dt
		self.Q = np.eye(x0.shape[0]) * var_w
		self.R = np.eye(x0.shape[0]) * var_v
		self.w_bnd, self.v_bnd = w_bnd, v_bnd

		def f(t, x_t, F_t):
			return x_t@F_t.T

		def g(t, x_t):
			return self.Q

		# TODO use w_bnd
		self.r = Integrator(f, g, self.ndim)
		self.r.set_initial_value(x0, 0.)

	def __call__(self):
		''' Observe process '''
		self.r.set_f_params(self.F(self.t))
		self.r.integrate(self.t + self.dt)
		x_t = self.r.y
		v_t = self.R @ np.random.normal(0.0, np.sqrt(self.dt), self.ndim)
		z_t = self.H@x_t + np.clip(v_t, self.v_bnd[0], self.v_bnd[1])
		return z_t 

	@property
	def t(self):
		return self.r.t


class BoundedLSProcess(LSProcess):
	def __init__(self, x0: np.ndarray, F: callable, H: np.ndarray, dt: float, var_w: float, var_v: float):
		super().__init__(x0, F, H, dt, var_w, var_v, w_bnd=(-3*np.sqrt(var_w), 3*np.sqrt(var_w)), v_bnd=(-3*np.sqrt(var_v), 3*np.sqrt(var_v)))

class Oscillator(BoundedLSProcess):
	def __init__(self, dt: float, var_w: float, var_v: float):
		F = lambda t: np.array([[-1.05,-3.60],[1.10, 1.05]])
		H = np.eye(2)
		x0 = np.array([-1., -1.])
		super().__init__(x0, F, H, dt, var_w, var_v)

class SpiralSink(BoundedLSProcess):
	def __init__(self, dt: float, var_w: float, var_v: float):
		F = lambda t: np.array([[-1.15864464, -3.68960651], [1.06937006,  0.91600663]])
		H = np.eye(2)
		x0 = np.array([-1., -1.])
		super().__init__(x0, F, H, dt, var_w, var_v)

class SpiralSource(BoundedLSProcess):
	def __init__(self, dt: float, var_w: float, var_v: float):
		F = lambda t: np.array([[-0.98092249, -3.67973989], [1.06922538,  1.20720164]])
		H = np.eye(2)
		x0 = np.array([-1., -1.])
		super().__init__(x0, F, H, dt, var_w, var_v)

class TimeVarying(BoundedLSProcess):
	""" Smooth interpolation between spiral sink, center, spiral source systems """ 
	def __init__(self, dt: float, var_w: float, var_v: float, f=1/5):
		self.F0 = np.array([[-1.05,-3.60],[1.10, 1.05]])
		self.F1 = np.array([[-1.15864464, -3.68960651], [1.06937006,  0.91600663]])
		self.F2 = np.array([[-0.98092249, -3.67973989], [1.06922538,  1.20720164]])
		def F(t: float):
			a = np.sin(2*np.pi*f*t)
			if a >= 0:
				return (1 - a)*self.F0 + a*self.F1
			else:
				return (1 + a)*self.F0 - a*self.F2
		H = np.eye(2)
		x0 = np.array([-1., -1.])
		super().__init__(x0, F, H, dt, var_w, var_v)

class Saddle(BoundedLSProcess):
	def __init__(self, dt: float, var_w: float, var_v: float):
		F = lambda t: np.array([[-1.,1.],[-1.25, -0.45]])
		H = np.eye(2)
		x0 = np.array([[0.70,-2.55],[-0.10, -2.50]])
		super().__init__(x0, F, H, dt, var_w, var_v)

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	z = Oscillator(0.001, 0.0, 1.0)
	hist_t = []
	hist_z = []
	for _ in range(10000):
		hist_z.append(z())
		hist_t.append(z.t)
	hist_z = np.asarray(hist_z)
	plt.plot(hist_z[:,0], hist_z[:,1])
	plt.show()
