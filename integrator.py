''' Stochastic integrators, based on https://github.com/mattja/sdeint/

	Mirrors the scipy.integrate.ode format
'''

from typing import Callable
import numpy as np

class Integrator:
	def __init__(self, f: Callable, g: Callable, ndim: int):
		'''
		Scipy-like euler-maryuama integrator.
			f: system function of the form f(t, state, **args); returns (d x 1)
			g: noise function taking g(t, **args) to apply to Wiener process; returns (d x d)
			ndim: dimension of state & Wiener process

		Column-major representation (state is (ndim x 1))
		'''
		self.ndim = ndim
		self.f = f
		self.g = g
		self.f_params = ()
		self.g_params = ()

	def set_initial_value(self, x0: np.ndarray, t0: float = 0.0):
		self.t = t0
		self.y = x0

	def set_f_params(self, *args):
		self.f_params = args

	def set_g_params(self, *args):
		self.g_params = args

	def integrate(self, t: float):
		dt = t - self.t
		dw = np.random.normal(0.0, np.sqrt(dt), self.ndim)
		dy = self.f(self.t, self.y, *self.f_params) * dt + dw @ self.g(self.t, self.y, *self.g_params).T
		self.t += dt
		self.y += dy


# TODO: stochastic Runge-Kutta 2nd-order integrator



