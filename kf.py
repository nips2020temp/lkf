''' Discretization of Kalman-Bucy filter
'''

from typing import Callable
import numpy as np
import scipy.integrate as scint

def kf_system(t, vars, z_t, F_t, H, Q, R):
	[x_t, P_t] = vars
	K_t = P_t@H@np.linalg.inv(R)
	d_x = F_t@x_t + K_t@(z_t - H@x_t)
	d_P = F_t@P_t + P_t@F_t + Q - K_t@R@K_t.T
	return [d_x, x_P]

def filter(z: Callable, F: Callable, H: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float, t_max: float):
	x_0 = np.linalg.inv(H)@z(0)
	P_0 = x_0@x_0.T
	r = scint.ode(kf_system).set_integrator('dop853')
	r.set_initial_value((x_0, P_0), 0.)

	hist_t = [0]
	hist_x = [x_0]
	hist_err = [0]
	while t < t_max:
		r.set_f_params(z(t), F(t), H, Q, R)
		r.integrate(r.t + dt)
		t += dt
		[x_t, P_t] = r.y
		hist_t.append(t)
		hist_x.append(x_t)
		hist_err.append(z(t) - H@x_t)

	return hist_t, hist_x, hist_err
