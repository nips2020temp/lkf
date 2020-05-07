''' Learning Kalman-Bucy filter
'''

from typing import Callable
import numpy as np
import scipy.integrate as scint

def lkf_system(t, vars, z_t, err_t, err_tau, F_hat_t, H, Q, R, tau):
	[x_t, P_t] = vars
	H_inv = np.linalg.inv(H)
	eta_t = H_inv@(err_t@err_t.T - err_tau@err_tau.T)@H_inv.T@np.linalg.inv(P_t) / (2*tau)
	F_t = F_hat_t - eta_t
	K_t = P_t@H@np.linalg.inv(R)
	d_x = F_t@x_t + K_t@(z_t - H@x_t)
	d_P = F_t@P_t + P_t@F_t.T + Q - K_t@R@K_t.T
	return [d_x, x_P]

def filter(z: Callable, F_hat: Callable, H: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float, tau: float, t_max: float):
	x_0 = np.linalg.inv(H)@z(0)
	P_0 = x_0@x_0.T
	r = scint.ode(lkf_system).set_integrator('dopri5')
	r.set_initial_value((x_0, P_0), 0.)

	hist_t = [0]
	hist_x = [x_0]
	hist_err = [0]
	while t < t_max:
		err_t = hist_err[-1]
		if tau / dt < len(hist_err):
			err_tau = 0
		else:
			err_tau = hist_err[-int(tau/dt)]
		r.set_f_params(z(t), err_t, err_tau, F_hat(t), H, Q, R, tau)
		r.integrate(r.t + dt)
		t += dt
		[x_t, P_t] = r.y
		hist_t.append(t)
		hist_x.append(x_t)
		hist_err.append(z(t) - H@x_t)

	return hist_t, hist_x, hist_err
