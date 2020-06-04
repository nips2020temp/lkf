from typing import Callable, Any
import multiprocessing
from tqdm import tqdm
import traceback
import numpy as np
from shortid import ShortId

from utils import *

# Multiprocessing cannot pickle lambdas
_implicit_gen = None

def worker(arg: dict, seed: Any):
	try:
		if seed is not None:
			set_seed(seed)

		z, f = _implicit_gen(arg)

		max_err = 2.

		hist_t = []
		hist_z = []
		hist_x = []
		hist_err = []
		while z.t <= arg['T']:
			z_t = z()
			x_t, err_t = f(z_t)
			hist_t.append(z.t)
			hist_z.append(z_t)
			hist_x.append(x_t)
			hist_err.append(err_t)

			if np.linalg.norm(err_t) > max_err:
				print('Error overflowed!')
				break
		hist_t = np.array(hist_t)
		hist_z = np.array(hist_z)
		hist_x = np.array(hist_x)
		hist_err = np.array(hist_err)

		return {'id': arg['id'], 'hist_t': hist_t, 'hist_z': hist_z, 'hist_x': hist_x, 'hist_err': hist_err}
	except:
		print(traceback.format_exc())
		return None


def pool_execute(args: list, gen_system: Callable, seed=None, reduce_result=lambda x:x):
	global _implicit_gen
	_implicit_gen = gen_system

	idgen = ShortId()
	results = dict()

	with tqdm(total=len(args)) as pbar:
		def add_result(result):
			results[result['id']] = reduce_result(result)
			pbar.update(1)

		with multiprocessing.Pool() as pool:
			for arg in args:
				arg['id'] = idgen.generate()
				pool.apply_async(worker, args=(arg, seed), callback=add_result)
			pool.close()
			pool.join()

	return [results[arg['id']] for arg in args]