import math
import pytest
import numpy as np

from AutoDiff import autodiff as ad
from AutoDiff import gradient_descent

################# Test functions #####################

def test_initvalue():
	assert abs(gradient_descent(lambda x: x**2, 1, precision=1e-3)[0]) <= 1e-3
	assert abs(gradient_descent(lambda x: x**2, 1, precision=1e-3)[1]) <= 1e-3


# def test_gd:
# 	Loss_function = lambda l1, l2:L(l1, l2, x, y)
# 	init_value=[2,2]
# 	optim_lambdas,grad_optim,history_GD,grad_norms_GD=rt.gradient_descent(Loss_function, init_value, step_size=0.01, precision=1e-4, max_steps=1e6, history=True)
