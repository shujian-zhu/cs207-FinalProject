import pytest
import numpy as np

from AutoDiff import autodiff as ad
from AutoDiff import gradient_descent, stochastic_gradient_descent

################# Test functions #####################

def test_graddescent_1D():
	assert abs(gradient_descent(lambda x: x**2, 1, precision=1e-3)[0]) <= 1e-3
	assert abs(gradient_descent(lambda x: x**2, 1, precision=1e-3)[1]) <= 1e-3

def test_graddescent_2D():
	L = lambda l1, l2, x, y: 0.000045 * l2**2 * y.sum() - 0.000098 * l1**2 * x.sum() +0.003926 * x.sum() * l1 * np.exp(-0.1 * (l1**2 + l2**2))
	x=np.array([i if i%2 else i-1 for i in range(-10000,10001)]+[i if i%2 else i-1 for i in range(-10000,10001)])/100
	y=np.array([-i if i%2 else -i-1 for i in range(-10000,10001)]+[i if i%2 else i-1 for i in range(-10000,10001)])/100
	Loss_function = lambda l1, l2:L(l1, l2, x, y)
	init_value=[2,2]
	precision = 1e-3
	range_result1 = [2.45-precision, 2.45+precision]
	range_result2 = [0-precision, 0+precision]
	assert (gradient_descent(Loss_function, init_value, precision=precision)[0][0] >= range_result1[0]) or (gradient_descent(Loss_function, init_value, precision=precision)[0][0] <= range_result1[1])
	assert (gradient_descent(Loss_function, init_value, precision=precision)[0][1] >= range_result2[0]) or (gradient_descent(Loss_function, init_value, precision=precision)[0][1] <= range_result2[1])

def test_stoch():
	L = lambda l1, l2, x, y: 0.000045 * l2**2 * y.sum() - 0.000098 * l1**2 * x.sum() +0.003926 * x.sum() * l1 * np.exp(-0.1 * (l1**2 + l2**2))

	x=np.array([i if i%2 else i-1 for i in range(-10000,10001)]+[i if i%2 else i-1 for i in range(-10000,10001)])/100
	y=np.array([-i if i%2 else -i-1 for i in range(-10000,10001)]+[i if i%2 else i-1 for i in range(-10000,10001)])/100
	init_value=[2,2]
	Loss_function_bis = lambda init_value,x ,y :L(init_value[0],init_value[1],x,y)
	assert (stochastic_gradient_descent(Loss_function_bis, init_value,x ,y)[0][0] <= 2.45) or (stochastic_gradient_descent(Loss_function_bis, init_value,x ,y)[0][0] >= 2.0)


