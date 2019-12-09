import numpy as np
import pytest
import warnings
from AutoDiff import root_finding as rf
from AutoDiff import autodiff as ad


def test_newton_method():
    func_lambda = lambda x: x**2
    init_value=100

    value=rf.newton_method(func_lambda, init_value)
    assert round(value,2) == 0.

def test_newton_method2():
    def func1(x):
        return x**2
    func_lambda = func1
    init_value= 100

    value=rf.newton_method(func_lambda, init_value)
    assert round(value,2) == 0.
    
def test_newton_method3():
    func_lambda = lambda x, y : [x*y**2, x + y -20]
    init_value= [1,3]

    res_x=rf.newton_method(func_lambda, init_value)
    res_y=func_lambda(*res_x)
    
    assert (round(res_x[0],2),res_x[1]) == (0, 20)
    assert (round(res_y[0],2), res_y[1]) == (0, 0)
    
def test_newton_method4():
    def func1(x,y):
        return x**2
    func_lambda = func1
    init_value= 100

    value=rf.newton_method(func_lambda, init_value)
    assert value == "Function and Init_values do not match"
    
    
def test_newton_method5():
    func_lambda = lambda x, y : [x*y**2, x + y -20]
    init_value= [1,3]
    with pytest.raises(RuntimeError):
        res_x=rf.newton_method(func_lambda, init_value, max_steps = 5)
        
def test_newton_method6():
    func_lambda = lambda x: x**2
    init_value=100
    with pytest.raises(RuntimeError):
        res_x=rf.newton_method(func_lambda, init_value, max_steps = 2)
        
def test_newton_method7():
    func_lambda = lambda x, y, z : [z + x*y**2, z + x + y -20]
    init_value= [1,3, 0]

    res_x=rf.newton_method(func_lambda, init_value)
    res_y=func_lambda(*res_x)
    print(res_x)
    print(res_y)
    
    with pytest.warns(UserWarning):
        warnings.warn("Matrix not squared: Using Moore Penrose Pseudo-Inverse", UserWarning)
    
    assert round(res_x[0],2) == -1.54  
    
    
    
def test_secant_method():
    func_lambda = lambda x : x-5
    x0 = -1
    x1 = 10
    
    res = rf.secant_method(func_lambda,x0,x1)
    assert res == 5
    
    
def test_secant_method2():
    def func1(x):
        return x - 5
    func_lambda = func1
    x0 = '-1'
    x1 = 10
    with pytest.raises(TypeError):
        res = rf.secant_method(func_lambda,x0,x1)
        
def test_secant_method3():
    func_lambda = lambda x : x-5
    x0 = -1
    x1 = -1
    
    with pytest.raises(ValueError):
        res = rf.secant_method(func_lambda,x0,x1)
        
def test_secant_method4():
    func_lambda = lambda x : x-5
    x0 = -1
    x1 = 10
    
    with pytest.raises(RuntimeError):
        res = rf.secant_method(func_lambda,x0,x1, max_steps = 1) 
        
        
        

def test_broyden_method():
    func_lambda = lambda x, y: [x + y , 2* (x + y)]
    init_value=np.array([1, 1])
    
    res = rf.broyden_method(func_lambda,init_value,use_der_for_init = False)
    print(res)
    assert (round(res[0],3), round(res[1],3)) == (0.333, -0.333)
    
    
def test_broyden_method2():
    func_lambda = lambda x, y: [x + y , 2* (x + y)]
    init_value=np.array([1, 1])
    with pytest.raises(RuntimeError):
        res = rf.broyden_method(func_lambda,init_value,use_der_for_init = False, max_steps = 10)
        
def test_broyden_method3():
    func_lambda = lambda x, y: [x + y , 2* (x + y)]
    init_value=np.array([1, 1])
    res = rf.broyden_method(func_lambda,init_value,use_der_for_init = True, max_steps = 10)
    assert (round(res[0], 3), round(res[1],0)) == (0,0)
        


