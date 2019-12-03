import math
import pytest
from AutoDiff import autodiff as ad

def test_negation():
    ad1 = ad.AD_eval('-x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (-1, -2)

def test_add():
    ad1 = ad.AD_eval('x + 2', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (1,4)

def test_add2():
    ad1 = ad.AD_eval('x + x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (2,4)

def test_radd():
    ad1 = ad.AD_eval('2 + x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (1,4)
    
def test_sub():
    ad1 = ad.AD_eval('x - 2', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (1,0)

def test_sub2():
    ad1 = ad.AD_eval('x - 2*x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (-1,-2)

def test_rsub():
    ad1 = ad.AD_eval('2 - x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (-1,0)

def test_mul():
    ad1 = ad.AD_eval('x * 2', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (2,4)

def test_mul2():
    ad1 = ad.AD_eval('x * x', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (4,4)

def test_rmul():
    ad1 = ad.AD_eval('2 * x', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (2,4)

def test_truediv():
    ad1 = ad.AD_eval('x/2', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (1/2,1)

def test_truediv2():
    ad1 = ad.AD_eval('(x+2)/x', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (-1/2,2)

def test_rtruediv():
    ad1 = ad.AD_eval('3/x', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (-3/4,3/2)

def test_pow():
    ad1 = ad.AD_eval('x**3', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (12,8)
def test_pow2():
    ad1 = ad.AD_eval('x**x','x',2)
    assert (ad1.derivative('x'), ad1.val) == (4+ 4*math.log(2),4)

def test_rpow():
    ad1 = ad.AD_eval('3**x','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (9*math.log(3),9)
    
def test_rpow2():
    ad1 = ad.AD_eval('0**x','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (0,0)
    
def test_powerrule():
    ad1 = ad.AD_eval('x**(x+1)','x', 0)
    assert (ad1.derivative('x'), ad1.val) == (0,0)
    
def test_exp():
    ad1 = ad.AD_eval('e(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (math.exp(2),math.exp(2))

def test_ln():
    ad1 = ad.AD_eval('ln(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (0.5, math.log(2))

def test_sin():
    ad1 = ad.AD_eval('sin(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (math.cos(2), math.sin(2))

def test_cos():
    ad1 = ad.AD_eval('cos(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (-math.sin(2), math.cos(2))

def test_tan():
    ad1 = ad.AD_eval('tan(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (1/ math.cos(2)**2, math.tan(2))
    
def test_input_function_types():
    x = 2
    with pytest.raises(AssertionError):
        ad.AD_eval(3*x+2, "x", 2)

def test_input_label_types():
    x = 2
    with pytest.raises(TypeError):
        ad.AD_eval('3x+2', x, 2)

def test_input_value_types():
    with pytest.raises(TypeError):
        ad.AD_eval('3x+2', 'x', '3')
        
def test_division_by_zero():
    with pytest.raises(ValueError):
        ad.AD_eval('x/0', 'x', 3)
        
def test_division_by_zero2():
    with pytest.raises(ValueError):
        ad.AD_eval('1/x', 'x', 0)
        
def test_division_by_zero3():
    with pytest.raises(ValueError):
        ad.AD_eval('(x+1)/x', 'x', 0)
        
def test_ln_with_negative():
    with pytest.raises(ValueError):
        ad.AD_eval('ln(-1*x)', 'x', 3)
        
def test_exponential_function_name():
    with pytest.raises(NameError):
        ad.AD_eval('3*ln(5)/exp(x)', 'x', 3)

def test_input_AD_Object():
    with pytest.raises(TypeError):
        ad.AD_Object('1')

# def test_repr_AD():
#     ad1 = repr(ad.AD_eval('x', 'x', 1))
#     val = 1
#     der = 1
#     assert ad1 == "AD Object: Value = %.3f, Derivative =%.3f"%(val, der)

# def test_repr_AD_Object():
#     ad1 = repr(ad.AD_Object(1))
#     val = 1
#     der = 1
#     assert ad1 == "AD Object: Value = %.3f, Derivative =%.3f"%(val, der)