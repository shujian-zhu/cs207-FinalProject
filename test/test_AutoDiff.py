import math
import pytest
from AutoDiff import autodiff as ad
from AutoDiff import *

################# Test functions #####################

def test_negation():
    ad1 = ad.AD_eval('-x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (-1, -2)

def test_add():
    ad1 = ad.AD_eval('x + 2', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (1,4)

def test_add2():
    ad1 = ad.AD_eval('x + x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (2,4)
    
def test_add3():
    x = ad.AD_Object(2, "x", 1)
    y = ad.AD_Object(3, "y", 1)
    f = 2*x**2 + y 
    assert (f.derivative('x'), f.derivative('y'), f.val) == (8, 1, 11)

def test_radd():
    ad1 = ad.AD_eval('2 + x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (1,4)
    
def test_sub():
    ad1 = ad.AD_eval('x - 2', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (1,0)

def test_sub2():
    ad1 = ad.AD_eval('x - 2*x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (-1,-2)
    
def test_sub3():
    x = ad.AD_Object(2, "x", 1)
    y = ad.AD_Object(3, "y", 1)
    f = 2*x**2 - y 
    assert (f.derivative('x'), f.derivative('y'), f.val) == (8, 1, 5)

def test_rsub():
    ad1 = ad.AD_eval('2 - x', "x", 2)
    assert (ad1.derivative('x'), ad1.val) == (-1,0)

def test_mul():
    ad1 = ad.AD_eval('x * 2', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (2,4)

def test_mul2():
    ad1 = ad.AD_eval('x * x', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (4,4)
    
def test_mul3():
    x = ad.AD_Object(2, "x", 1)
    y = ad.AD_Object(3, "y", 1)
    f = x*y 
    assert (f.derivative('x'), f.derivative('y'), f.val) == (3, 2, 6)

def test_rmul():
    ad1 = ad.AD_eval('2 * x', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (2,4)

def test_truediv():
    ad1 = ad.AD_eval('x/2', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (1/2,1)

def test_truediv2():
    ad1 = ad.AD_eval('(x+2)/x', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (-1/2,2)
    
def test_truediv3():
    x = ad.AD_Object(2, "x", 1)
    y = ad.AD_Object(1, "y", 1)
    f = x/y 
    assert (f.derivative('x'), f.derivative('y'), f.val) == (1, -2, 2)

def test_rtruediv():
    ad1 = ad.AD_eval('3/x', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (-3/4,3/2)

def test_pow():
    ad1 = ad.AD_eval('x**3', 'x', 2)
    assert (ad1.derivative('x'), ad1.val) == (12,8)
def test_pow2():
    ad1 = ad.AD_eval('x**x','x',2)
    assert (ad1.derivative('x'), ad1.val) == (4+ 4*math.log(2),4)
    
def test_pow3():
    x = ad.AD_Object(2, "x", 1)
    y = ad.AD_Object(1, "y", 1)
    f = x**y 
    assert (f.derivative('x'), f.derivative('y'), f.val) == (1, 2*math.log(2), 2)

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
    ad2 = ad.AD_eval('e(2*x + 1)','x',1)
    assert (ad1.derivative('x'), ad1.val) == (math.exp(2),math.exp(2))
    assert (ad2.derivative('x'), ad2.val) == (2* math.exp(3),math.exp(3))
    
def test_log():
    ad1 = ad.AD_eval('log(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (0.5, math.log(2))

def test_sin():
    ad1 = ad.AD_eval('sin(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (math.cos(2), math.sin(2))
    
def test_arcsin():
    ad1 = ad.AD_eval('arcsin(x)','x',1/2)
    assert (ad1.derivative('x'),ad1.val) == (1 / math.sqrt(1 - 0.5**2),math.asin(1/2))
    
def test_sinh():
    ad1 = ad.AD_eval('sinh(x)','x',2)
    assert (ad1.derivative('x'), ad1.val) == (math.cosh(2), math.sinh(2))

def test_cos():
    ad1 = ad.AD_eval('cos(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (-math.sin(2), math.cos(2))
    
def test_arccos():
    ad1 = ad.AD_eval('arccos(x)','x',0.5)
    assert (ad1.derivative('x'), ad1.val) == (-1 / math.sqrt(1 - 0.25),math.acos(0.5))
    
def test_cosh():
    ad1 = ad.AD_eval('cosh(x)','x',2)
    assert (ad1.derivative('x'), ad1.val) == (math.sinh(2),math.cosh(2))

def test_tan():
    ad1 = ad.AD_eval('tan(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (1/ math.cos(2)**2, math.tan(2))
    
def test_arctan():
    ad1 = ad.AD_eval('arctan(x)','x', 2)
    assert (ad1.derivative('x'), ad1.val) == (1 / 5,math.atan(2))
    
def test_tanh():
    ad1 = ad.AD_eval('tanh(x)','x',2)
    assert (ad1.derivative('x'), ad1.val) == (2 / (1 + math.cosh(4)),math.tanh(2))
    
#def test_log():
#    ad1 = ad.AD_eval('log(x,2)','x',2)
#    assert (ad1.derivative('x'), ad1.val) == (1/(2*math.log(2)), math.log(2,2))
    
def test_log2():
    with pytest.raises(ValueError):
        ad.AD_eval('log(x)','x',0)
        
#def test_log3():
#    with pytest.raises(ValueError):
#        ad.AD_eval('log(x,0)','x',1)
 
def test_sigmoid():
    ad1 = ad.AD_eval('sigmoid(x)','x',2)
    val = 1.0/(1.0 + math.exp(-2))
    assert (round(ad1.derivative('x'),5), ad1.val) == (round(val*(1-val),5),val)

    
def test_sqrt():
    ad0 = ad.AD_eval('sqrt(x)','x', 4)
    ad1 = ad.AD_Object(4, 'x').sqrt()
    assert (ad0.derivative('x'), ad0.val) == (1/4,2)
    assert (ad1.der['x'], ad1.val) == (1/4,2)
    
def test_jacobian():
    ad1 = ad.AD_Object(1,'x', 2)
    ad2 = ad.AD_Object(1, {'x':'x', 'y':'y'},{'x':4, 'y':5})
    ad3 = ad.AD_Object(1, {'x':'x', 'y':'y'},{'x':6, 'y':7})\
    
    assert ad.jacobian(ad1, ['x']) == [2]
    res = ad.jacobian([ad1, ad2, ad3], ['x','y'])
    print(np.sum(res - np.array([[2, 0],[4,5],[6,7]])))
    assert np.sum(res - np.array([[2, 0],[4,5],[6,7]])) == 0
    with pytest.raises(TypeError):
        ad.jacobian(res, 'x')
    


############ Test AD_eval class assertions ################

def test_repr_AD_eval():
     ad1 = repr(ad.AD_eval('x', 'x', 1))
     val = 1
     der = 1
     assert ad1 == "AD Object: Value = %.3f, Derivative: d(x)= %.3f ; "%(val, der)
     
def test_AD_eval_multiple1_var1():
    with pytest.raises(AssertionError):
        ad.AD_eval('2*x+y', ['x', 'y'],[1])

def test_AD_eval_multiple1_var2():
    with pytest.raises(AssertionError):
        ad.AD_eval('2*x+y', ['x', 'y'],[1, '2'])

def test_AD_eval_multiple1_var3():
    with pytest.raises(NameError):
        ad.AD_eval('exp(x)+y', ['x', 'y'],[1,2])
        
def test_AD_eval_multiple1_var4():
    ad1 = ad.AD_eval('x**2+y', ['x', 'y'],[1,2])
    assert (ad1.der['x'], ad1.der['y'], ad1.val) == (2,1,3)
    
############ Test AD_Vector class assertions ################

def test_AD_Vector_iterable():
    with pytest.raises(AssertionError):
        ad1 = AD_Vector(1, 'x')
def test_AD_Vector():
    x = ad.AD_Vector(np.arange(1,10), label='x')
    z = x**2
    der = ad.derivative(z,'x')
    val = ad.value(z)
    print(der)
    for i in range(1,10):
        assert der[i-1] == 2*i
        assert val[i-1] == i**2 
        
def test_AD_Vector2():
    x = ad.AD_Vector(np.arange(1,5), label='x')
    y = ad.AD_Vector(np.arange(1,5), label='y')
    z = AD_FuncVector([2*x + ad.e(y),x**2*y]) 
    assert ad.value(z) == [[2+np.exp(1), 4+np.exp(2), 6+np.exp(3), 8+np.exp(4)],[1,8,27,64]]
    assert ad.derivative(z, 'x') == [[2,2,2,2], [2,8,18,32]]
    assert ad.derivative(z, 'y') == [[np.exp(1), np.exp(2), np.exp(3), np.exp(4)],[1,4,9,16]]
    




###########  Test AD_object assertions and comparisions ###################
def test_input_AD_Object():
    with pytest.raises(AssertionError):
        ad.AD_Object(value = '1', label = 'x')
        
def test_AD_object_input1():
    x = 1
    with pytest.raises(TypeError):
        ad1 = ad.AD_Object(value = 1, label = x, der_initial= 1)
        
def test_AD_object_input2():
    with pytest.raises(TypeError):
        ad1 = ad.AD_Object(value = 1, label = 'x', der_initial = '1')
        
def test_AD_object_derivative():
    with pytest.raises(AssertionError):
        ad1 = ad.AD_Object(1,'x')
        ad1.derivative(1)

def test_AD_object_eq1():
    ad1 = ad.AD_Object(1,'x',1)
    with pytest.raises(AssertionError):
        ad1 == 1

def test_AD_object_eq2():
    ad1 = ad.AD_Object(1,'x',1)
    ad2 = ad.AD_Object(2,'x',1)
    assert((ad1 == ad2) == False)
    
def test_AD_object_eq3():
    ad1 = ad.AD_Object(1, 'x', 1)
    ad2 = ad.AD_Object(1, 'y', 1)
    assert((ad1 == ad2) == False)
    
def test_AD_object_eq4():
    ad1 = ad.AD_Object(1, 'x', 2)
    ad2 = ad.AD_Object(1, 'x', 4)
    assert((ad1 == ad2) == False)
    
def test_AD_object_ne():
    ad1 = ad.AD_Object(1,'x',1)
    ad1 == ad1
    assert((ad1 != ad1) == False)
    
def test_AD_object_lt():
    ad1 = ad.AD_Object(1,'x',1)
    ad2 = ad.AD_Object(2,'x',1)
    assert(ad1 < ad2)
    
def test_AD_object_lt2():
    ad1 = AD_Object(1,'x',1)
    with pytest.raises(AssertionError):
        ad1 < 1
    
def test_AD_object_gt():
    ad1 = ad.AD_Object(1,'x',1)
    ad2 = ad.AD_Object(2,'x',1)
    assert(ad2 > ad1)
    
def test_AD_object_gt2():
    ad1 = ad.AD_Object(1,'x',1)
    with pytest.raises(AssertionError):
        ad1 > 1

def test_AD_object_le():
    ad1 = ad.AD_Object(1,'x',1)
    ad2 = ad.AD_Object(2,'x',1)
    assert(ad1 <= ad2)
    
def test_AD_object_le2():
    ad1 = ad.AD_Object(1,'x',1)
    with pytest.raises(AssertionError):
        ad1 <= 1
    
def test_AD_object_ge():
    ad1 = ad.AD_Object(1,'x',1)
    ad2 = ad.AD_Object(2,'x',1)
    assert(ad2 >= ad1)
    
def test_AD_object_ge2():
    ad1 = ad.AD_Object(1,'x',1)
    with pytest.raises(AssertionError):
        ad1 >= 1

def test_repr_AD_Object():
     ad1 = repr(ad.AD_Object(1, 'x'))
     val = 1
     der = 1
     assert ad1 == "AD Object: Value = %.3f, Derivative: d(x)= %.3f ; "%(val, der)

def test_input_function_types():
    x = 2
    with pytest.raises(AssertionError):
        ad.AD_eval(3*x+2, "x", 2)

def test_input_label_types():
    x = 2
    with pytest.raises(AssertionError):
        ad.AD_eval('3x+2', x, 2)

def test_input_value_types():
    with pytest.raises(AssertionError):
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
        ad.AD_eval('log(-1*x)', 'x', 3)
        
def test_exponential_function_name():
    with pytest.raises(NameError):
        ad.AD_eval('3*log(5)/exp(x)', 'x', 3)


