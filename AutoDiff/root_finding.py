import numpy as np
import AutoDiff as ad


def newton_method(func_lambda, init_value, precision=1e-6,max_steps=10000):
    #Reuse f in order to accept f(array) AND f(float,float,...,float)
    if isinstance(init_value,list):
        init_value=np.array(init_value)
    try :
        func_lambda(init_value)
        func=func_lambda
    except :
        try :
            func = lambda arg : func_lambda(*arg)
            func(init_value)
        except Exception as e :
            return("Function and Init_values do not match")
    # For single variable returning a float
    if isinstance(init_value, (int, float)):
        found_value= _newton_method_1D(func, init_value, precision=precision,max_steps=max_steps)
    elif isinstance(init_value, (list, np.ndarray)):
        found_value= _newton_method_array(func, init_value, precision=precision,max_steps=max_steps)
    else :
        raise TypeError("Not Accepting init_value type")

    return found_value


def _newton_method_1D(func_lambda, init_value, precision=1e-6,max_steps=10000):
    counter=0
    label='x0'
    new_value = 1.0*init_value
    f = func_lambda(ad.AD_Object(init_value, label=label))
    while abs(ad.value(f)) > precision :
        new_value -= ad.value(f)/ad.derivative(f,label)
        f = func_lambda(ad.AD_Object(new_value, label=label))
        counter+=1
        if counter == max_steps :
            msg = ("Failed to converge after %d iterations,last value is %s." % (max_steps + 1, ad.value(f)))
            raise RuntimeError(msg)
        if abs(ad.derivative(f,label)) < precision / 100:
            raise ZeroDivisionError("Stationary point Encountered")
    return new_value

def _newton_method_array(func, init_value, precision=1e-6, max_steps=10000):
    counter=0
    # For multiple variables
    new_value=np.array(init_value)
    label = ['x%s'%i for i in range(len(init_value))]
    new_value = 1.0*new_value
    f = func(ad.AD_Vector(init_value, label=label))

    if isinstance(func(new_value), (list,np.ndarray)) and len(f) != len(label):
        import warnings
        warnings.warn("Matrix not squared: Using Moore Penrose Pseudo-Inverse",UserWarning)

    while np.linalg.norm(ad.value(f),2) > precision:
        if counter == max_steps:
            msg = ("Failed to converge after %d iterations,last value is %s,norm is %s."% (max_steps + 1, ad.value(f),np.linalg.norm(ad.value(f),2)))
            raise RuntimeError(msg)
        jacob=ad.jacobian(f,label=label)
        new_value -= np.dot(np.linalg.pinv(jacob),ad.value(f))
        f = func(ad.AD_Vector(new_value, label=label))
        counter+=1
        if np.linalg.norm(jacob[0],2)<precision/100:
            raise ZeroDivisionError("Stationary point Encountered")

    return new_value

def secant_method(func_lambda,x0,x1, tol=1e-6,max_steps=10000):
    # Reuse f in order to accept f(array) AND f(float,float,...,float)
    try:
        func_lambda(x0)
        func = func_lambda
    except:
        try:
            func = lambda arg: func_lambda(*arg)
            func(x0)
        except Exception as e:
            raise TypeError("Function and Init_values do not match")
    counter=0
    assert type(x0) in [int, float], ("Use broyden_method if multidimensionnal function")
    # Secant method
    if x1 == x0:
        raise ValueError("The initialization values must be different")
    x0, x1 = 1.0 * x0, 1.0 * x1

    if abs(func(x1)) < abs(func(x0)):
        x0, x1= x1, x0

    xn,xn1=x0,x1
    xn2=xn1 #for first loop
    while abs(func(xn2))> tol and counter<max_steps:
        counter+=1
        xn2 = xn1 - func(xn1) * (xn1 - xn) / (func(xn1) - func(xn))
        xn,xn1=xn1,xn2

    if counter==max_steps :
        msg = (" Failed to converge after %d iterations, value is %s."
                % (max_steps, xn2))
        raise RuntimeError(msg)

    return xn2

def broyden_method(func_lambda,init_value, J=None,use_der_for_init=True, tol=1e-6,max_steps=10000):
    # Change f in order to accept f(array) AND f(float,float,...,float)
    try:
        func_lambda(init_value)
        func = func_lambda
    except:
        try:
            func = lambda arg: func_lambda(*arg)
            func(init_value)
        except Exception as e:
            return ("Function and Init_values do not match")

    counter=0

    assert isinstance(init_value,(list,np.ndarray)), ("Use secant_method for one-dimension function")
    new_value=1.0*np.array(init_value)

    if use_der_for_init:
        label = ['x%s' % i for i in range(len(init_value))]
        try :
            f = func(ad.AD_Object(init_value, label=label))
        except :
            try :
                f = func(*ad.AD_Vector(init_value, label=label))
            except:
                f = func(ad.AD_Vector(init_value, label=label))
        J=ad.jacobian(f,label=label)
    elif J is None :
        J=np.identity(len(new_value))

    while np.linalg.norm(func(new_value),2)> tol and counter<max_steps:
        try :
            counter+=1
            delta_values=-np.dot(np.linalg.pinv(J),func(new_value))
            delta_func=np.array(func(new_value+delta_values))-np.array(func(new_value))
            new_value +=delta_values
            J=np.add(J,(1/np.sum(new_value**2))*(delta_func-np.dot(J,delta_values))*delta_values.reshape(-1,1),casting='unsafe')
        except Exception as e:
            print(e)
            raise ValueError("Could not converge")
    if counter==max_steps :
        msg = (
                " Failed to converge after %d iterations, value is %s."
                % (max_steps, func(new_value)))
        raise RuntimeError(msg)
    return new_value


