import numpy as np



def newton_method(func_lambda, init_value, tol=1e-12,max_iter=1000):
    import autodiff as ad
    new_value=init_value
    iter=0

    #Reuse f in order to accept f(array) AND f(float,float,...,float)
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
    if type(init_value) in [int,float]:
        label='x0'
        new_value = 1.0*new_value
        f = func(ad.AD_Object(init_value, label=label))
        try:
            while abs(ad.value(f)) > tol :
                if iter == max_iter :
                    msg = ("Failed to converge after %d iterations, value is %s."
                           % (max_iter + 1, ad.value(f)))
                    raise RuntimeError(msg)
                new_value -= ad.value(f)/ad.derivative(f,label)
                f = func_lambda(ad.AD_Object(new_value, label=label))
                iter+=1
        except Exception as e:
            print("Stationary point Encountered")
            print(e)
        return new_value

    # For multiple variables
    new_value=np.array(init_value)
    label = ['x%s'%i for i in range(len(init_value))]
    new_value = 1.0*new_value
    f = func(ad.AD_Vector(init_value, label=label))

    #if len(f) > len(label) :
    #    raise Exception("Newton-Raphson does not work when there is more equations then variables ")

    try:
        while np.linalg.norm(ad.value(f),2) > tol:
            if iter == max_iter:
                msg = ("Failed to converge after %d iterations, value is %s."
                       % (max_iter + 1, ad.value(f)))
                raise RuntimeError(msg)
            new_value -= np.dot(np.linalg.pinv(ad.jacobian(f,label=label)),ad.value(f))
            f = func(ad.AD_Vector(new_value, label=label))
            iter+=1
    except Exception as e:
        print("Stationary point Encountered")
        print(e)

    return new_value

def secant_method(func_lambda,x0,x1, tol=1e-12,max_iter=1000):
    # Reuse f in order to accept f(array) AND f(float,float,...,float)
    try:
        func_lambda(x0)
        func = func_lambda
    except:
        try:
            func = lambda arg: func_lambda(*arg)
            func(x0)
        except Exception as e:
            return ("Function and Init_values do not match")
    iter=0
    assert type(x0) in [int, float], ("Use broyden_method if multidimensionnal function")
    # Secant method
    if x1 == x0:
        raise ValueError("The initialization values must be different")
    x0, x1 = 1.0 * x0, 1.0 * x1

    if abs(func(x1)) < abs(func(x0)):
        x0, x1= x1, x0

    xn,xn1=x0,x1
    xn2=xn1 #for first loop
    while abs(func(xn2))> tol and iter<max_iter:
        iter+=1
        xn2 = xn1 - func(xn1) * (xn1 - xn) / (func(xn1) - func(xn))
        xn,xn1=xn1,xn2

    if iter==max_iter :
        msg = (" Failed to converge after %d iterations, value is %s."
                % (max_iter, xn2))
        raise RuntimeError(msg)

    return xn2

def broyden_method(func_lambda,init_value, J=None,use_der_for_init=True, tol=1e-12,max_iter=1000):
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

    iter=0
    assert type(init_value) not in [int, float], ("Use secant_method for one-dimension function")
    x0 = 1.0 *np.array(init_value)

    if use_der_for_init:
        import autodiff as ad
        label = ['x%s' % i for i in range(len(init_value))]
        try :
            f = func(ad.AD_Object(init_value, label=label))
        except :
            f = func(*ad.AD_Vector(init_value, label=label))
        J=ad.jacobian(f,label=label)
    elif J is None :
        J=np.identity(len(x0))

    new_value=1.0*np.array(init_value)
    while np.linalg.norm(func(*new_value),2)> tol and iter<max_iter:
        iter+=1
        delta_values=-np.dot(np.linalg.pinv(J),func(*new_value))
        delta_func=np.array(func(*(new_value+delta_values)))-np.array(func(*new_value))
        new_value +=delta_values
        J+=(1/np.sum(new_value**2))*np.dot(delta_func-np.dot(np.linalg.pinv(J),delta_values),np.transpose(delta_values))
    if iter==max_iter :
        msg = (
                " Failed to converge after %d iterations, value is %s."
                % (max_iter, func(*new_value)))
        raise RuntimeError(msg)

    return new_value


func_lambda = lambda x,y: [x*y**2,x+y-20]
init_value=np.array([1,3])
print(func_lambda(*init_value))
value=newton_method(func_lambda, init_value)

print("The root is ",value, "        ",func_lambda(*init_value))