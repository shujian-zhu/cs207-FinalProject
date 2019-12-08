import numpy as np
import autodiff as ad

def gradient_descent(func_lambda, init_value, step_size=0.001, precision=1e-3, max_steps=None, history=False):
    try :
        func_lambda(init_value)
        func=func_lambda
    except :
        try :
            func = lambda arg : func_lambda(*arg)
            func(init_value)
        except Exception as e :
            return("Function and Init_values do not match")

    assert isinstance(func(init_value),(int,float)), "Not a loss function, the output needs to be a Real number"
    init_value_array=np.array(init_value)
    if isinstance(init_value,(int,float)):
        history_GD,grad_norms_GD =_gradient_descent_oneD(func,init_value_array, step_size=step_size, precision=precision, max_steps=max_steps )
    elif isinstance(init_value,(list,np.ndarray)):
        history_GD,grad_norms_GD =_gradient_descent_array(func,init_value_array, step_size=step_size, precision=precision, max_steps=max_steps )
    else :
        raise TypeError("Not Accepting init_value type")
    if history :
        return history_GD[-1],grad_norms_GD[-1],history_GD,grad_norms_GD

    return history_GD[-1],grad_norms_GD[-1]

def _gradient_descent_oneD(loss_func,init_value, step_size=0.001, precision=1e-3, max_steps=None):
    history = [init_value]  # to store all parameters
    counter = 0
    label = 'lambda0'
    new_value = 1.0 * init_value
    f = loss_func(ad.AD_Object(new_value, label=label))
    grad_norms=[ad.derivative(f,label)]
    # Do descent while stopping condition not met
    while abs(grad_norms[-1]) > precision:
        # get gradient of loss function
        gradient =ad.derivative(f,label)
        # take one step in the gradient direction
        new_value = new_value - step_size * gradient
        #update our loss_function value and derivative
        f = loss_func(ad.AD_Object(new_value, label=label))
        # add our new parameters to the history
        history.append(new_value)
        grad_norms.append(ad.derivative(f,label))
        # tick off one more step
        counter += 1
        # if we've hit maximum steps allowed, stop!
        if max_steps is not None:
            if counter == max_steps:
                msg = ("Failed to converge after %d iterations,last value is %0.3f with a gradient norm of %0.3f" %
                       (max_steps + 1, ad.value(f), abs(ad.derivative(f,label))))
                raise RuntimeError(msg)
    return history,grad_norms

def _gradient_descent_array(loss_func,init_value, step_size=0.001, precision=1e-3, max_steps=None ):
    counter = 0
    label = ['lambda%s' % i for i in range(len(init_value))]
    new_value = 1.0 * init_value
    f = loss_func(ad.AD_Vector(new_value, label=label))
    print(f)
    history = [init_value]  # to store all parameters
    grad_norms=[np.linalg.norm([f.der[l] for l in label],2)]
    # Do descent while stopping condition not met
    while grad_norms[-1]> precision:
        # get gradient of loss function
        gradient = np.array([ad.derivative(f,l) for l in label])
        # take one step in the gradient direction
        new_value = new_value - step_size * gradient
        #update our loss_function value and derivative
        f = loss_func(ad.AD_Vector(new_value, label=label))
        # add our new parameters to the history and grad_norms
        history.append(new_value)
        grad_norms.append(np.linalg.norm([f.der[l] for l in label],2))
        # tick off one more step
        counter += 1
        # if we've hit maximum steps allowed, stop!
        if max_steps is not None:
            if counter == max_steps:
                msg = ("Failed to converge after %d iterations,last value is %f with a gradient norm of %f" %
                       (max_steps + 1, ad.value(f), abs(ad.derivative(f, label))))
    return history

def stochastic_gradient_descent(loss_func, init_value, x, y, step_size=0.01,batch_size=32, max_epochs=5,precision=1e-3, history=False,verbose=True):
    assert loss_func.__code__.co_argcount ==3, "The Loss function needs to have 3 inputs f(lambda,x,y)"
    assert isinstance(init_value,(list,np.ndarray))
    assert isinstance(x, (np.ndarray))
    assert isinstance(y, (np.ndarray))

    N=len(x)
    counter = 0
    label = ['lambda%s' % i for i in range(len(init_value))]
    new_value = 1.0 * np.array(init_value)
    history_SGD = [new_value]
    for epoch in range(max_epochs):
        if verbose :
            print("Epoch %s out of %s"%(epoch,max_epochs))
        #Shuffle so that we don't introduce bias
        shuffle = np.random.permutation(N)
        x = x[shuffle]
        y = y[shuffle]
        for i in range(round(N/batch_size)):
            # Define partial loss_func
            partial_loss_func = lambda values: loss_func(values, x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            # for a batch_size number of data_points, compute the loss_function
            f = partial_loss_func(ad.AD_Vector(new_value, label=label))
            # get gradient of loss function
            partial_gradient = np.array([ad.derivative(f, l) for l in label])
            # take one step in the gradient direction
            new_value = new_value - step_size * partial_gradient
            # add our new parameters to the history
            history_SGD.append(new_value)
            # tick off one more step
            counter += 1
            # if we've hit maximum steps allowed, stop!

    full_f=lambda values: loss_func(values, x, y)
    f=full_f(ad.AD_Vector(new_value, label=label))
    if history :
        return new_value,np.array([ad.derivative(f, l) for l in label]),history_SGD

    return new_value,np.array([ad.derivative(f, l) for l in label])

def newton_method(func_lambda, init_value, precision=1e-12,max_steps=1000):
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
    if isinstance(init_value, (int, float)):
        found_value= _newton_method_1D(func, init_value, precision=precision,max_steps=max_steps)
    elif isinstance(init_value, (list, np.ndarray)):
        found_value= _newton_method_array(func, init_value, precision=precision,max_steps=max_steps)
    else :
        raise TypeError("Not Accepting init_value type")

    return found_value


def _newton_method_1D(func_lambda, init_value, precision=1e-12,max_steps=1000):
    counter=0
    label='x0'
    new_value = 1.0*init_value
    f = func_lambda(ad.AD_Object(init_value, label=label))
    try:
        while abs(ad.value(f)) > precision :
            new_value -= ad.value(f)/ad.derivative(f,label)
            f = func_lambda(ad.AD_Object(new_value, label=label))
            counter+=1
            if counter == max_steps :
                msg = ("Failed to converge after %d iterations,last value is %s." % (max_steps + 1, ad.value(f)))
                raise RuntimeError(msg)
    except Exception as e:
        print("Stationary point Encountered")
        print(e)
    return new_value

def _newton_method_array(func, init_value, precision=1e-12, max_steps=1000):
    counter=0
    # For multiple variables
    new_value=np.array(init_value)
    label = ['x%s'%i for i in range(len(init_value))]
    new_value = 1.0*new_value
    f = func(ad.AD_Vector(init_value, label=label))
    if len(f) != len(label) :
        import warnings
        warnings.warn("Matrix not squared: Using Moore Penrose Pseudo-Inverse",UserWarning)
    try:
        while np.linalg.norm(ad.value(f),2) > precision:
            if counter == max_steps:
                msg = ("Failed to converge after %d iterations,last value is %s."% (max_steps + 1, ad.value(f)))
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


func_lambda = lambda lamba0,x,y : lamba0[0]**2*lamba0[1]**2+45 +x.sum()+y.sum()
init_value=[1,2]
x=np.arange(1,100000)
y=np.arange(1,100000)
print(stochastic_gradient_descent(func_lambda, init_value,x,y,history=False))