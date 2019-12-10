import AutoDiff as ad
import numpy as np


def gradient_descent(func_lambda, init_value, step_size=0.001, precision=1e-3, max_steps=1e5, history=False):
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
                       (max_steps + 1, ad.value(f), grad_norms[-1]))
    return history,grad_norms

def stochastic_gradient_descent(loss_func, init_value, x, y, step_size=0.01,batch_size=32, max_epochs=5,history=False,verbose=True):
    assert loss_func.__code__.co_argcount ==3, "The Loss function needs to have 3 inputs f(lambdas,x,y)"
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

    full_f=lambda values: loss_func(values, x, y)
    f=full_f(ad.AD_Vector(new_value, label=label))
    if history :
        return new_value,np.array([ad.derivative(f, l) for l in label]),history_SGD

    return new_value,np.array([ad.derivative(f, l) for l in label])
