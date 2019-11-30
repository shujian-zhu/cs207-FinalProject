import math
import numpy as np

#=====================================Math functions=====================================================#

def e(x):
    try:
        return math.exp(x)
    except:
        return x.exp()

def ln(x):
    try:
        return math.log(x)
    except:
        return x.ln()
    
def sin(x):
    try:
        return math.sin(x)
    except:
        return x.sin()

def asin(x):
    try:
        return math.asin(x)
    except:
        return x.asin()

def sinh(x):
    try:
        return math.sinh(x)
    except:
        return x.sinh()

def cos(x):
    try:
        return math.cos(x)
    except:
        return x.cos()

def acos(x):
    try:
        return math.acos(x)
    except:
        return x.acos()

def cosh(x):
    try:
        return math.cosh(x)
    except:
        return x.cosh()

def tan(x):
    try:
        return math.tan(x)
    except:
        return x.tan()

def atan(x):
    try:
        return math.atan(x)
    except:
        return x.atan()

def tanh(x):
    try:
        return math.tanh(x)
    except:
        return x.tanh()

def log(x, base):
    try:
        return math.log(x, base)
    except:
        return x.log(base)


#=====================================AD_eval=====================================================#

class AD_eval():
    def __init__(self, func_string, variable_label, init_value):

        assert isinstance(func_string, str), "Input function must be a string"
        
        multiple_variables = isinstance(variable_label, list)
        
        # if we have multiple variables (x,y, ..etc)
        if multiple_variables:
            assert len(variable_label) == len(init_value), "Variable labels must be the same length as initial values"

            for i in range(len(variable_label)):
                assert isinstance(variable_label[i], str), "Variable label must be a string"
                assert isinstance(init_value[i], (int, float)), "Input value must be numeric"
                                        
            self.vars = {variable_label[i]: AD_Object(init_value[i], variable_label[i]) for i in range(len(variable_label))}
            
            if 'exp(' in func_string:
                raise NameError('Please use e(x) instead of exp(x) for exponential function')
            
            for label in variable_label:
                func_string = func_string.replace(label, "self.vars['%s']"%label)
            
            # evaluate function using AD object
            self.f = eval(func_string)
            self.der = self.f.der
            self.val = self.f.val

        else:
            assert isinstance(variable_label, str), "Variable label must be a string"
            assert isinstance(init_value, (int, float)), "Input value must be numeric"

            self.x = AD_Object(init_value, variable_label)

            if 'exp(' in func_string:
                raise NameError('Please use e(x) instead of exp(x) for exponential function')

            # evaluate function using AD object
            self.f = eval(func_string.replace(variable_label, 'self.x'))

        self.der = self.f.der
        self.val = self.f.val


    def __repr__(self):
        der_txt = ["d(%s)= %.3f ; "%(k, self.der[k]) for k in self.der]
        return "AD Object: Value = %.3f, Derivative: %s"%(self.val, "".join(der_txt))

    def derivative(self, label):
        assert isinstance(label, str), "Input label must be string"
        return self.der[label]


#=====================================AD_Vector=====================================================#

def AD_Vector(values, label):
    assert hasattr(values, '__iter__'), "Input values must be iterable"
    return np.array([AD_Object(float(val), label) for val in values])

def value(x):
    if isinstance(x, AD_Object):
        return x.val
    elif hasattr(x, '__iter__'):
        return np.array([k.val for k in x])
    else:
        raise TypeError ("Input must be AD_Object or array of AD_Objects")

def derivative(x, label):
    assert isinstance(label, str), "Input label must be string"
    if isinstance(x, AD_Object):
        return x.der[label]
    elif hasattr(x, '__iter__'):
        return np.array([k.der[label] for k in x])
    else:
        raise TypeError ("Input must be AD_Object or array of AD_Objects")

#=====================================AD_Object=====================================================#

class AD_Object():
    def __init__(self, value, label, der_initial=1):

        assert isinstance(value, (int, float, np.number)), "Input value must be numeric"
        self.val = value 

        if isinstance(label, dict):
            self.label = label
            self.der = der_initial
        
        elif isinstance(label, str):
            if isinstance(der_initial, (float, int)):
                self.label = {label: label}
                self.der = {label: der_initial}
            else:
                raise TypeError("der_initial must be numerical")
        
        else:
            raise TypeError("label must be string")

    def __repr__(self):
        der_txt = ["d(%s)= %.3f ; "%(k, self.der[k]) for k in self.der]
        return "AD Object: Value = %.3f, Derivative: %s"%(self.val, "".join(der_txt))

    def derivative(self, label):
        assert isinstance(label, str), "Input label must be string"
        return self.der[label]

    def __neg__(self):
        return AD_Object(-1*self.val, self.label, {k: (-1*self.der[k]) for k in self.der})


    def __radd__(self, other):
        return AD_Object.__add__(self, other)


    def __add__(self, other):
        if isinstance(other, AD_Object):
            value = self.val + other.val
            der = dict()
            label = dict()
            for key in self.der:
                der[key] = (self.der[key] + other.der[key]) if (key in other.der) else self.der[key]
                label[key] = self.label[key] 
            for key in other.der:
                if key not in der:
                    der[key] =  other.der[key]  
                    label[key] = other.label[key]
            return AD_Object(value, label, der)
        #-----
        return AD_Object(self.val+other, self.label, self.der)


    def __rsub__(self, other):
        return AD_Object(other-self.val, self.label, {k: -1*self.der[k] for k in self.der})


    def __sub__(self, other):
        if isinstance(other, AD_Object):
            value = self.val - other.val
            der = dict()
            label = dict()
            for key in self.der:
                der[key] = (self.der[key] - other.der[key]) if (key in other.der) else self.der[key]
                label[key] = self.label[key] 
            for key in other.der:
                if key not in der:
                    der[key] =  other.der[key]  
                    label[key] = other.label[key]
            return AD_Object(value, label, der)
        #-----
        return AD_Object(self.val-other, self.label, self.der)


    def productrule(self, other, key): # both self and other are autodiff objects
        return (other.val*self.der[key] + self.val*other.der[key]) if (key in other.der) else (other.val*self.der[key])

    def __rmul__(self, other):
        return AD_Object.__mul__(self, other)

    def __mul__(self, other):
        if isinstance(other, AD_Object):
            value = self.val * other.val
            der = dict()
            label = dict()
            for key in self.der:
                der[key] = self.productrule(other, key)
                label[key] = self.label[key] 
            for key in other.der:
                if key not in der:
                    der[key] =  other.productrule(self, key)  
                    label[key] = other.label[key]
            return AD_Object(value, label, der)
        #-----
        return AD_Object(other*self.val, self.label, {k: other*self.der[k] for k in self.der})


    def quotientrule(self, other, key): # both self and other are autodiff objects, and the function is self / other
        return ((other.val*self.der[key] - self.val*other.der[key])/(other.val**2)) if (key in other.der) else (self.der[key]/other.val)

    def __truediv__(self, other):
        if other.val == 0:
            raise ValueError('Cannot divide by 0')                

        if isinstance(other, AD_Object):
            value = self.val/other.val
            der = dict()
            label = dict()
            for key in self.der:
                der[key] = self.quotientrule(other, key)
                label[key] = self.label[key] 
            for key in other.der:
                if key not in der:
                    der[key] = ((-self.val * other.der[key])/(other.val**2))  
                    label[key] = other.label[key]
            return AD_Object(value, label, der)
        #-----
        if other == 0:
            raise ValueError('Cannot divide by 0')  
        return AD_Object(self.val/other, self.label, {k: self.der[k]/other for k in self.der})

    def __rtruediv__(self, other):
        #when other is a constant, e.g. f(x) = 2/x = 2*x^-1 -> f'(x) =  -2/(x^-2)
        if self.val == 0:
            raise ValueError('Cannot divide by 0')         
        return AD_Object(other/self.val, self.label, {k: ((-other * self.der[k])/(self.val**2)) for k in self.der}) 

    def powerrule(self, other, key):
        # for when both self and other are autodiff objects
        # in general, if f(x) = u(x)^v(x) -> f'(x) = u(x)^v(x) * [ln(u(x)) * v(x)]'
        if self.val == 0:
            return 0            
        return self.val**other.val * other.productrule(self.ln(), key)

    def __pow__(self, other):
        # when both self and other are autodiff object, implement the powerrule
        if isinstance(other, AD_Object):
            value = self.val**other.val
            der = dict()
            label = dict()
            for key in self.der:
                der[key] = self.powerrule(other, key)
                label[key] = self.label[key] 
            for key in other.der:
                if key not in der:
                    der[key] =  other.powerrule(self, key)
                    label[key] = other.label[key]
            return AD_Object(value, label, der)
        # when the input for 'other' is a constant
        return AD_Object(self.val**other, self.label, {k: (other * (self.val ** (other-1)) * self.der[k]) for k in self.der})

    def __rpow__(self, other):
        # when other is a constant, e.g. f(x) = 2^x -> f'(x) =  2^x * ln(2)
        if other == 0:
            AD_Object(self.val**other, self.label, {k: 0*self.der[k] for k in self.der})
        #------
        return AD_Object(self.val**other, self.label, {k: (other**self.val * math.log(other) * self.der[k]) for k in self.der})

    def sqrt(self):
        return AD_Object(math.sqrt(self.val), self.label, {k: ( (1 / (2*math.sqrt(self.val)) ) * self.der[k]) for k in self.der})

    def exp(self):
        return AD_Object(math.exp(self.val), self.label, {k: (math.exp(self.val) * self.der[k]) for k in self.der})

    def ln(self):
        if (self.val) <= 0:
            raise ValueError('log only takes positive number')
        return AD_Object(math.log(self.val), self.label, {k: ((1/self.val)*self.der[k]) for k in self.der})

    def log(self, base=math.e):
        if (self.val) <= 0:
            raise ValueError('log only takes positive number')
        if base <= 0:
            raise ValueError('log base must be a positive number')
        return AD_Object(math.log(self.val, base), self.label, {k: ((1/(self.val*math.log(base)))*self.der[k]) for k in self.der})

    def sin(self):
        return AD_Object(math.sin(self.val), self.label, {k: (math.cos(self.val) * self.der[k]) for k in self.der})

    def asin(self):
        return AD_Object(math.asin(self.val), self.label, {k: ((1 / math.sqrt(1 - self.val**2)) * self.der[k]) for k in self.der})

    def sinh(self):
        return AD_Object(math.sinh(self.val), self.label, {k: (math.cosh(self.val) * self.der[k]) for k in self.der})

    def cos(self):
        return AD_Object(math.cos(self.val), self.label, {k: (-1 * math.sin(self.val) * self.der[k]) for k in self.der})

    def acos(self):
        return AD_Object(math.acos(self.val), self.label, {k: ((-1 / math.sqrt(1 - self.val**2)) * self.der[k]) for k in self.der})

    def cosh(self):
        return AD_Object(math.cosh(self.val), self.label, {k: (math.sinh(self.val) * self.der[k]) for k in self.der})

    def tan(self):
        return AD_Object(math.tan(self.val), self.label, {k: (self.der[k] / math.cos(self.val)**2) for k in self.der})

    def atan(self):
        return AD_Object(math.atan(self.val), self.label, {k: ((1 / (1 + self.val**2)) * self.der[k]) for k in self.der})

    def tanh(self):
        return AD_Object(math.tanh(self.val), self.label, {k: ((2 / (1 + math.cosh(2*self.val))) * self.der[k]) for k in self.der})

    def sigmoid(self, b_0=1, b_1=1):
        def calc_s(x, b_0, b_1):
            # Sigmoid/Logisitic = 1 / 1 + exp(- (b_0 + b_1*x))
            return (1 / (1+math.exp(-(b_0 + b_1*x))))
        return AD_Object(calc_s(self.val, b_0, b_1), self.label, {k: (calc_s(self.val, b_0, b_1)*(1-calc_s(self.val, b_0, b_1))) * self.der[k]) for k in self.der})