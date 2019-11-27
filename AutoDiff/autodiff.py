import math

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
def cos(x):
    try:
        return math.cos(x)
    except:
        return x.cos()
def tan(x):
    try:
        return math.tan(x)
    except:
        return x.tan()
    

class AD():
    def __init__(self, func_string, variable_label, init_value):
        if type(func_string) != str:
            raise TypeError('input function must be a string')
        if type(variable_label) != str:
            raise TypeError('variable name must be a string')
        if (not isinstance(init_value, int)) and (not isinstance(init_value, float)):
            raise TypeError('input value must be numeric')
        self.var_label = variable_label
        self.func_label = func_string
        self.init_value = init_value

        self.x = AD_Object(self.init_value)
        if 'eself.xp' in func_string.replace(self.var_label, 'self.x'):
            raise NameError('Please use e(x) instead of exp(x) for exponential function')
        self.f = eval(func_string.replace(self.var_label, 'self.x'))
        self.der = self.f.der
        self.val = self.f.val
    
    def __repr__(self):
        return "AD Object: Value = %.3f, Derivative =%.3f"%(self.val, self.der)


class AD_Object():

    def __init__(self, value, label, der_initial=1):
        self.val = value 
        if isinstance(label, dict):
            self.label = label
            self.der = der_initial
        elif isinstance(value, int) or (isinstance(value, float)):
            self.label = {label: label}
            self.der = {label: der_initial}
        else:
            raise TypeError('input value must be numeric')


    def __repr__(self):
        der_txt = ["d(%s)= %.3f ; "%(k, self.der[k]) for k in self.der]
        return "AD Object: Value = %.3f, Derivative: %s"%(self.val, "".join(der_txt))


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


    def exp(self):
        return AD_Object(math.exp(self.val), self.label, {k: (math.exp(self.val) * self.der[k]) for k in self.der})


    def ln(self):
        if (self.val) <= 0:
            raise ValueError('log only takes positive number')
        return AD_Object(math.log(self.val), self.label, {k: ((1/self.val)*self.der[k]) for k in self.der})


    def sin(self):
        return AD_Object(math.sin(self.val), self.label, {k: (math.cos(self.val) * self.der[k]) for k in self.der})


    def cos(self):
        return AD_Object(math.cos(self.val), self.label, {k: (-1 * math.sin(self.val) * self.der[k]) for k in self.der})


    def tan(self):
        return AD_Object(math.tan(self.val), self.label, {k: (self.der[k] / math.cos(self.val)**2) for k in self.der})
