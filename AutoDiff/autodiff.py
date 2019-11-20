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

    def __init__(self, init_value):
        if (not isinstance(init_value, int)) and (not isinstance(init_value, float)):
            raise TypeError('input value must be numeric')
        self.val = init_value
        self.der = 1

    def __repr__(self):
        return "AD Object: Value = %.3f, Derivative =%.3f"%(self.val, self.der)

    def __neg__(self):
        result = AD_Object(-1*self.val)
        result.der = -1*self.der
        return result

    def __radd__(self, other):
        return AD_Object.__add__(self, other)

    def __add__(self, other):
        try:
            result = AD_Object(self.val+other.val)
            result.der = self.der + other.der
            return result	
        except AttributeError:
            result = AD_Object(self.val+other)
            result.der = self.der
            return result

    def __rsub__(self, other):
        result = AD_Object(other - self.val)
        result.der = -self.der
        return result

    def __sub__(self, other):
        try:
            result = AD_Object(self.val-other.val)
            result.der = self.der - other.der
            return result	
        except AttributeError:
            result = AD_Object(self.val-other)
            result.der = self.der
            return result

    def productrule(self,other): # both self and other are autodiff objects
        return other.val*self.der + self.val*other.der

    def __rmul__(self, other):
        return AD_Object.__mul__(self, other)

    def __mul__(self, other):
        try: # when both self and other are autodiff object, we implement the product rule
            result = AD_Object(self.val*other.val)
            result.der = self.productrule(other)
            return result
        except: #when other is a constant, not an autodiff object, we simply multiply them
            result = AD_Object(other*self.val)
            result.der = other*self.der
            return result

    def quotientrule(self,other): # both self and other are autodiff objects, and the function is self / other
        return (other.val*self.der - self.val*other.der)/(other.val**2)

    def __truediv__(self, other):
        try:
            if other.val == 0:
                raise ValueError('Cannot divide by 0')                
            result = AD_Object(self.val/other.val)
            result.der = self.quotientrule(other) #when both self and other are autodiff object, implement the quotient rule
            return result
        except AttributeError: #when other is a constant, e.g. f(x) = x/2 -> f'(x) = x'/2 = 1/2
            if other == 0:
                raise ValueError('Cannot divide by 0')    
            result = AD_Object(self.val/other)
            result.der = self.der / other
            return result

    def __rtruediv__(self, other):
        #when other is a constant, e.g. f(x) = 2/x = 2*x^-1 -> f'(x) =  -2/(x^-2)
        if self.val == 0:
            raise ValueError('Cannot divide by 0')         
        result = AD_Object(other / self.val)
        result.der = (-other * self.der)/(self.val**2)
        return result 

    def powerrule(self,other):
        # for when both self and other are autodiff objects
        # in general, if f(x) = u(x)^v(x) -> f'(x) = u(x)^v(x) * [ln(u(x)) * v(x)]'
        if self.val == 0:
            return 0            
        return self.val**other.val * other.productrule(self.ln())

    def __pow__(self, other):
        try: 
            result = AD_Object(self.val**other.val)
            result.der = self.powerrule(other) # when both self and other are autodiff object, implement the powerrule
            return result
        except AttributeError: # when the input for 'other' is a constant
            result = AD_Object(self.val**other)
            result.der = other * (self.val ** (other-1)) * self.der
            return result

    def __rpow__(self, other):
        #when other is a constant, e.g. f(x) = 2^x -> f'(x) =  2^x * ln(2)
        result = AD_Object(other**self.val)
        if other == 0:
            result.der = 0
            return result
        result.der = other**self.val * math.log(other) * self.der
        return result

    def exp(self):
        result = AD_Object(math.exp(self.val))
        result.der = math.exp(self.val) * self.der
        return result


    def ln(self):
        if (self.val) <= 0:
            raise ValueError('log only takes positive number')
        result = AD_Object(math.log(self.val))
        result.der = 1/self.val
        return result


    def sin(self):
            result = AD_Object(math.sin(self.val))
            result.der = math.cos(self.val) * self.der
            return result


    def cos(self):
            result = AD_Object(math.cos(self.val))
            result.der = -1 * math.sin(self.val) * self.der
            return result


    def tan(self):
            result = AD_Object(math.tan(self.val))
            result.der = self.der / math.cos(self.val)**2
            return result