import math
import numpy as np

#=====================================Elementary functions=====================================================#

def e(x):
    #try: 
    return np.exp(x)
    #except: 
    #    return x.exp()
    
def sin(x):
    #try:
    return np.sin(x)
    #except:
    #    return x.sin()

def arcsin(x):
    #try:
    return np.arcsin(x)
    #except:
    #    return x.arcsin()

def sinh(x):
    #try:
    return np.sinh(x)
    #except:
    #    return x.sinh()

def cos(x):
    #try:
    return np.cos(x)
    #except:
    #    return x.cos()

def arccos(x):
    #try:
    return np.arccos(x)
    #except:
    #    return x.arccos()

def cosh(x):
    #try:
    return np.cosh(x)
    #except:
    #   return x.cosh()

def tan(x):
    #try:
    return np.tan(x)
    #except:
    #    return x.tan()

def arctan(x):
    #try:
    return np.arctan(x)
    #except:
    #    return x.arctan()

def tanh(x):
    #try:
    return np.tanh(x)
    #except:
    #    return x.tanh()

#def ln(x):
#     try: 
#         return np.log(x)
#     except: 
#         return x.ln()

def log(x):
    #try:
    return np.log(x)
    #except:
    #   return x.log()

def sigmoid(x, b_0=0, b_1=1):
    #try:
    return (1 / (1+np.exp(-(b_0 + b_1*x))))
    #except:
    #    return x.sigmoid(b_0, b_1)

def sqrt(x):
    #try:
    return np.sqrt(x)
    #except:
    #    return x.sqrt()


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
            self.label = variable_label

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
            self.label = variable_label


    def __repr__(self):
        der_txt = ["d(%s)= %.3f ; "%(k, self.der[k]) for k in self.der]
        return "AD Object: Value = %.3f, Derivative: %s"%(self.val, "".join(der_txt))

    def derivative(self, label):
        assert isinstance(label, str), "Input label must be string"
        return self.der[label]


#=====================================AD_Vector=====================================================#

def AD_Vector(values, label): #Vector Input Values
    assert hasattr(values, '__iter__'), "Input values must be iterable"
    if type(label)==str :
        return np.array([AD_Object(float(val), label) for val in values])
    else :
        return np.array([AD_Object(float(val), label[i]) for i,val in enumerate(values)])

def value(x):
    if isinstance(x, AD_Object):
        return x.val
    elif hasattr(x, '__iter__'):
        try: #for single function with vector input values
            return np.array([k.val for k in x])
        except: #for vector function with vector input values
            temp = []
            for k in x:
                temp.append([l.val for l in k])
            return temp
    else:
        raise TypeError ("Input must be AD_Object or array of AD_Objects")

def derivative(x, label):
    assert isinstance(label, str), "Input label must be string"
    if isinstance(x, AD_Object):
        return x.der[label]
    elif hasattr(x, '__iter__'):
        try: #for single function with vector input values
            return np.array([k.der[label] for k in x])
        except: #for vector function with vector input values
            temp = []
            for k in x:
                temp.append([l.der[label] for l in k])
            return temp
    else:
        raise TypeError ("Input must be AD_Object or array of AD_Objects")

def jacobian(x,label):
    if isinstance(x, AD_Object):
        return np.array(list(x.der.values()))
    elif hasattr(x, '__iter__'):
        jacob=[]
        for k in x :
            if not isinstance(k, AD_Object):
                raise TypeError ("Input must be AD_Object or array of AD_Objects")
            df_i = []
            for l in label :
                try :
                    df_i.append(k.der[l])
                except :
                    df_i.append(0)
            jacob.append(np.array(df_i))
        return np.array(jacob)
    else:
        raise TypeError ("Input must be AD_Object or array of AD_Objects")


def AD_FuncVector(func:list): #Vector Functions
    assert hasattr(func, '__iter__'), "Input function must be iterable"
    return [f for f in func]


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
               
        if isinstance(other, AD_Object):
            if other.val == 0:
                raise ValueError('Cannot divide by 0') 

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
        return self.val**other.val * other.productrule(self.log(), key)

    def __pow__(self, other):
        # when both self and other are autodiff object, implement the powerrule
        if isinstance(other, AD_Object):
            value = self.val**other.val
            der = dict()
            label = dict()
            for key in self.der:
                if key in other.label:
                    der[key] = self.powerrule(other, key)
                    label[key] = self.label[key]
                else:
                    der[key] = other.val * (self.val ** other.val - 1) * self.der[key]
                    
            for key in other.der:
                if key in der:
                    continue # skip the variables already in der{}
                # The following code will only be run when ohter.key not in self.key 
                # for example: f = x ** y 
                der[key] = self.val**other.val * np.log(self.val) #k^x -> k^x * ln(k)
                label[key] = other.label[key]
                
            return AD_Object(value, label, der)
        # when the input for 'other' is a constant
        return AD_Object(self.val**other, self.label, {k: (other * (self.val ** (other-1)) * self.der[k]) for k in self.der})

    def __rpow__(self, other):
        # when other is a constant, e.g. f(x) = 2^x -> f'(x) =  2^x * ln(2)
        if other == 0:
            return AD_Object(other**self.val, self.label, {k: 0*self.der[k] for k in self.der})
        #------
        return AD_Object(other**self.val, self.label, {k: (other**self.val * np.log(other) * self.der[k]) for k in self.der})

    def sqrt(self):
        return AD_Object(np.sqrt(self.val), self.label, {k: ( (1 / (2*np.sqrt(self.val)) ) * self.der[k]) for k in self.der})

    def exp(self):
        return AD_Object(np.exp(self.val), self.label, {k: (np.exp(self.val) * self.der[k]) for k in self.der})

    def log(self):
        if (self.val) <= 0:
            raise ValueError('log only takes positive number')
        return AD_Object(np.log(self.val), self.label, {k: ((1/self.val)*self.der[k]) for k in self.der})

    # def log(self, base=math.e):
    #     if (self.val) <= 0:
    #         raise ValueError('log only takes positive number')
    #     if base <= 0:
    #         raise ValueError('log base must be a positive number')
    #     return AD_Object(math.log(self.val, base), self.label, {k: ((1/(self.val*math.log(base)))*self.der[k]) for k in self.der})

    def sin(self):
        return AD_Object(np.sin(self.val), self.label, {k: (np.cos(self.val) * self.der[k]) for k in self.der})

    def arcsin(self):
        return AD_Object(np.arcsin(self.val), self.label, {k: ((1 / np.sqrt(1 - self.val**2)) * self.der[k]) for k in self.der})

    def sinh(self):
        return AD_Object(np.sinh(self.val), self.label, {k: (np.cosh(self.val) * self.der[k]) for k in self.der})

    def cos(self):
        return AD_Object(np.cos(self.val), self.label, {k: (-1 * np.sin(self.val) * self.der[k]) for k in self.der})

    def arccos(self):
        return AD_Object(np.arccos(self.val), self.label, {k: ((-1 / np.sqrt(1 - self.val**2)) * self.der[k]) for k in self.der})

    def cosh(self):
        return AD_Object(np.cosh(self.val), self.label, {k: (np.sinh(self.val) * self.der[k]) for k in self.der})

    def tan(self):
        return AD_Object(np.tan(self.val), self.label, {k: (self.der[k] / np.cos(self.val)**2) for k in self.der})

    def arctan(self):
        return AD_Object(np.arctan(self.val), self.label, {k: ((1 / (1 + self.val**2)) * self.der[k]) for k in self.der})

    def tanh(self):
        return AD_Object(np.tanh(self.val), self.label, {k: ((2 / (1 + np.cosh(2*self.val))) * self.der[k]) for k in self.der})

    def sigmoid(self, b_0=1, b_1=1):
        def calc_s(x, b_0, b_1):
            # Sigmoid/Logisitic = 1 / 1 + exp(- (b_0 + b_1*x))
            return (1 / (1+np.exp(-(b_0 + b_1*x))))
        return AD_Object(calc_s(self.val, b_0, b_1), self.label, {k: ((calc_s(self.val, b_0, b_1)*(1-calc_s(self.val, b_0, b_1))) * self.der[k]) for k in self.der})

    def __eq__(self, other):
        assert isinstance(other, AD_Object), "Input must be an AD_object"
        
        #check function value
        if self.val != other.val: 
            return False

        #check input variable ('label')
        self_label = list(set(sorted(self.label.keys())))
        other_label = list(set(sorted(other.label.keys())))
        for k in range(len(self_label)):
            if (self_label[k] != other_label[k]):
                return False

        #check derivative of each input variable
        for k in self_label:
            if self.der[k] != other.der[k]:
                return False

        #if it passed all the checks above, return True
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other): #this only compares the function value
        assert isinstance(other, AD_Object), "Input must be an AD_object"
        return (self.val < other.val)

    def __gt__(self, other): #this only compares the function value
        assert isinstance(other, AD_Object), "Input must be an AD_object"
        return (self.val > other.val)

    def __le__(self, other): #this only compares the function value
        assert isinstance(other, AD_Object), "Input must be an AD_object"
        return (self.val <= other.val)

    def __ge__(self, other): #this only compares the function value
        assert isinstance(other, AD_Object), "Input must be an AD_object"
        return (self.val >= other.val)

