import math

class AD():
	def __init__(self, func_string, variable, a):
		self.var_label = variable
		self.func_label = func_string
		self.a = a

		self.x = AutoDiff1(self.a)
		self.f = eval(func_string.replace(self.var_label, 'self.x'))
		self.der = self.f.der
		self.val = self.f.val

class AutoDiff1():

	def __init__(self, a):
		self.val = a
		self.der = 1

	def __radd__(self, other):
		return AutoDiff1.__add__(self, other)

	def __add__(self, other):
		try:
			result = AutoDiff1(self.val+other.val)
			result.der = self.der + other.der
			return result	
		except AttributeError:
			result = AutoDiff1(self.val+other)
			result.der = self.der
			return result

	def __rsub__(self, other):
		return AutoDiff1.__sub__(self, other)

	def __sub__(self, other):
		try:
			result = AutoDiff1(self.val-other.val)
			result.der = self.der - other.der
			return result	
		except AttributeError:
			result = AutoDiff1(self.val-other)
			result.der = self.der
			return result

	def productrule(self,other): # both self and other are autodiff objects
		return other.val*self.der + self.val*other.der

	def __rmul__(self, other):
		return AutoDiff1.__mul__(self, other)
		
	def __mul__(self, other):
		try: # when both self and other are autodiff object, we implement the product rule
			result = AutoDiff1(self.val*other.val)
			result.der = self.productrule(other)
			return result
		except: #when other is a constant, not an autodiff object, we simply multiply them
			result = AutoDiff1(other*self.val)
			result.der = other*self.der
			return result

	def quotientrule(self,other): # both self and other are autodiff objects, and the function is self / other
		return (other.val*self.der - self.val*other.der)/(other.val**2)
		
	def __truediv__(self, other):
		try:
			result = AutoDiff1(self.val/other.val)
			result.der = self.quotientrule(other) #when both self and other are autodiff object, implement the quotient rule
			return self.quotientrule(other)
		except AttributeError: #when other is a constant, e.g. f(x) = x/2 -> f'(x) = x'/2 = 1/2
			result = AutoDiff1(self.val/other)
			result.der = self.der / other
			return result

	def __rtruediv__(self, other):
		try: #when both self and other are autodiff object, we implement the quotient rule (through __truediv__)
			return self.__truediv__(other)
		except AttributeError: #when other is a constant, e.g. f(x) = 2/x = 2*x^-1 -> f'(x) =  -2/(x^-2)
			return other*AutoDiff1.__pow__(self, -1)
	
	def powerrule(self,other):
		# for when both self and other are autodiff objects
		# in general, if f(x) = u(x)^v(x) -> f'(x) = u(x)^v(x) * [ln(u(x)) * v(x)]'
		return self.val**other.val * other.productrule(self.ln())

	def __pow__(self, other):
		try: 
			print(self.val,other.val)
			result = AutoDiff1(self.val**other.val)
			result.der = self.powerrule(other) # when both self and other are autodiff object, implement the powerrule
			return result
		except AttributeError: # when the input for 'other' is a constant
			result = AutoDiff1(self.val**other)
			result.der = other * (self.val ** (other-1)) * self.der
			return result

	def __rpow__(self, other):
		try: #when both self and other are autodiff object, we implement the powerrule (through __pow__)
			return self.__pow__(other)
		except AttributeError: #when other is a constant, e.g. f(x) = 2^x -> f'(x) =  2^x * ln(2)
			result = AutoDiff1(other**self.val)
			result.der = result.val * math.log(other)
			return result
			
	def exp(self):
		try:
			result = AutoDiff1(math.exp(self.val))
			result.der = math.exp(self.val) * self.der
			return result	
		except:
			print ("this function does not allow user to pass any inputs. Try x = AutoDiff1(10); x.exp()")


	def ln(self):
		try:
			result = AutoDiff1(math.log(self.val))
			result.der = 1/self.val
			return result	
		except AttributeError:
			print ("this function does not allow user to pass any inputs. Try x = AutoDiff1(10); x.ln()")


	def sin(self):
		try:
			result = AutoDiff1(math.sin(self.val))
			result.der = math.cos(self.val) * self.der
			return result
		except AttributeError:
			print ("this function does not allow user to pass any inputs. Try x = AutoDiff1(10); x.sin()")



#demo

ad1 = AD('2*x**2 + 3*x + 5', 'x', 2)
print("val: ",ad1.val,"\nder: ", ad1.der)

# a = 2.0 
# x = AutoDiff1(a)
# print(x.val, x.der)

# f1 = 2*x + 3
# print("2*x + 3")
# print(f1.val, f1.der)

# f2 = 3*x
# print("3*x")
# print(f2.val, f2.der)

# f3 = f1+f2
# print("2*x + 3 + 3*x")
# print(f3.val, f3.der)

# f4 = f2 - 3
# print("3*x - 3")
# print(f4.val, f4.der)

# f5 = x / 2
# print("x/2")
# print(f5.val, f5.der)

# f6 = 2*x**3
# print("2*x**3")
# print(f6.val, f6.der)

# f7 = (x**2).exp()
# print("x**2.exp()")
# print(f7.val, f7.der) 

# f7 = 2/x
# print("2/x")
# print(f7.val, f7.der) 

# f8 = x**f2
# print("x^(3x)")
# print(f8.val, f8.der)


