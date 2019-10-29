# Introduction
Evaluation of derivatives is integral to many machine learning methods. For this purpose, two main methods could be used: symbolical and numerical differentiation. Symbolical differentiation, though straightforward, its implementation requires complex expression manipulation in computer algebra systems, making it very costly to evaluate; the method is also limited to closed-form input expressions. On the other hand, numerical differentiation computes the function derivative by approximating it using small values of step size h; though numerical simpler and faster than symbolic diffferentiation, it suffers from stability issues and round-off or truncation errors.

To address the weaknesses of both these methods, automatic differentiation (AD) was introduced. Since, it has been applied in different areas, such as  engineering design optimization, structural mechanics, and atmospheric sciences; its application to machine learning methods popularised AD. Therefore, due to the important role AD plays in many scientific fields, we introduce a python package that provides user-friendly methods for performing forward-mode AD. Our package supports the evaluation of first derivatives of functions defined by user at given input value. 


# Background
All numerical computation can be seen as a combination of elementary operations for which the derivatives are known. The derivatives of the overall composition can be found by combining the derivatives of elementary operations through the chain rule. Such elementary functions include arithmetic operations (addition, subtraction, multiplication, division), sign switch, and functions such as exponential, logarithm, and the trigonometric (e.g. sin(x), cos(x)). Traces of these elementary operations can be represented by a trace table or a computational graph. Trace table are originally used to trace the value of variables as each line of code is executed. As an example of this flow, Table 1 shows the evaluation trace of elementary operations of the computation f(x<sub>1</sub>) = ln(x<sub>1</sub>) + 3 * x<sub>1</sub> and Figure 1 gives an example of a graphic representation of function f(x<sub>1</sub>) by its elementary operations. 

### Table 1

| Trace       | Elementary function | Current Function Value | Function Derivative |
| ------------- |:-------------:|:-------------:|:-------------:|
| X1      | X1            |  c             | 1  |
| X2      | ln(X1)            | ln(c)      | 1/c |
| X3      | 3 * X1            |  3c        | 3   |
| X4      | X2 + X3             | ln(c) + 3c | 1/c + 3 |

![Figure 1](https://github.com/we-the-diff/cs207-FinalProject/blob/milestone1/docs/sample_trace_graph.png)


The forward mode of AD starts from the input value and compute the derivative of intermediate variables with respect to the input value. Applying the chain rule to each elementary operation in the forward primal trace, we generate the corresponding derivative trace, which gives us the derivative in the final variable. Forward  mode AD can also be viewed as evaluating a function using dual numbers, which are defined as a+b*&epsilon;, where a, b &#1013; R and &epsilon; is a nilpotent number such that &epsilon;^2 = 0 and &epsilon; &ne; 0. It can be shown that the coefficient of &epsilon; after evaluating a function is exactly the derivative of that function, which also works for chain rule.


# How to Use PackageName
<pre><code>
#import package
import AutomaticDifferentiation as AD
import elementary_functions as elem

# user defined function
f = lambda x,y : elem.exp(x) + 2*y 

# Set AD tracker for function variables
f_AD = AD(f, initial_values=(x_0,y_0), labels=[‘x’,’y’])

# function value
f_AD

#function derivative wrt x
f_AD.derivative(x) 
</code></pre>

# Software Organization
The package will have the following directory structure:
- README.md
- setup.py
- LICENSE
- test/ (using CodeCov and TravisCI)
- package/
    - \__init\__.py
    - forward_mode.py
    - elementary_functions.py
    - application_AD.py

The package uses standard .py standalone files and setup.py for package installation; it will be also distributed through conda.

# Implementation
We will implement an Automatic Differentiation Class (*Autodiff*) that stores and tracks the value of the given function and its gradient. Class attributes will be:
1. variable_values: initial variable values evaluation ot the function and gradient.
2. function_value: the function value at that variable_values. 
3. derivative_values: the function gradient, derivatives wrt to each declared variable at variable_values.

The *Autodiff* class will implement all relevant dunder methods(\__repr\__, \__add\__, \__mul\__, etc.). In addition, we will implement a “summing_everything_up” function that takes as an input the function and the values at which we want to compute the gradient. This function will create the neural network of elementary operations for the input function and then compute the grad of the composition of those elementary operations that constitute the function, using the chain rule.

For the elementary functions, we will implement all relevent functions from the *math* module of the Python Standard Library, such as (exp, sin, cos, sin, etc.). The functions will be able to handle two types of input: AD objects and numbers. For the first input type , the elem_function will return the function evaluation and derivative; for the second type, it will behave as a math.function and just return the evaluation.

Finally, the package will be designed to have minimal dependencies; we envision only the *numpy* and *math* packages will be needed.
