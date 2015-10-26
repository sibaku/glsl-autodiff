# glsl-autodiff
Automatic differentiation for glsl

This is a very simple implementation for forward mode automatic differentiation in glsl. It provides some macros to make initial callings easier.
So far, the following can be evaluated
* Scalar functions of one variable
* Scalar functions of 2,3,4 variables
* 2,3,4D functions of 2,3,4 variables

Macros are provided for Gradients, Jacobians and Hessians. For vector valued functions, each component has to be described by its own function.
Scalar variables are stored in a vec2, if only the first derivative is desired and in a vec3 if additionally a second derivative is desired.
If a Hessian matrix needs to be calculated, variables are encapsulated in the structs HessNum2/HessNum3/HessNum4. The function you need to evaluate have to use these types for their computation. Implemented functions on these objects are listed below. 

## Operations on expressions

Derivable expressions are either stored in a vec2, vec3 or one of the HessNum2/3/4 structs, as explained above. On these, a number of operations are defined:
* add(a,b): a + b, first and second order derivatives (vec2,vec3 types) can also be added by usual addition: a + b
* sub(a,b): a - b, first and second order derivatives (vec2,vec3 types) can also be subtracted by usual subtraction: a - b
* mult(a,b): a * b, first and second order derivatives (vec2,vec3 types) can be multiplied with a constant factor (e.g. a float) with the usual multiplication from left or right. Mult is overloaded to allow constant factor multiplication from left or right for HessNum
* div(a,b): a / b, first and second order derivatives (vec2,vec3 types) can be divided by a constant factor (e.g. a float) with the usual division with the vector on the left. Div is overloaded to allow constant factor division on the right of HessNum
* a_sin(a): sin(a)
* a_cos(a): cos(a)
* a_exp(a): exp(a)
* a_log(a): log(a)
* a_pow(a,k): a^k, k can only be a float (maybe a more general version will be introduced later)
* a_abs(a): abs(a), the second derivative/Hessian of this may contain infinite values
* a_sqrt(a): sqrt(a)
* a_const(v),a_const2D(v),a_constH2(v),a_constH3(v),a_constH4(v): creates a constant with value v. a_const will only create a vec2 for first order derivatives. a_const2D creates a second order constant and the a_constHx variants create Hessian constants.
* neg(a): -a, creates the negation of the input. For vec2 and vec3 types, you can also use -a

## Creating a variable

Variables have to be created. Macros are provided, so you don't have to do that manually. You can do it as follows:

### First order

For a variable with value val:
```
vec2 x = vec2(val,1.);
```
If you want to compute partial derivatives, you have to set all variables besides the one to be differentiated by as constant. For this and generally for constants you initialize as
```
vec2 x = vec2(val,0.);
```
Constants can also be created by the a_const functions
### Second order
Same as with first order, but with the third component always zero
```
vec3 variable = vec3(val,1.,0.);
vec3 constant = vec3(val,0.,0.);
```
### Hessians
Hessian variables contain a value, the gradient and the Hessian. For variables and constants, the Hessian is zero. For constant the gradient too. For a variable the gradient contains one zero at the variables position in the function
```
HessNum2 firstVariableHessian = HessNum2(firstVariableValue,vec2(1.,0.),mat2(0.));
HessNum2 secondVariableHessian = HessNum2(secondVariableValue,vec2(0.,1.),mat2(0.));
HessNum2 constantHessian = HessNum2(constantValue,vec2(0.),,mat2(0.));
```
The same applies to the higher Hessians, just with the vec2/mat2 replaced by vec3/mat3 and vec4/mat4.

# Example usage
Here are a few basic examples. Of course you can create more complex functions. The nice thing about autodiff is, that you can derive algorithms, for example loops.

## Compute first derivative
Compute the derivative of sin(exp(x))x^3 at x=3

Define variable. Variables have their value at the first position and 1 at the second. 
```
vec2 x = vec2(3.,1.);
```
Compute function
```
vec2 result = mult(a_sin(a_exp(x)),a_pow(x,3.));
```
The function value can now be accessed as the first component and the value of the derivative as the second
```
float funcVal = result.x;
float derivVal = result.y;
```
## Compute second derivative
Same example as before

Define variable. Variables have their value at the first position,1 at the second and 0 at the third
```
vec3 x = vec3(3.,1.,0.);
```
Compute function
```
vec3 result = mult(a_sin(a_exp(x)),a_pow(x,3.));
```
The function value can now be accessed as the first component and the value of the derivative as the second
```
float funcVal = result.x;
float derivVal = result.y;
float secondDerivVal = result.z;
```

## Compute Gradient of 2D function

Define a function of two variables (each a vec2) and returning a vec2. The function is x^3  + 4 y^2
```
vec2 exampleFunc(in vec2 x, in vec2 y)
{
    return add(a_pow(x,3.),4.*mult(y,y));
}
```

Define a placeholder for the gradient in your program flow
```
 vec2 gradExample;
```
Compute gradient. x and y are ordinary floats (variables or constants)
```
 GRAD2_v(exampleFunc,x,y,gradExample);
```

The gradient is now in gradExample. GRAD3_v and GRAD4_v work exactly the same. There are macros to additionally get the value: GRAD2_VALUE_v, GRAD3_VALUE_v, GRAD4_VALUE_v

Specialized gradient numbers GradNum2, GradNum3, GradNum4 exist to make this easier(and maybe faster due to vector operations). You can also retrieve the value in the val field of the GradNum. The gradient is stored in the g field.
```
GradNum2 exampleFuncGN(in GradNum2 x, in GradNum2 y)
{
    return add(a_pow(x,3.),mult(4.,mult(y,y)));
}
```
The usage is as follows:
```
 GradNum2 gradNumExample;
 GRAD2(exampleFuncGN,x,y,gradNumExample);
 
 float value = gradNumExample.val;
 vec2 gradient = gradNumExample.g;
```
## Compute Jacobi matrix 

As an example, we use a 2D function of two variables u and v: f1 = u^2 + v^2*exp(u), f2 = sin(u) - cos(v)^2
First define both component functions
```
vec2 f1(in vec2 u, in vec2 v)
{
	return add(mult(u,u),mult(mult(v,v),a_exp(u)));
}
vec2 f2(in vec2 u, in vec2 v)
{
	return sub(a_sin(u),a_pow(a_cos(v),2.));
}
```
The Jacobian is a 2x2 matrix. To compute its value at position (u,v) (both floats) first declare a placeholder for the result
```
mat2 J;
```
Then call the Jacobian macro
```
JACOBI2_v(f1,f2,u,v,J);
```
The Jacobian is now in J. The rows are the gradients of the component functions. JACOBI3_v and JACOBI4_v work exactly the same. The mixed Jacobians (e.g. JACOBI23_v) will always be stored in a  matrix of the bigger size (e.g. JACOBI23_v -> mat3).

Jacobians for functions that take GradNum variables do not have the _v and are probably preferrable. 
There are also macros to retreive the value. These are called the same as the usual Jacobian, but with _VALUE attached, e.g JACOBI2_VALUE_v and JACOBI2_VALUE

## Compute Hessian
We choose a scalar valued function of three variables: x^2 + sin(cos(y+z))

Since we have three variables, we have to use the HessNum3 type for our function definition.
```
HessNum3 func(in HessNum3 x, in HessNum3 y, in HessNum3 z)
{
	return add(mult(x,x),a_sin(a_cos(add(y,z))));
}
```

Declare result variable
```
HessNum3 r;
```
Call HESSIAN3 macro for float input (x,y,z)
```
HESSIAN3(func,x,y,z,r);
```
The Hessian is now in r. You can acces the value of the function, gradient and Hessian as follows
```
float val = r.val;
vec3 gradient = r.g;
mat3 hessian = r.h;
```
The same procedure applies to HESSIAN2 and HESSIAN4

# Testing/Bugs/Suggestions
So far, some variations have been tested in WebGL and found to be working, though not everything was tested. Bug reports/tips/suggestions are always welcome.
