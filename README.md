# glsl-autodiff

Automatic differentiation for glsl

This is a very simple implementation for forward mode automatic differentiation in glsl. It provides some macros to make initial callings easier.
So far, the following can be evaluated

* Scalar functions of one variable 
* Scalar functions of 2,3,4 variables
* 2,3,4D functions of 2,3,4 variables

Macros are provided for Gradients, Jacobians and Hessians. For vector valued functions, each component has to be described by its own function.
Scalar variables are stored in a vec2, if only the first derivative is desired and in a vec3 if additionally a second derivative is desired.
To compute gradients, use the GNum2/GNum3/GNum4 types.
If a Hessian matrix needs to be calculated, variables are encapsulated in the structs HNum2/HNum3/HNum4. The function you need to evaluate have to use these types for their computation. Implemented functions on these objects are listed below.

## Operations on expressions

Differentiable expressions are either stored in a vec2, vec3 or one of the GNum2/3/4 or HNum2/3/4 structs, as explained above. On these, a number of operations are defined:

* var\*(v): Different functions with a postfix corresponding to their type. Used to create variables (identity functions) for that type
* const\*(v): Different functions with a postfix corresponding to their type. Used to create constant value for that type
* add(a,b): a + b, first and second order derivatives (vec2,vec3 types) can also be added by usual addition: a + b
* sub(a,b): a - b, first and second order derivatives (vec2,vec3 types) can also be subtracted by usual subtraction: a - b
* mult(a,b): a * b, first and second order derivatives (vec2,vec3 types) can be multiplied with a constant factor (e.g. a float) with the usual multiplication from left or right. Mult is overloaded to allow constant factor multiplication from left or right for HNum
* div(a,b): a / b, first and second order derivatives (vec2,vec3 types) can be divided by a constant factor (e.g. a float) with the usual division with the vector on the left. Div is overloaded to allow constant factor division on the right of HNum
* a_min(a, b)
* a_max(a, b)
* a_exp2(a)
* a_inversesqrt(a)
* a_atan(a)
* a_sqrt(a)
* a_sinh(a)
* a_ceil(a)
* a_tan(a)
* a_asinh(a)
* a_asin(a)
* a_acosh(a)
* a_abs(a)
* a_exp(a)
* a_cosh(a)
* a_floor(a)
* a_log(a)
* a_atanh(a)
* a_log2(a)
* a_acos(a)
* a_tanh(a)
* a_cos(a)
* a_sin(a)
* a_atan2(y,x)
* a_mix(a,b,t)
* a_pow(a,b): Due to the GLSL implementation of pow, the result is always undefined for a < 0
* neg(a): -a, creates the negation of the input. For vec2 and vec3 types, you can also use -a
* inv(a): 1/a

For functions that are not differentiable at certain points (such as min, max, abs) no error value will be generated to prevent unpredictable code behavior. min for example will choose the smaller function to return with its derivatives. If both functions have the same value, one is still chosen with its derivatives, which might not exist for the minimum. Basically, one side is extended into the non-differentiable point.

## Creating a variable

Variables have to be created explicitely. You can do it as follows:

### First order (Scalar univariate functions)

For a variable with value val:

```glsl
vec2 x = varD1(val);
// or manually
vec2 x = vec2(val,1.0);
```

For a constant with value val:

```glsl
vec2 x = constD1(val);
// or manually
vec2 x = vec2(val, 0.0);
```

Constants can also be created by the a_const functions

### Second order (Scalar univariate functions)

Same as with first order, but with the third component always zero

```glsl
vec3 variable = varD2(val);
vec3 constant = constD2(val);
// or manually
vec3 variable = vec3(val, 1.0, 0.0);
vec3 constant = vec3(val, 0.0, 0.0);
```

### Gradients (Scalar multivariate functions)

Hessian variables contain a value, the gradient and the Hessian. For variables and constants, the Hessian is zero. For constants the gradient is too. For a variable the gradient contains one zero at the variable's position in the function signature, i.e. in f(x,y,z,w) z has the index 2, counting from 0.
The following shows an example for 2D functions. For 3D and 4D only the naming changes (GNum3, GNum4, constG3, varH4w, ...).

```glsl
// varG(dim) functions create variables
// Create variable with specified index
GNum2 firstVar = varG2(firstValue, 0);
GNum2 secondVar = varG2(secondValue, 1);
// Create with named variable instead of index
// Indices and names are as follows:
// 0->x, 1->y, 2->z, 3->w
// varH(dim)(name)
GNum2 firstVar = varG2x(firstValue);
GNum2 secondVar = varG2y(secondValue);
// constG(dim) functions create constants
GNum2 constVar = constG2(constValue);
// or manually
// For higher dimensions, the vec dimension also changes accordingly
GNum2 firstVar = GNum2(firstValue, vec2(1.0, 0.0));
GNum2 secondVar = GNum2(secondValue, vec2(0.0, 1.0));
GNum2 constVar =  GNum2(constValue, vec2(0.0));
```

### Hessians (Scalar multivariate functions)

Gradient variables contain a value and the gradient. For constants the gradient is zero. For a variable the gradient contains one zero at the variable's position in the function signature, i.e. in f(x,y,z,w) z has the index 2, counting from 0.
The following shows an example for 2D functions. For 3D and 4D only the naming changes (HNum3, HNum4, constH3, varH4w, ...).

```glsl
// varH(dim) functions create variables
// Create variable with specified index
HNum2 firstVar = varH2(firstValue,0);
HNum2 secondVar = varH2(secondValue,1);
// Create with named variable instead of index
// Indices and names are as follows:
// 0->x, 1->y, 2->z, 3->w
// varH(dim)(name)
HNum2 firstVar = varH2x(firstValue);
HNum2 secondVar = varH2y(secondValue);
// constH(dim) functions create constants
HNum2 constVar = constH2(constValue);
// or manually
// For higher dimensions, the vec and mat dimension also changes accordingly
HNum2 firstVar = HNum2(firstValue, vec2(1.0, 0.0), mat2(0.0));
HNum2 secondVar = HNum2(secondValue, vec2(0.0, 1.0), mat2(0.0));
HNum2 constVar =  HNum2(constValue, vec2(0.0), mat2(0.0));
```

## Example usage

Here are a few basic examples. Of course you can create more complex functions. The nice thing about autodiff is, that you can derive algorithms, for example loops.

### Compute first derivative

Compute the derivative of sin(exp(x))x^3 at x=3

Define variable. Variables have their value at the first position and 1 at the second.

```glsl
vec2 x = varD1(3.0);
```

Compute the function

```glsl
vec2 result = mult(a_sin(a_exp(x)), a_pow(x, 3.0));
```

The function value can now be accessed as the first component and the value of the derivative as the second

```glsl
float funcVal = result.x;
float derivVal = result.y;
```

### Compute second derivative

Same example as before

Define variable. Variables have their value at the first position,1 at the second and 0 at the third

```glsl
vec3 x = varD2(3.0);
```

Compute the function

```glsl
vec3 result = mult(a_sin(a_exp(x)), a_pow(x, 3.0));
```

The function value can now be accessed as the first component and the value of the derivative as the second

```glsl
float funcVal = result.x;
float derivVal = result.y;
float secondDerivVal = result.z;
```

### Compute Gradient of 2D function

Define a function of two variables. The function is x^3  + 4 y^2

Specialized gradient numbers GNum2, GNum3, GNum4 exist to make this easier. You can also retrieve the value in the val field of the GNum. The gradient is stored in the g field.

```glsl
GNum2 exampleFuncGN(in GNum2 x, in GNum2 y)
{
    return add(a_pow(x,3.),mult(4.,mult(y,y)));
}
```

The usage is as follows:

```glsl
GNum2 gNumExample;
// x and y are the values (float)
GRAD2(exampleFuncGN, x, y, gNumExample);

float value = gNumExample.val;
vec2 gradient = gNumExample.g;
// You can also compute this easily manually without a macro
gNumExample = exampleFuncGN(varG2x(x), varG2y(y));
```

### Compute Jacobi matrix

As an example, we use a 2D function of two variables u and v: f1 = u^2 + v^2*exp(u), f2 = sin(u) - cos(v)^2
First define both component functions

```glsl
GNum2 f1(in GNum2 u, in GNum2 v)
{
 return add(mult(u, u), mult(mult(v, v), a_exp(u)));
} 
GNum2 f2(in GNum2 u, in GNum2 v)
{
 return sub(a_sin(u), a_pow(a_cos(v), 2.0));
}
```

The Jacobian is a 2x2 matrix. To compute its value at position (u,v) (both floats) first declare a placeholder for the result

```glsl
mat2 J;
```

Then call the Jacobian macro

```glsl
JACOBI2(f1,f2,u,v,J);
```

The Jacobian is now in J. The rows are the gradients of the component functions. The macro just computes those gradients and stores them in the rows for you. JACOBI3 and JACOBI4 work exactly the same, as do the mixed Jacobians (e.g. JACOBI23, JACOBI42). The result matrix can be any matrix that is large enough to hold the Jacobian. JACOBI(m)(n) corresponds to a resulting mxn matrix.

There are also macros to retreive the value of the computed functions. These are called the same as the usual Jacobian, but with _VALUE attached, e.g JACOBI2_VALUE and JACOBI2_VALUE. The result will be stored in a vector or array at least the size of the number of functions (m).

## Compute Hessian

We choose a scalar valued function of three variables: x^2 + sin(cos(y+z))

Since we have three variables, we have to use the HNum3 type for our function definition.

```glsl
HNum3 func(in HNum3 x, in HNum3 y, in HNum3 z)
{
 return add(mult(x,cx),ca_sin(a_cos(add(y,cz))));
}
```

Declare result variable

```glsl
HNum3 r;
```

Call HESSIAN3 macro for float input (x,y,z)

```glsl
HESSIAN3(func,x,y,z,r);
```

This is equivalent to:

```glsl
r = func(varH3x(x), varH3y(y), varH3z(z)); 
```

The Hessian is now in r. You can acces the value of the function, gradient and Hessian as follows

```glsl
float val = r.val;
vec3 gradient = r.g;
mat3 hessian = r.h;
```

The same procedure applies to HESSIAN2 and HESSIAN4

## Changes from version 1

* Missing math functions added
* Renamed HessianNum to HNum to make it shorter
* Renamed GradNum to GNum to make it shorter
* Removed gradient/Jacobi macros for scalar derivative types. Use GNum instead
* All code is now generated via script
* For easier usage, code is split/duplicated in different units
  * deriv_1 contains first order functions
  * deriv_2 contains second order functions
  * grad_(d) contains (d) dimensional GNum
  * hessian_(d) contains (d) dimensional HNum
  * \*_all contains everything with the same prefix, i.e. hessian_all contains all HNum (HNum2, HNum3, HNum4)
  * autodiff contains everything

## Testing/Bugs/Suggestions

So far, some variations though not all have been tested in WebGL and found to be working. Bug reports/tips/suggestions are always welcome.
