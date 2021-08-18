#ifndef HESSNUM_3_H_
#define HESSNUM_3_H_

// This file contains methods to compute the gradient and hessian 
// of a scalar valued 3 dimensional function using automatic forward differentiation

//--------------------------------
// Types
//--------------------------------

// Data type to hold information about a scalar valued 3 dimensional function
// These should be created by the constH3 (for constants) and varH3 (for variables) helpers
struct HNum3
{
    // The current value
    float val;
    // The current gradient
    vec3 g;
    // The current hessian
    mat3 h;
};

//--------------------------------
// Prototypes
//--------------------------------

/**
* Creates a constant HNum3
* @param val The current value of the constant
*/
HNum3 constH3(in float val);
/**
* Creates a HNum3 corresponding to the variable with the given index
* @param val The current value of the variable
* @param index The variable's index
*/
HNum3 varH3(in float val, in int index);
/**
* Creates a HNum3 corresponding to the variable x (index = 0)
* @param val The current value of the variable
*/
HNum3 varH3x(in float val);
/**
* Creates a HNum3 corresponding to the variable y (index = 1)
* @param val The current value of the variable
*/
HNum3 varH3y(in float val);
/**
* Creates a HNum3 corresponding to the variable z (index = 2)
* @param val The current value of the variable
*/
HNum3 varH3z(in float val);
HNum3 add(in HNum3 a, in HNum3 b);
HNum3 add(in HNum3 a, in float b);
HNum3 add(in float a, in HNum3 b);
HNum3 sub(in HNum3 a, in HNum3 b);
HNum3 sub(in HNum3 a, in float b);
HNum3 sub(in float a, in HNum3 b);
HNum3 mult(in HNum3 a, in HNum3 b);
HNum3 mult(in HNum3 a, in float b);
HNum3 mult(in float a, in HNum3 b);
HNum3 neg(in HNum3 a);
HNum3 div(in HNum3 a, in HNum3 b);
HNum3 div(in HNum3 a, in float b);
HNum3 div(in float a, in HNum3 b);
HNum3 inv(in HNum3 a);
HNum3 a_pow(in HNum3 a, in HNum3 b);
HNum3 a_pow(in HNum3 a, in float b);
HNum3 a_pow(in float a, in HNum3 b);
HNum3 a_min(in HNum3 a, in HNum3 b);
HNum3 a_max(in HNum3 a, in HNum3 b);
HNum3 a_exp2(in HNum3 a);
HNum3 a_inversesqrt(in HNum3 a);
HNum3 a_atan(in HNum3 a);
HNum3 a_sqrt(in HNum3 a);
HNum3 a_sinh(in HNum3 a);
HNum3 a_ceil(in HNum3 a);
HNum3 a_tan(in HNum3 a);
HNum3 a_asinh(in HNum3 a);
HNum3 a_asin(in HNum3 a);
HNum3 a_acosh(in HNum3 a);
HNum3 a_abs(in HNum3 a);
HNum3 a_exp(in HNum3 a);
HNum3 a_cosh(in HNum3 a);
HNum3 a_floor(in HNum3 a);
HNum3 a_log(in HNum3 a);
HNum3 a_atanh(in HNum3 a);
HNum3 a_log2(in HNum3 a);
HNum3 a_acos(in HNum3 a);
HNum3 a_tanh(in HNum3 a);
HNum3 a_cos(in HNum3 a);
HNum3 a_sin(in HNum3 a);

//--------------------------------
// Macros
//--------------------------------

#define HESSIAN3(f,x, y, z,result)  {     result = f(varH3x(x), varH3y(y), varH3z(z)); }

//--------------------------------
// Utilities prototypes
//--------------------------------

mat3 a_outerProduct(in vec3 a, in vec3 b);

//--------------------------------
// Implementation
//--------------------------------

HNum3 constH3(in float val)
{
    return HNum3(val, vec3(0.0), mat3(0.0));
}
//--------------------------------
HNum3 varH3(in float val, in int index)
{   
    vec3 g = vec3(0.0);
    g[index] = 1.0;
    return HNum3(val, g, mat3(0.0));
}
//--------------------------------
HNum3 varH3x(in float val)
{   
    vec3 g = vec3(0.0);
    g[0] = 1.0;
    return HNum3(val, g, mat3(0.0));
}
//--------------------------------
HNum3 varH3y(in float val)
{   
    vec3 g = vec3(0.0);
    g[1] = 1.0;
    return HNum3(val, g, mat3(0.0));
}
//--------------------------------
HNum3 varH3z(in float val)
{   
    vec3 g = vec3(0.0);
    g[2] = 1.0;
    return HNum3(val, g, mat3(0.0));
}
//--------------------------------
HNum3 add(in HNum3 a, in HNum3 b)
{
    return HNum3(a.val + b.val , a.g + b.g, a.h + b.h);
}
//--------------------------------
HNum3 add(in HNum3 a, in float b)
{
    return HNum3(a.val + b , a.g, a.h);
}
//--------------------------------
HNum3 add(in float a, in HNum3 b)
{
    return HNum3(a + b.val , b.g, b.h);
}
//--------------------------------
HNum3 sub(in HNum3 a, in HNum3 b)
{
    return HNum3(a.val - b.val , a.g - b.g, a.h - b.h);
}
//--------------------------------
HNum3 sub(in HNum3 a, in float b)
{
    return HNum3(a.val - b , a.g, a.h);
}
//--------------------------------
HNum3 sub(in float a, in HNum3 b)
{
    return HNum3(a - b.val , - b.g, - b.h);
}
//--------------------------------
HNum3 mult(in HNum3 a, in HNum3 b)
{
    return HNum3(a.val * b.val, 
        a.val*b.g + b.val*a.g, 
        a.val*b.h + b.val*a.h + a_outerProduct(b.g,a.g) + a_outerProduct(a.g,b.g)
    );
}
//--------------------------------
HNum3 mult(in HNum3 a, in float b)
{
    return HNum3(a.val * b, b*a.g, b*a.h);
}
//--------------------------------
HNum3 mult(in float a, in HNum3 b)
{
    return HNum3(a * b.val, a*b.g, a*b.h);
}
//--------------------------------
HNum3 neg(in HNum3 a)
{
    return mult(-1.0,a);
}
//--------------------------------
HNum3 div(in HNum3 a, in HNum3 b)
{
    float b1 = b.val;
    float b2 = b1*b1;
    float b3 = b2*b1;

    return HNum3(a.val / b.val , 
        (b.val*a.g - a.val*b.g)/b2, 
        2.0*a.val/b3*a_outerProduct(b.g,b.g) 
        - a.val/b2*b.h
        + a.h/b1 
        - a_outerProduct(b.g/b2, a.g)
        - a_outerProduct(a.g/b2, b.g)
    );
}
//--------------------------------
HNum3 div(in HNum3 a, in float b)
{
    return HNum3(a.val / b, a.g/b, a.h/b);
}
//--------------------------------
HNum3 div(in float a, in HNum3 b)
{
    float b1 = b.val;
    float b2 = b1*b1;
    float b3 = b2*b1;

    return HNum3(a / b.val, 
        -a*b.g/b2, 
        2.0*a/b3*a_outerProduct(b.g,b.g) - a/b2*b.h
    );
}
//--------------------------------
HNum3 inv(in HNum3 a)
{
    return div(1.0, a);
}
//--------------------------------
HNum3 a_pow(in HNum3 a, in HNum3 b)
{
    return a_exp(mult(b,a_log(a)));
}
//--------------------------------
HNum3 a_pow(in HNum3 a, in float b)
{
    // constant exponent -> make special case
    float v = pow(a.val, b); // value f(a(x))
    float da = b*pow(a.val,b-1.0); // first derivative f'(a(x))
    float dda = b*(b-1.0)*pow(a.val,b-2.0); // second derivative f''(a(x))
    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_pow(in float a, in HNum3 b)
{
    return a_exp(mult(b,log(a)));
}
//--------------------------------
HNum3 a_min(in HNum3 a, in HNum3 b)
{
    if(a.val < b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
HNum3 a_max(in HNum3 a, in HNum3 b)
{
    if(a.val > b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
HNum3 a_exp2(in HNum3 a)
{
    float v = exp2(a.val); // value f(a(x))
    float da = log(2.0)*exp2(a.val); // first derivative f'(a(x))
    float dda = log(2.0)*log(2.0)*exp2(a.val); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_inversesqrt(in HNum3 a)
{
    float v = inversesqrt(a.val); // value f(a(x))
    float da = -0.5/pow(sqrt(a.val),3.0); // first derivative f'(a(x))
    float dda = 0.75/pow(sqrt(a.val),5.0); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_atan(in HNum3 a)
{
    float v = atan(a.val); // value f(a(x))
    float da = 1.0/(1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -2.0*a.val/pow(1.0 + a.val * a.val, 2.0); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_sqrt(in HNum3 a)
{
    float v = sqrt(a.val); // value f(a(x))
    float da = 0.5/sqrt(a.val); // first derivative f'(a(x))
    float dda = -0.25/pow(sqrt(a.val),3.0); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_sinh(in HNum3 a)
{
    float v = sinh(a.val); // value f(a(x))
    float da = cosh(a.val); // first derivative f'(a(x))
    float dda = sinh(a.val); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_ceil(in HNum3 a)
{
    float v = ceil(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_tan(in HNum3 a)
{
    float v = tan(a.val); // value f(a(x))
    float da = 1.0 + pow(tan(a.val),2.0); // first derivative f'(a(x))
    float dda = 2.0*tan(a.val)*(1.0 + pow(tan(a.val),2.0)); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_asinh(in HNum3 a)
{
    float v = asinh(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(1.0 + a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_asin(in HNum3 a)
{
    float v = asin(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = a.val/pow(sqrt(1.0 - a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_acosh(in HNum3 a)
{
    float v = acosh(a.val); // value f(a(x))
    float da = 1.0/sqrt(-1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(-1.0 + a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_abs(in HNum3 a)
{
    float v = abs(a.val); // value f(a(x))
    float da = a.val < 0.0 ? -1.0 : 1.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_exp(in HNum3 a)
{
    float v = exp(a.val); // value f(a(x))
    float da = exp(a.val); // first derivative f'(a(x))
    float dda = exp(a.val); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_cosh(in HNum3 a)
{
    float v = cosh(a.val); // value f(a(x))
    float da = sinh(a.val); // first derivative f'(a(x))
    float dda = cosh(a.val); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_floor(in HNum3 a)
{
    float v = floor(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_log(in HNum3 a)
{
    float v = log(a.val); // value f(a(x))
    float da = 1.0/a.val; // first derivative f'(a(x))
    float dda = -1.0/(a.val * a.val); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_atanh(in HNum3 a)
{
    float v = atanh(a.val); // value f(a(x))
    float da = 1.0/(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = 2.0*a.val/pow(1.0 - a.val * a.val,2.0); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_log2(in HNum3 a)
{
    float v = log2(a.val); // value f(a(x))
    float da = 1.0/(a.val * log(2.0)); // first derivative f'(a(x))
    float dda = -1.0/(a.val * a.val * log(2.0)); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_acos(in HNum3 a)
{
    float v = acos(a.val); // value f(a(x))
    float da = -1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(1.0 - a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_tanh(in HNum3 a)
{
    float v = tanh(a.val); // value f(a(x))
    float da = 1.0 - pow(tanh(a.val),2.0); // first derivative f'(a(x))
    float dda = -2.0*tanh(a.val)*(1.0 - pow(tanh(a.val),2.0)); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_cos(in HNum3 a)
{
    float v = cos(a.val); // value f(a(x))
    float da = -sin(a.val); // first derivative f'(a(x))
    float dda = -cos(a.val); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum3 a_sin(in HNum3 a)
{
    float v = sin(a.val); // value f(a(x))
    float da = cos(a.val); // first derivative f'(a(x))
    float dda = -sin(a.val); // second derivative f''(a(x))

    return HNum3(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}

//--------------------------------
// Implementation prototypes
//--------------------------------

mat3 a_outerProduct(in vec3 a, in vec3 b)
{
    return mat3(a * b[0], a * b[1], a * b[2]);
}


#endif // HESSNUM_3_H_