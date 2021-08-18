#ifndef GRADNUM_2_H_
#define GRADNUM_2_H_

// This file contains methods to compute the gradient of a scalar valued 2 dimensional
// function using automatic forward differentiation

//--------------------------------
// Types
//--------------------------------

// Data type to hold information about a scalar valued 2 dimensional function
// These should be created by the constG2 (for constants) and varG2 (for variables) helpers
struct GNum2
{
    // The current value
    float val;
    // The current gradient
    vec2 g;
};

//--------------------------------
// Prototypes
//--------------------------------

/**
* Creates a constant GNum2
* @param val The current value of the constant
*/
GNum2 constG2(in float val);
GNum2 varG2(in float val, in int index);
GNum2 varG2x(in float val);
GNum2 varG2y(in float val);
GNum2 add(in GNum2 a, in GNum2 b);
GNum2 add(in GNum2 a, in float b);
GNum2 add(in float a, in GNum2 b);
GNum2 sub(in GNum2 a, in GNum2 b);
GNum2 sub(in GNum2 a, in float b);
GNum2 sub(in float a, in GNum2 b);
GNum2 mult(in GNum2 a, in GNum2 b);
GNum2 mult(in GNum2 a, in float b);
GNum2 mult(in float a, in GNum2 b);
GNum2 neg(in GNum2 a);
GNum2 div(in GNum2 a, in GNum2 b);
GNum2 div(in GNum2 a, in float b);
GNum2 div(in float a, in GNum2 b);
GNum2 inv(in GNum2 a);
GNum2 a_pow(in GNum2 a, in GNum2 b);
GNum2 a_pow(in GNum2 a, in float b);
GNum2 a_pow(in float a, in GNum2 b);
GNum2 a_min(in GNum2 a, in GNum2 b);
GNum2 a_max(in GNum2 a, in GNum2 b);
GNum2 a_exp2(in GNum2 a);
GNum2 a_inversesqrt(in GNum2 a);
GNum2 a_atan(in GNum2 a);
GNum2 a_sqrt(in GNum2 a);
GNum2 a_sinh(in GNum2 a);
GNum2 a_ceil(in GNum2 a);
GNum2 a_tan(in GNum2 a);
GNum2 a_asinh(in GNum2 a);
GNum2 a_asin(in GNum2 a);
GNum2 a_acosh(in GNum2 a);
GNum2 a_abs(in GNum2 a);
GNum2 a_exp(in GNum2 a);
GNum2 a_cosh(in GNum2 a);
GNum2 a_floor(in GNum2 a);
GNum2 a_log(in GNum2 a);
GNum2 a_atanh(in GNum2 a);
GNum2 a_log2(in GNum2 a);
GNum2 a_acos(in GNum2 a);
GNum2 a_tanh(in GNum2 a);
GNum2 a_cos(in GNum2 a);
GNum2 a_sin(in GNum2 a);

//--------------------------------
// Macros
//--------------------------------

#define GRAD2(f,x, y,result)  {     result = f(varG2x(x), varG2y(y)); }
//--------------------------------
#define JACOBI2(f1, f2, x, y, result)  {     GNum2 gradResult;  	GRAD2(f1, x, y, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1];  	GRAD2(f2, x, y, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1];  }
//--------------------------------
#define JACOBI2_VALUE(f1, f2, x, y, result, value)  {     GNum2 gradResult;  	GRAD2(f1, x, y, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1];  	GRAD2(f2, x, y, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1];  }
//--------------------------------
#define JACOBI32(f1, f2, f3, x, y, result)  {     GNum2 gradResult;  	GRAD2(f1, x, y, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1];  	GRAD2(f2, x, y, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1];  	GRAD2(f3, x, y, gradResult);     result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1];  }
//--------------------------------
#define JACOBI32_VALUE(f1, f2, f3, x, y, result, value)  {     GNum2 gradResult;  	GRAD2(f1, x, y, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1];  	GRAD2(f2, x, y, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1];  	GRAD2(f3, x, y, gradResult); 	value[2] = gradResult.val; 	result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1];  }
//--------------------------------
#define JACOBI42(f1, f2, f3, f4, x, y, result)  {     GNum2 gradResult;  	GRAD2(f1, x, y, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1];  	GRAD2(f2, x, y, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1];  	GRAD2(f3, x, y, gradResult);     result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1];  	GRAD2(f4, x, y, gradResult);     result[0][3] = gradResult.g[0]; 	result[1][3] = gradResult.g[1];  }
//--------------------------------
#define JACOBI42_VALUE(f1, f2, f3, f4, x, y, result, value)  {     GNum2 gradResult;  	GRAD2(f1, x, y, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1];  	GRAD2(f2, x, y, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1];  	GRAD2(f3, x, y, gradResult); 	value[2] = gradResult.val; 	result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1];  	GRAD2(f4, x, y, gradResult); 	value[3] = gradResult.val; 	result[0][3] = gradResult.g[0]; 	result[1][3] = gradResult.g[1];  }

//--------------------------------
// Implementation
//--------------------------------

GNum2 constG2(in float val)
{
    return GNum2(val, vec2(0.0));
}
//--------------------------------
GNum2 varG2(in float val, in int index)
{   
    vec2 g = vec2(0.0);
    g[index] = 1.0;
    return GNum2(val, g);
}
//--------------------------------
GNum2 varG2x(in float val)
{   
    vec2 g = vec2(0.0);
    g[0] = 1.0;
    return GNum2(val, g);
}
//--------------------------------
GNum2 varG2y(in float val)
{   
    vec2 g = vec2(0.0);
    g[1] = 1.0;
    return GNum2(val, g);
}
//--------------------------------
GNum2 add(in GNum2 a, in GNum2 b)
{
    return GNum2(a.val + b.val, a.g + b.g);
}
//--------------------------------
GNum2 add(in GNum2 a, in float b)
{
    return GNum2(a.val + b, a.g);
}
//--------------------------------
GNum2 add(in float a, in GNum2 b)
{
    return GNum2(a + b.val, b.g);
}
//--------------------------------
GNum2 sub(in GNum2 a, in GNum2 b)
{
    return GNum2(a.val - b.val, a.g - b.g);
}
//--------------------------------
GNum2 sub(in GNum2 a, in float b)
{
    return GNum2(a.val - b, a.g);
}
//--------------------------------
GNum2 sub(in float a, in GNum2 b)
{
    return GNum2(a - b.val, -b.g);
}
//--------------------------------
GNum2 mult(in GNum2 a, in GNum2 b)
{
    return GNum2(a.val * b.val, 
        a.val*b.g + b.val*a.g
        );
}
//--------------------------------
GNum2 mult(in GNum2 a, in float b)
{
    return GNum2(a.val * b, b*a.g);
}
//--------------------------------
GNum2 mult(in float a, in GNum2 b)
{
    return GNum2(a * b.val, a*b.g);
}
//--------------------------------
GNum2 neg(in GNum2 a)
{
    return mult(-1.0,a);
}
//--------------------------------
GNum2 div(in GNum2 a, in GNum2 b)
{
    float b1 = b.val;
    float b2 = b1*b1;

    return GNum2(a.val / b.val , 
        (b.val*a.g - a.val*b.g)/b2
    );
}
//--------------------------------
GNum2 div(in GNum2 a, in float b)
{
    return GNum2(a.val / b, a.g/b);
}
//--------------------------------
GNum2 div(in float a, in GNum2 b)
{
    float b1 = b.val;
    float b2 = b1*b1;

    return GNum2(a / b.val, 
        -a*b.g/b2
    );
}
//--------------------------------
GNum2 inv(in GNum2 a)
{
    return div(1.0, a);
}
//--------------------------------
GNum2 a_pow(in GNum2 a, in GNum2 b)
{
    return a_exp(mult(b,a_log(a)));
}
//--------------------------------
GNum2 a_pow(in GNum2 a, in float b)
{
    // constant exponent -> make special case
    float v = pow(a.val, b); // value f(a(x))
    float da = b*pow(a.val,b-1.0); // first derivative f'(a(x))
    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_pow(in float a, in GNum2 b)
{
    return a_exp(mult(b,log(a)));
}
//--------------------------------
GNum2 a_min(in GNum2 a, in GNum2 b)
{
    if(a.val < b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
GNum2 a_max(in GNum2 a, in GNum2 b)
{
    if(a.val > b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
GNum2 a_exp2(in GNum2 a)
{
    float v = exp2(a.val); // value f(a(x))
    float da = log(2.0)*exp2(a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_inversesqrt(in GNum2 a)
{
    float v = inversesqrt(a.val); // value f(a(x))
    float da = -0.5/pow(sqrt(a.val),3.0); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_atan(in GNum2 a)
{
    float v = atan(a.val); // value f(a(x))
    float da = 1.0/(1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_sqrt(in GNum2 a)
{
    float v = sqrt(a.val); // value f(a(x))
    float da = 0.5/sqrt(a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_sinh(in GNum2 a)
{
    float v = sinh(a.val); // value f(a(x))
    float da = cosh(a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_ceil(in GNum2 a)
{
    float v = ceil(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_tan(in GNum2 a)
{
    float v = tan(a.val); // value f(a(x))
    float da = 1.0 + pow(tan(a.val),2.0); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_asinh(in GNum2 a)
{
    float v = asinh(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_asin(in GNum2 a)
{
    float v = asin(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_acosh(in GNum2 a)
{
    float v = acosh(a.val); // value f(a(x))
    float da = 1.0/sqrt(-1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_abs(in GNum2 a)
{
    float v = abs(a.val); // value f(a(x))
    float da = a.val < 0.0 ? -1.0 : 1.0; // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_exp(in GNum2 a)
{
    float v = exp(a.val); // value f(a(x))
    float da = exp(a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_cosh(in GNum2 a)
{
    float v = cosh(a.val); // value f(a(x))
    float da = sinh(a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_floor(in GNum2 a)
{
    float v = floor(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_log(in GNum2 a)
{
    float v = log(a.val); // value f(a(x))
    float da = 1.0/a.val; // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_atanh(in GNum2 a)
{
    float v = atanh(a.val); // value f(a(x))
    float da = 1.0/(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_log2(in GNum2 a)
{
    float v = log2(a.val); // value f(a(x))
    float da = 1.0/(a.val * log(2.0)); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_acos(in GNum2 a)
{
    float v = acos(a.val); // value f(a(x))
    float da = -1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_tanh(in GNum2 a)
{
    float v = tanh(a.val); // value f(a(x))
    float da = 1.0 - pow(tanh(a.val),2.0); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_cos(in GNum2 a)
{
    float v = cos(a.val); // value f(a(x))
    float da = -sin(a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}
//--------------------------------
GNum2 a_sin(in GNum2 a)
{
    float v = sin(a.val); // value f(a(x))
    float da = cos(a.val); // first derivative f'(a(x))

    return GNum2(v , da * a.g);
}

#endif // GRADNUM_2_H_