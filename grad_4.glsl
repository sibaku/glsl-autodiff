#ifndef GRADNUM_4_H_
#define GRADNUM_4_H_

// This file contains methods to compute the gradient of a scalar valued 4 dimensional
// function using automatic forward differentiation

//--------------------------------
// Types
//--------------------------------

// Data type to hold information about a scalar valued 4 dimensional function
// These should be created by the constG4 (for constants) and varG4 (for variables) helpers
struct GNum4
{
    // The current value
    float val;
    // The current gradient
    vec4 g;
};

//--------------------------------
// Prototypes
//--------------------------------

/**
* Creates a constant GNum4
* @param val The current value of the constant
*/
GNum4 constG4(in float val);
GNum4 varG4(in float val, in int index);
GNum4 varG4x(in float val);
GNum4 varG4y(in float val);
GNum4 varG4z(in float val);
GNum4 varG4w(in float val);
GNum4 add(in GNum4 a, in GNum4 b);
GNum4 add(in GNum4 a, in float b);
GNum4 add(in float a, in GNum4 b);
GNum4 sub(in GNum4 a, in GNum4 b);
GNum4 sub(in GNum4 a, in float b);
GNum4 sub(in float a, in GNum4 b);
GNum4 mult(in GNum4 a, in GNum4 b);
GNum4 mult(in GNum4 a, in float b);
GNum4 mult(in float a, in GNum4 b);
GNum4 neg(in GNum4 a);
GNum4 div(in GNum4 a, in GNum4 b);
GNum4 div(in GNum4 a, in float b);
GNum4 div(in float a, in GNum4 b);
GNum4 inv(in GNum4 a);
GNum4 a_pow(in GNum4 a, in GNum4 b);
GNum4 a_pow(in GNum4 a, in float b);
GNum4 a_pow(in float a, in GNum4 b);
GNum4 a_min(in GNum4 a, in GNum4 b);
GNum4 a_max(in GNum4 a, in GNum4 b);
GNum4 a_exp2(in GNum4 a);
GNum4 a_inversesqrt(in GNum4 a);
GNum4 a_atan(in GNum4 a);
GNum4 a_sqrt(in GNum4 a);
GNum4 a_sinh(in GNum4 a);
GNum4 a_ceil(in GNum4 a);
GNum4 a_tan(in GNum4 a);
GNum4 a_asinh(in GNum4 a);
GNum4 a_asin(in GNum4 a);
GNum4 a_acosh(in GNum4 a);
GNum4 a_abs(in GNum4 a);
GNum4 a_exp(in GNum4 a);
GNum4 a_cosh(in GNum4 a);
GNum4 a_floor(in GNum4 a);
GNum4 a_log(in GNum4 a);
GNum4 a_atanh(in GNum4 a);
GNum4 a_log2(in GNum4 a);
GNum4 a_acos(in GNum4 a);
GNum4 a_tanh(in GNum4 a);
GNum4 a_cos(in GNum4 a);
GNum4 a_sin(in GNum4 a);

//--------------------------------
// Macros
//--------------------------------

#define GRAD4(f,x, y, z, w,result)  {     result = f(varG4x(x), varG4y(y), varG4z(z), varG4w(w)); }
//--------------------------------
#define JACOBI24(f1, f2, x, y, z, w, result)  {     GNum4 gradResult;  	GRAD4(f1, x, y, z, w, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2]; 	result[3][0] = gradResult.g[3];  	GRAD4(f2, x, y, z, w, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2]; 	result[3][1] = gradResult.g[3];  }
//--------------------------------
#define JACOBI24_VALUE(f1, f2, x, y, z, w, result, value)  {     GNum4 gradResult;  	GRAD4(f1, x, y, z, w, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2]; 	result[3][0] = gradResult.g[3];  	GRAD4(f2, x, y, z, w, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2]; 	result[3][1] = gradResult.g[3];  }
//--------------------------------
#define JACOBI34(f1, f2, f3, x, y, z, w, result)  {     GNum4 gradResult;  	GRAD4(f1, x, y, z, w, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2]; 	result[3][0] = gradResult.g[3];  	GRAD4(f2, x, y, z, w, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2]; 	result[3][1] = gradResult.g[3];  	GRAD4(f3, x, y, z, w, gradResult);     result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1]; 	result[2][2] = gradResult.g[2]; 	result[3][2] = gradResult.g[3];  }
//--------------------------------
#define JACOBI34_VALUE(f1, f2, f3, x, y, z, w, result, value)  {     GNum4 gradResult;  	GRAD4(f1, x, y, z, w, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2]; 	result[3][0] = gradResult.g[3];  	GRAD4(f2, x, y, z, w, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2]; 	result[3][1] = gradResult.g[3];  	GRAD4(f3, x, y, z, w, gradResult); 	value[2] = gradResult.val; 	result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1]; 	result[2][2] = gradResult.g[2]; 	result[3][2] = gradResult.g[3];  }
//--------------------------------
#define JACOBI4(f1, f2, f3, f4, x, y, z, w, result)  {     GNum4 gradResult;  	GRAD4(f1, x, y, z, w, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2]; 	result[3][0] = gradResult.g[3];  	GRAD4(f2, x, y, z, w, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2]; 	result[3][1] = gradResult.g[3];  	GRAD4(f3, x, y, z, w, gradResult);     result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1]; 	result[2][2] = gradResult.g[2]; 	result[3][2] = gradResult.g[3];  	GRAD4(f4, x, y, z, w, gradResult);     result[0][3] = gradResult.g[0]; 	result[1][3] = gradResult.g[1]; 	result[2][3] = gradResult.g[2]; 	result[3][3] = gradResult.g[3];  }
//--------------------------------
#define JACOBI4_VALUE(f1, f2, f3, f4, x, y, z, w, result, value)  {     GNum4 gradResult;  	GRAD4(f1, x, y, z, w, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2]; 	result[3][0] = gradResult.g[3];  	GRAD4(f2, x, y, z, w, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2]; 	result[3][1] = gradResult.g[3];  	GRAD4(f3, x, y, z, w, gradResult); 	value[2] = gradResult.val; 	result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1]; 	result[2][2] = gradResult.g[2]; 	result[3][2] = gradResult.g[3];  	GRAD4(f4, x, y, z, w, gradResult); 	value[3] = gradResult.val; 	result[0][3] = gradResult.g[0]; 	result[1][3] = gradResult.g[1]; 	result[2][3] = gradResult.g[2]; 	result[3][3] = gradResult.g[3];  }

//--------------------------------
// Implementation
//--------------------------------

GNum4 constG4(in float val)
{
    return GNum4(val, vec4(0.0));
}
//--------------------------------
GNum4 varG4(in float val, in int index)
{   
    vec4 g = vec4(0.0);
    g[index] = 1.0;
    return GNum4(val, g);
}
//--------------------------------
GNum4 varG4x(in float val)
{   
    vec4 g = vec4(0.0);
    g[0] = 1.0;
    return GNum4(val, g);
}
//--------------------------------
GNum4 varG4y(in float val)
{   
    vec4 g = vec4(0.0);
    g[1] = 1.0;
    return GNum4(val, g);
}
//--------------------------------
GNum4 varG4z(in float val)
{   
    vec4 g = vec4(0.0);
    g[2] = 1.0;
    return GNum4(val, g);
}
//--------------------------------
GNum4 varG4w(in float val)
{   
    vec4 g = vec4(0.0);
    g[3] = 1.0;
    return GNum4(val, g);
}
//--------------------------------
GNum4 add(in GNum4 a, in GNum4 b)
{
    return GNum4(a.val + b.val, a.g + b.g);
}
//--------------------------------
GNum4 add(in GNum4 a, in float b)
{
    return GNum4(a.val + b, a.g);
}
//--------------------------------
GNum4 add(in float a, in GNum4 b)
{
    return GNum4(a + b.val, b.g);
}
//--------------------------------
GNum4 sub(in GNum4 a, in GNum4 b)
{
    return GNum4(a.val - b.val, a.g - b.g);
}
//--------------------------------
GNum4 sub(in GNum4 a, in float b)
{
    return GNum4(a.val - b, a.g);
}
//--------------------------------
GNum4 sub(in float a, in GNum4 b)
{
    return GNum4(a - b.val, -b.g);
}
//--------------------------------
GNum4 mult(in GNum4 a, in GNum4 b)
{
    return GNum4(a.val * b.val, 
        a.val*b.g + b.val*a.g
        );
}
//--------------------------------
GNum4 mult(in GNum4 a, in float b)
{
    return GNum4(a.val * b, b*a.g);
}
//--------------------------------
GNum4 mult(in float a, in GNum4 b)
{
    return GNum4(a * b.val, a*b.g);
}
//--------------------------------
GNum4 neg(in GNum4 a)
{
    return mult(-1.0,a);
}
//--------------------------------
GNum4 div(in GNum4 a, in GNum4 b)
{
    float b1 = b.val;
    float b2 = b1*b1;

    return GNum4(a.val / b.val , 
        (b.val*a.g - a.val*b.g)/b2
    );
}
//--------------------------------
GNum4 div(in GNum4 a, in float b)
{
    return GNum4(a.val / b, a.g/b);
}
//--------------------------------
GNum4 div(in float a, in GNum4 b)
{
    float b1 = b.val;
    float b2 = b1*b1;

    return GNum4(a / b.val, 
        -a*b.g/b2
    );
}
//--------------------------------
GNum4 inv(in GNum4 a)
{
    return div(1.0, a);
}
//--------------------------------
GNum4 a_pow(in GNum4 a, in GNum4 b)
{
    return a_exp(mult(b,a_log(a)));
}
//--------------------------------
GNum4 a_pow(in GNum4 a, in float b)
{
    // constant exponent -> make special case
    float v = pow(a.val, b); // value f(a(x))
    float da = b*pow(a.val,b-1.0); // first derivative f'(a(x))
    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_pow(in float a, in GNum4 b)
{
    return a_exp(mult(b,log(a)));
}
//--------------------------------
GNum4 a_min(in GNum4 a, in GNum4 b)
{
    if(a.val < b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
GNum4 a_max(in GNum4 a, in GNum4 b)
{
    if(a.val > b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
GNum4 a_exp2(in GNum4 a)
{
    float v = exp2(a.val); // value f(a(x))
    float da = log(2.0)*exp2(a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_inversesqrt(in GNum4 a)
{
    float v = inversesqrt(a.val); // value f(a(x))
    float da = -0.5/pow(sqrt(a.val),3.0); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_atan(in GNum4 a)
{
    float v = atan(a.val); // value f(a(x))
    float da = 1.0/(1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_sqrt(in GNum4 a)
{
    float v = sqrt(a.val); // value f(a(x))
    float da = 0.5/sqrt(a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_sinh(in GNum4 a)
{
    float v = sinh(a.val); // value f(a(x))
    float da = cosh(a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_ceil(in GNum4 a)
{
    float v = ceil(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_tan(in GNum4 a)
{
    float v = tan(a.val); // value f(a(x))
    float da = 1.0 + pow(tan(a.val),2.0); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_asinh(in GNum4 a)
{
    float v = asinh(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_asin(in GNum4 a)
{
    float v = asin(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_acosh(in GNum4 a)
{
    float v = acosh(a.val); // value f(a(x))
    float da = 1.0/sqrt(-1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_abs(in GNum4 a)
{
    float v = abs(a.val); // value f(a(x))
    float da = a.val < 0.0 ? -1.0 : 1.0; // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_exp(in GNum4 a)
{
    float v = exp(a.val); // value f(a(x))
    float da = exp(a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_cosh(in GNum4 a)
{
    float v = cosh(a.val); // value f(a(x))
    float da = sinh(a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_floor(in GNum4 a)
{
    float v = floor(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_log(in GNum4 a)
{
    float v = log(a.val); // value f(a(x))
    float da = 1.0/a.val; // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_atanh(in GNum4 a)
{
    float v = atanh(a.val); // value f(a(x))
    float da = 1.0/(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_log2(in GNum4 a)
{
    float v = log2(a.val); // value f(a(x))
    float da = 1.0/(a.val * log(2.0)); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_acos(in GNum4 a)
{
    float v = acos(a.val); // value f(a(x))
    float da = -1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_tanh(in GNum4 a)
{
    float v = tanh(a.val); // value f(a(x))
    float da = 1.0 - pow(tanh(a.val),2.0); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_cos(in GNum4 a)
{
    float v = cos(a.val); // value f(a(x))
    float da = -sin(a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}
//--------------------------------
GNum4 a_sin(in GNum4 a)
{
    float v = sin(a.val); // value f(a(x))
    float da = cos(a.val); // first derivative f'(a(x))

    return GNum4(v , da * a.g);
}

#endif // GRADNUM_4_H_