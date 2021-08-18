#ifndef GRADNUM_3_H_
#define GRADNUM_3_H_

// This file contains methods to compute the gradient of a scalar valued 3 dimensional
// function using automatic forward differentiation

//--------------------------------
// Types
//--------------------------------

// Data type to hold information about a scalar valued 3 dimensional function
// These should be created by the constG3 (for constants) and varG3 (for variables) helpers
struct GNum3
{
    // The current value
    float val;
    // The current gradient
    vec3 g;
};

//--------------------------------
// Prototypes
//--------------------------------

/**
* Creates a constant GNum3
* @param val The current value of the constant
*/
GNum3 constG3(in float val);
GNum3 varG3(in float val, in int index);
GNum3 varG3x(in float val);
GNum3 varG3y(in float val);
GNum3 varG3z(in float val);
GNum3 add(in GNum3 a, in GNum3 b);
GNum3 add(in GNum3 a, in float b);
GNum3 add(in float a, in GNum3 b);
GNum3 sub(in GNum3 a, in GNum3 b);
GNum3 sub(in GNum3 a, in float b);
GNum3 sub(in float a, in GNum3 b);
GNum3 mult(in GNum3 a, in GNum3 b);
GNum3 mult(in GNum3 a, in float b);
GNum3 mult(in float a, in GNum3 b);
GNum3 neg(in GNum3 a);
GNum3 div(in GNum3 a, in GNum3 b);
GNum3 div(in GNum3 a, in float b);
GNum3 div(in float a, in GNum3 b);
GNum3 inv(in GNum3 a);
GNum3 a_pow(in GNum3 a, in GNum3 b);
GNum3 a_pow(in GNum3 a, in float b);
GNum3 a_pow(in float a, in GNum3 b);
GNum3 a_min(in GNum3 a, in GNum3 b);
GNum3 a_max(in GNum3 a, in GNum3 b);
GNum3 a_exp2(in GNum3 a);
GNum3 a_inversesqrt(in GNum3 a);
GNum3 a_atan(in GNum3 a);
GNum3 a_sqrt(in GNum3 a);
GNum3 a_sinh(in GNum3 a);
GNum3 a_ceil(in GNum3 a);
GNum3 a_tan(in GNum3 a);
GNum3 a_asinh(in GNum3 a);
GNum3 a_asin(in GNum3 a);
GNum3 a_acosh(in GNum3 a);
GNum3 a_abs(in GNum3 a);
GNum3 a_exp(in GNum3 a);
GNum3 a_cosh(in GNum3 a);
GNum3 a_floor(in GNum3 a);
GNum3 a_log(in GNum3 a);
GNum3 a_atanh(in GNum3 a);
GNum3 a_log2(in GNum3 a);
GNum3 a_acos(in GNum3 a);
GNum3 a_tanh(in GNum3 a);
GNum3 a_cos(in GNum3 a);
GNum3 a_sin(in GNum3 a);
GNum3 a_atan2(in GNum3 y, in GNum3 x);
GNum3 a_atan2(in GNum3 y, in float x);
GNum3 a_atan2(in float y, in GNum3 x);
GNum3 a_mix(in GNum3 a, in GNum3 b, in GNum3 t);
GNum3 a_mix(in GNum3 a, in GNum3 b, in float t);
GNum3 a_mix(in GNum3 a, in float b, in GNum3 t);
GNum3 a_mix(in GNum3 a, in float b, in float t);
GNum3 a_mix(in float a, in GNum3 b, in GNum3 t);
GNum3 a_mix(in float a, in GNum3 b, in float t);
GNum3 a_mix(in float a, in float b, in GNum3 t);

//--------------------------------
// Macros
//--------------------------------

#define GRAD3(f,x, y, z,result)  {     result = f(varG3x(x), varG3y(y), varG3z(z)); }
//--------------------------------
#define JACOBI23(f1, f2, x, y, z, result)  {     GNum3 gradResult;  	GRAD3(f1, x, y, z, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2];  	GRAD3(f2, x, y, z, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2];  }
//--------------------------------
#define JACOBI23_VALUE(f1, f2, x, y, z, result, value)  {     GNum3 gradResult;  	GRAD3(f1, x, y, z, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2];  	GRAD3(f2, x, y, z, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2];  }
//--------------------------------
#define JACOBI3(f1, f2, f3, x, y, z, result)  {     GNum3 gradResult;  	GRAD3(f1, x, y, z, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2];  	GRAD3(f2, x, y, z, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2];  	GRAD3(f3, x, y, z, gradResult);     result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1]; 	result[2][2] = gradResult.g[2];  }
//--------------------------------
#define JACOBI3_VALUE(f1, f2, f3, x, y, z, result, value)  {     GNum3 gradResult;  	GRAD3(f1, x, y, z, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2];  	GRAD3(f2, x, y, z, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2];  	GRAD3(f3, x, y, z, gradResult); 	value[2] = gradResult.val; 	result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1]; 	result[2][2] = gradResult.g[2];  }
//--------------------------------
#define JACOBI43(f1, f2, f3, f4, x, y, z, result)  {     GNum3 gradResult;  	GRAD3(f1, x, y, z, gradResult);     result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2];  	GRAD3(f2, x, y, z, gradResult);     result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2];  	GRAD3(f3, x, y, z, gradResult);     result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1]; 	result[2][2] = gradResult.g[2];  	GRAD3(f4, x, y, z, gradResult);     result[0][3] = gradResult.g[0]; 	result[1][3] = gradResult.g[1]; 	result[2][3] = gradResult.g[2];  }
//--------------------------------
#define JACOBI43_VALUE(f1, f2, f3, f4, x, y, z, result, value)  {     GNum3 gradResult;  	GRAD3(f1, x, y, z, gradResult); 	value[0] = gradResult.val; 	result[0][0] = gradResult.g[0]; 	result[1][0] = gradResult.g[1]; 	result[2][0] = gradResult.g[2];  	GRAD3(f2, x, y, z, gradResult); 	value[1] = gradResult.val; 	result[0][1] = gradResult.g[0]; 	result[1][1] = gradResult.g[1]; 	result[2][1] = gradResult.g[2];  	GRAD3(f3, x, y, z, gradResult); 	value[2] = gradResult.val; 	result[0][2] = gradResult.g[0]; 	result[1][2] = gradResult.g[1]; 	result[2][2] = gradResult.g[2];  	GRAD3(f4, x, y, z, gradResult); 	value[3] = gradResult.val; 	result[0][3] = gradResult.g[0]; 	result[1][3] = gradResult.g[1]; 	result[2][3] = gradResult.g[2];  }

//--------------------------------
// Implementation
//--------------------------------

GNum3 constG3(in float val)
{
    return GNum3(val, vec3(0.0));
}
//--------------------------------
GNum3 varG3(in float val, in int index)
{   
    vec3 g = vec3(0.0);
    g[index] = 1.0;
    return GNum3(val, g);
}
//--------------------------------
GNum3 varG3x(in float val)
{   
    vec3 g = vec3(0.0);
    g[0] = 1.0;
    return GNum3(val, g);
}
//--------------------------------
GNum3 varG3y(in float val)
{   
    vec3 g = vec3(0.0);
    g[1] = 1.0;
    return GNum3(val, g);
}
//--------------------------------
GNum3 varG3z(in float val)
{   
    vec3 g = vec3(0.0);
    g[2] = 1.0;
    return GNum3(val, g);
}
//--------------------------------
GNum3 add(in GNum3 a, in GNum3 b)
{
    return GNum3(a.val + b.val, a.g + b.g);
}
//--------------------------------
GNum3 add(in GNum3 a, in float b)
{
    return GNum3(a.val + b, a.g);
}
//--------------------------------
GNum3 add(in float a, in GNum3 b)
{
    return GNum3(a + b.val, b.g);
}
//--------------------------------
GNum3 sub(in GNum3 a, in GNum3 b)
{
    return GNum3(a.val - b.val, a.g - b.g);
}
//--------------------------------
GNum3 sub(in GNum3 a, in float b)
{
    return GNum3(a.val - b, a.g);
}
//--------------------------------
GNum3 sub(in float a, in GNum3 b)
{
    return GNum3(a - b.val, -b.g);
}
//--------------------------------
GNum3 mult(in GNum3 a, in GNum3 b)
{
    return GNum3(a.val * b.val, 
        a.val*b.g + b.val*a.g
        );
}
//--------------------------------
GNum3 mult(in GNum3 a, in float b)
{
    return GNum3(a.val * b, b*a.g);
}
//--------------------------------
GNum3 mult(in float a, in GNum3 b)
{
    return GNum3(a * b.val, a*b.g);
}
//--------------------------------
GNum3 neg(in GNum3 a)
{
    return mult(-1.0,a);
}
//--------------------------------
GNum3 div(in GNum3 a, in GNum3 b)
{
    float b1 = b.val;
    float b2 = b1*b1;

    return GNum3(a.val / b.val , 
        (b.val*a.g - a.val*b.g)/b2
    );
}
//--------------------------------
GNum3 div(in GNum3 a, in float b)
{
    return GNum3(a.val / b, a.g/b);
}
//--------------------------------
GNum3 div(in float a, in GNum3 b)
{
    float b1 = b.val;
    float b2 = b1*b1;

    return GNum3(a / b.val, 
        -a*b.g/b2
    );
}
//--------------------------------
GNum3 inv(in GNum3 a)
{
    return div(1.0, a);
}
//--------------------------------
GNum3 a_pow(in GNum3 a, in GNum3 b)
{
    return a_exp(mult(b,a_log(a)));
}
//--------------------------------
GNum3 a_pow(in GNum3 a, in float b)
{
    // constant exponent -> make special case
    float v = pow(a.val, b); // value f(a(x))
    float da = b*pow(a.val,b-1.0); // first derivative f'(a(x))
    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_pow(in float a, in GNum3 b)
{
    return a_exp(mult(b,log(a)));
}
//--------------------------------
GNum3 a_min(in GNum3 a, in GNum3 b)
{
    if(a.val < b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
GNum3 a_max(in GNum3 a, in GNum3 b)
{
    if(a.val > b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
GNum3 a_exp2(in GNum3 a)
{
    float v = exp2(a.val); // value f(a(x))
    float da = log(2.0)*exp2(a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_inversesqrt(in GNum3 a)
{
    float v = inversesqrt(a.val); // value f(a(x))
    float da = -0.5/pow(sqrt(a.val),3.0); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_atan(in GNum3 a)
{
    float v = atan(a.val); // value f(a(x))
    float da = 1.0/(1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_sqrt(in GNum3 a)
{
    float v = sqrt(a.val); // value f(a(x))
    float da = 0.5/sqrt(a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_sinh(in GNum3 a)
{
    float v = sinh(a.val); // value f(a(x))
    float da = cosh(a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_ceil(in GNum3 a)
{
    float v = ceil(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_tan(in GNum3 a)
{
    float v = tan(a.val); // value f(a(x))
    float da = 1.0 + pow(tan(a.val),2.0); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_asinh(in GNum3 a)
{
    float v = asinh(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_asin(in GNum3 a)
{
    float v = asin(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_acosh(in GNum3 a)
{
    float v = acosh(a.val); // value f(a(x))
    float da = 1.0/sqrt(-1.0 + a.val * a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_abs(in GNum3 a)
{
    float v = abs(a.val); // value f(a(x))
    float da = a.val < 0.0 ? -1.0 : 1.0; // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_exp(in GNum3 a)
{
    float v = exp(a.val); // value f(a(x))
    float da = exp(a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_cosh(in GNum3 a)
{
    float v = cosh(a.val); // value f(a(x))
    float da = sinh(a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_floor(in GNum3 a)
{
    float v = floor(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_log(in GNum3 a)
{
    float v = log(a.val); // value f(a(x))
    float da = 1.0/a.val; // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_atanh(in GNum3 a)
{
    float v = atanh(a.val); // value f(a(x))
    float da = 1.0/(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_log2(in GNum3 a)
{
    float v = log2(a.val); // value f(a(x))
    float da = 1.0/(a.val * log(2.0)); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_acos(in GNum3 a)
{
    float v = acos(a.val); // value f(a(x))
    float da = -1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_tanh(in GNum3 a)
{
    float v = tanh(a.val); // value f(a(x))
    float da = 1.0 - pow(tanh(a.val),2.0); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_cos(in GNum3 a)
{
    float v = cos(a.val); // value f(a(x))
    float da = -sin(a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_sin(in GNum3 a)
{
    float v = sin(a.val); // value f(a(x))
    float da = cos(a.val); // first derivative f'(a(x))

    return GNum3(v , da * a.g);
}
//--------------------------------
GNum3 a_atan2(in GNum3 y, in GNum3 x)
{
    const float pi = 3.14159265; 
    // from https://en.wikipedia.org/wiki/Atan2
    if(x.val > 0.0)
    {
        GNum3 n = a_sqrt(add(mult(x,x),mult(y,y)));
        GNum3 inner = div(y, add(n,x));
        
        return mult(2.0,a_atan(inner));
        
    }else if(x.val <= 0.0 && abs(y.val) > 1E-6)
    {
        GNum3 n = a_sqrt(add(mult(x,x),mult(y,y)));
        GNum3 inner = div(sub(n,x),y);
         return mult(2.0,a_atan(inner));
    }else if(x.val < 0.0 && abs(y.val) <= 1E-6)
    {
        return constG3(pi);
    }
    // return 0 for undefined
    return constG3(0.0); 
}
//--------------------------------
GNum3 a_atan2(in GNum3 y, in float x)
{
    return a_atan2(y,constG3(x));
}
//--------------------------------
GNum3 a_atan2(in float y, in GNum3 x)
{
    return a_atan2(constG3(y),x);
}
//--------------------------------
GNum3 a_mix(in GNum3 a, in GNum3 b, in GNum3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum3 a_mix(in GNum3 a, in GNum3 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

//--------------------------------
GNum3 a_mix(in GNum3 a, in float b, in GNum3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum3 a_mix(in GNum3 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

//--------------------------------
GNum3 a_mix(in float a, in GNum3 b, in GNum3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum3 a_mix(in float a, in GNum3 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

//--------------------------------
GNum3 a_mix(in float a, in float b, in GNum3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}


#endif // GRADNUM_3_H_