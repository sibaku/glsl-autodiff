#ifndef DERIVATIVES_1_H_
#define DERIVATIVES_1_H_

// This file contains methods to compute the first derivative
// of a scalar function using automatic forward differentiation

// The data is stored in a vec2
// v[0] contains the value of the function
// v[1] contains the first derivative


//--------------------------------
// Prototypes
//--------------------------------

/**
* Creates a constant derivative number
* @param val The current value of the constant
*/
vec2 constD1(in float val);
/**
* Creates a derivative number stored in a vec2
* @param val The current value of the variable
*/
vec2 varD1(in float val);
vec2 add(in vec2 a, in vec2 b);
vec2 add(in vec2 a, in float b);
vec2 add(in float a, in vec2 b);
vec2 sub(in vec2 a, in vec2 b);
vec2 sub(in vec2 a, in float b);
vec2 sub(in float a, in vec2 b);
vec2 mult(in vec2 a, in vec2 b);
vec2 mult(in vec2 a, in float b);
vec2 mult(in float a, in vec2 b);
vec2 neg(in vec2 a);
vec2 div(in vec2 a, in vec2 b);
vec2 div(in vec2 a, in float b);
vec2 div(in float a, in vec2 b);
vec2 inv(in vec2 a);
vec2 a_pow(in vec2 a, in vec2 b);
vec2 a_pow(in vec2 a, in float b);
vec2 a_pow(in float a, in vec2 b);
vec2 a_min(in vec2 a, in vec2 b);
vec2 a_max(in vec2 a, in vec2 b);
vec2 a_exp2(in vec2 a);
vec2 a_inversesqrt(in vec2 a);
vec2 a_atan(in vec2 a);
vec2 a_sqrt(in vec2 a);
vec2 a_sinh(in vec2 a);
vec2 a_ceil(in vec2 a);
vec2 a_tan(in vec2 a);
vec2 a_asinh(in vec2 a);
vec2 a_asin(in vec2 a);
vec2 a_acosh(in vec2 a);
vec2 a_abs(in vec2 a);
vec2 a_exp(in vec2 a);
vec2 a_cosh(in vec2 a);
vec2 a_floor(in vec2 a);
vec2 a_log(in vec2 a);
vec2 a_atanh(in vec2 a);
vec2 a_log2(in vec2 a);
vec2 a_acos(in vec2 a);
vec2 a_tanh(in vec2 a);
vec2 a_cos(in vec2 a);
vec2 a_sin(in vec2 a);
vec2 a_atan2(in vec2 y, in vec2 x);
vec2 a_atan2(in vec2 y, in float x);
vec2 a_atan2(in float y, in vec2 x);
vec2 a_mix(in vec2 a, in vec2 b, in vec2 t);
vec2 a_mix(in vec2 a, in vec2 b, in float t);
vec2 a_mix(in vec2 a, in float b, in vec2 t);
vec2 a_mix(in vec2 a, in float b, in float t);
vec2 a_mix(in float a, in vec2 b, in vec2 t);
vec2 a_mix(in float a, in vec2 b, in float t);
vec2 a_mix(in float a, in float b, in vec2 t);

//--------------------------------
// Implementation
//--------------------------------

vec2 constD1(in float val)
{
    vec2 c = vec2(0.0);
    c[0] = val;
    return c;
}
//--------------------------------
vec2 varD1(in float val)
{   
    return vec2(val, 1.0);
}
//--------------------------------
vec2 add(in vec2 a, in vec2 b)
{
    return a + b;
}
//--------------------------------
vec2 add(in vec2 a, in float b)
{
    a[0] += b;
    return a;
}
//--------------------------------
vec2 add(in float a, in vec2 b)
{
    b[0] += a;
    return b;
}
//--------------------------------
vec2 sub(in vec2 a, in vec2 b)
{
    return a - b;
}
//--------------------------------
vec2 sub(in vec2 a, in float b)
{
    a[0] -= b;
    return a;
}
//--------------------------------
vec2 sub(in float a, in vec2 b)
{
    b *= -1.0;
    b[0] += a;
    return b;
}
//--------------------------------
vec2 mult(in vec2 a, in vec2 b)
{
    float v = a[0] * b[0];
	float da = a[0]*b[1] + b[0]*a[1];

    return vec2(v,da);
}
//--------------------------------
vec2 mult(in vec2 a, in float b)
{
    return a*b;
}
//--------------------------------
vec2 mult(in float a, in vec2 b)
{
    return a*b;
}
//--------------------------------
vec2 neg(in vec2 a)
{
    return mult(-1.0,a);
}
//--------------------------------
vec2 div(in vec2 a, in vec2 b)
{
    float v = a[0] / b[0];
	float b2 = b[0] * b[0];
float da = a[1]/b[0] - a[0]*b[1]/b2;

    return vec2(v,da);
}
//--------------------------------
vec2 div(in vec2 a, in float b)
{
    return a / b;
}
//--------------------------------
vec2 div(in float a, in vec2 b)
{
    float v = a / b[0];
	float b2 = b[0] * b[0];
float da = - a*b[1]/b2;

    return vec2(v, da);
}
//--------------------------------
vec2 inv(in vec2 a)
{
    return div(1.0, a);
}
//--------------------------------
vec2 a_pow(in vec2 a, in vec2 b)
{
    return a_exp(mult(b,a_log(a)));
}
//--------------------------------
vec2 a_pow(in vec2 a, in float b)
{
    float v = pow(a[0], b); // value f(a(x))
	float da = b*pow(a[0],b-1.0); // first derivative f'(a(x))

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_pow(in float a, in vec2 b)
{
    return a_exp(mult(b,log(a)));
}
//--------------------------------
vec2 a_min(in vec2 a, in vec2 b)
{
    if(a[0] < b[0])
    {
        return a;
    }
    return b;
}
//--------------------------------
vec2 a_max(in vec2 a, in vec2 b)
{
    if(a[0] > b[0])
    {
        return a;
    }
    return b;
}
//--------------------------------
vec2 a_exp2(in vec2 a)
{
    float v = exp2(a[0]);
	float da = log(2.0)*exp2(a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_inversesqrt(in vec2 a)
{
    float v = inversesqrt(a[0]);
	float da = -0.5/pow(sqrt(a[0]),3.0);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_atan(in vec2 a)
{
    float v = atan(a[0]);
	float da = 1.0/(1.0 + a[0] * a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_sqrt(in vec2 a)
{
    float v = sqrt(a[0]);
	float da = 0.5/sqrt(a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_sinh(in vec2 a)
{
    float v = sinh(a[0]);
	float da = cosh(a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_ceil(in vec2 a)
{
    float v = ceil(a[0]);
	float da = 0.0;

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_tan(in vec2 a)
{
    float v = tan(a[0]);
	float da = 1.0 + pow(tan(a[0]),2.0);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_asinh(in vec2 a)
{
    float v = asinh(a[0]);
	float da = 1.0/sqrt(1.0 + a[0] * a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_asin(in vec2 a)
{
    float v = asin(a[0]);
	float da = 1.0/sqrt(1.0 - a[0] * a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_acosh(in vec2 a)
{
    float v = acosh(a[0]);
	float da = 1.0/sqrt(-1.0 + a[0] * a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_abs(in vec2 a)
{
    float v = abs(a[0]);
	float da = a[0] < 0.0 ? -1.0 : 1.0;

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_exp(in vec2 a)
{
    float v = exp(a[0]);
	float da = exp(a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_cosh(in vec2 a)
{
    float v = cosh(a[0]);
	float da = sinh(a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_floor(in vec2 a)
{
    float v = floor(a[0]);
	float da = 0.0;

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_log(in vec2 a)
{
    float v = log(a[0]);
	float da = 1.0/a[0];

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_atanh(in vec2 a)
{
    float v = atanh(a[0]);
	float da = 1.0/(1.0 - a[0] * a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_log2(in vec2 a)
{
    float v = log2(a[0]);
	float da = 1.0/(a[0] * log(2.0));

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_acos(in vec2 a)
{
    float v = acos(a[0]);
	float da = -1.0/sqrt(1.0 - a[0] * a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_tanh(in vec2 a)
{
    float v = tanh(a[0]);
	float da = 1.0 - pow(tanh(a[0]),2.0);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_cos(in vec2 a)
{
    float v = cos(a[0]);
	float da = -sin(a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_sin(in vec2 a)
{
    float v = sin(a[0]);
	float da = cos(a[0]);

    return vec2(v, da * a[1]);
}
//--------------------------------
vec2 a_atan2(in vec2 y, in vec2 x)
{
    const float pi = 3.14159265; 
    // from https://en.wikipedia.org/wiki/Atan2
    if(x[0] > 0.0)
    {
        vec2 n = a_sqrt(add(mult(x,x),mult(y,y)));
        vec2 inner = div(y, add(n,x));
        
        return mult(2.0,a_atan(inner));
        
    }else if(x[0] <= 0.0 && abs(y[0]) > 1E-6)
    {
        vec2 n = a_sqrt(add(mult(x,x),mult(y,y)));
        vec2 inner = div(sub(n,x),y);
         return mult(2.0,a_atan(inner));
    }else if(x[0] < 0.0 && abs(y[0]) <= 1E-6)
    {
        return constD1(pi);
    }
    // return 0 for undefined
    return constD1(0.0); 
}
//--------------------------------
vec2 a_atan2(in vec2 y, in float x)
{
    return a_atan2(y,constD1(x));
}
//--------------------------------
vec2 a_atan2(in float y, in vec2 x)
{
    return a_atan2(constD1(y),x);
}
//--------------------------------
vec2 a_mix(in vec2 a, in vec2 b, in vec2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
vec2 a_mix(in vec2 a, in vec2 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

//--------------------------------
vec2 a_mix(in vec2 a, in float b, in vec2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
vec2 a_mix(in vec2 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

//--------------------------------
vec2 a_mix(in float a, in vec2 b, in vec2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
vec2 a_mix(in float a, in vec2 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

//--------------------------------
vec2 a_mix(in float a, in float b, in vec2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}


#endif // DERIVATIVES_1_H_