#ifndef AUTODIFF_H_
#define AUTODIFF_H_

// This file contains methods to compute the first and second order derivatives, gradient and hessian
// of a scalar valued multivariate (2-4 variables) and univariate function using automatic forward differentiation

//--------------------------------
// Types
//--------------------------------

//--------------------------------
// Hessian
//--------------------------------

// Data type to hold information about a scalar valued 2 dimensional function
// These should be created by the constH2 (for constants) and varH2 (for variables) helpers
struct HNum2
{
    // The current value
    float val;
    // The current gradient
    vec2 g;
    // The current hessian
    mat2 h;
};
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
// Data type to hold information about a scalar valued 4 dimensional function
// These should be created by the constH4 (for constants) and varH4 (for variables) helpers
struct HNum4
{
    // The current value
    float val;
    // The current gradient
    vec4 g;
    // The current hessian
    mat4 h;
};

//--------------------------------
// Gradient
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

//--------------------------------
// Hessian
//--------------------------------

/**
* Creates a constant HNum2
* @param val The current value of the constant
*/
HNum2 constH2(in float val);
/**
* Creates a HNum2 corresponding to the variable with the given index
* @param val The current value of the variable
* @param index The variable's index
*/
HNum2 varH2(in float val, in int index);
/**
* Creates a HNum2 corresponding to the variable x (index = 0)
* @param val The current value of the variable
*/
HNum2 varH2x(in float val);
/**
* Creates a HNum2 corresponding to the variable y (index = 1)
* @param val The current value of the variable
*/
HNum2 varH2y(in float val);
HNum2 add(in HNum2 a, in HNum2 b);
HNum2 add(in HNum2 a, in float b);
HNum2 add(in float a, in HNum2 b);
HNum2 sub(in HNum2 a, in HNum2 b);
HNum2 sub(in HNum2 a, in float b);
HNum2 sub(in float a, in HNum2 b);
HNum2 mult(in HNum2 a, in HNum2 b);
HNum2 mult(in HNum2 a, in float b);
HNum2 mult(in float a, in HNum2 b);
HNum2 neg(in HNum2 a);
HNum2 div(in HNum2 a, in HNum2 b);
HNum2 div(in HNum2 a, in float b);
HNum2 div(in float a, in HNum2 b);
HNum2 inv(in HNum2 a);
HNum2 a_pow(in HNum2 a, in HNum2 b);
HNum2 a_pow(in HNum2 a, in float b);
HNum2 a_pow(in float a, in HNum2 b);
HNum2 a_min(in HNum2 a, in HNum2 b);
HNum2 a_max(in HNum2 a, in HNum2 b);
HNum2 a_exp2(in HNum2 a);
HNum2 a_inversesqrt(in HNum2 a);
HNum2 a_atan(in HNum2 a);
HNum2 a_sqrt(in HNum2 a);
HNum2 a_sinh(in HNum2 a);
HNum2 a_ceil(in HNum2 a);
HNum2 a_tan(in HNum2 a);
HNum2 a_asinh(in HNum2 a);
HNum2 a_asin(in HNum2 a);
HNum2 a_acosh(in HNum2 a);
HNum2 a_abs(in HNum2 a);
HNum2 a_exp(in HNum2 a);
HNum2 a_cosh(in HNum2 a);
HNum2 a_floor(in HNum2 a);
HNum2 a_log(in HNum2 a);
HNum2 a_atanh(in HNum2 a);
HNum2 a_log2(in HNum2 a);
HNum2 a_acos(in HNum2 a);
HNum2 a_tanh(in HNum2 a);
HNum2 a_cos(in HNum2 a);
HNum2 a_sin(in HNum2 a);
HNum2 a_atan2(in HNum2 y, in HNum2 x);
HNum2 a_atan2(in HNum2 y, in float x);
HNum2 a_atan2(in float y, in HNum2 x);
HNum2 a_mix(in HNum2 a, in HNum2 b, in HNum2 t);
HNum2 a_mix(in HNum2 a, in HNum2 b, in float t);
HNum2 a_mix(in HNum2 a, in float b, in HNum2 t);
HNum2 a_mix(in HNum2 a, in float b, in float t);
HNum2 a_mix(in float a, in HNum2 b, in HNum2 t);
HNum2 a_mix(in float a, in HNum2 b, in float t);
HNum2 a_mix(in float a, in float b, in HNum2 t);
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
HNum3 a_atan2(in HNum3 y, in HNum3 x);
HNum3 a_atan2(in HNum3 y, in float x);
HNum3 a_atan2(in float y, in HNum3 x);
HNum3 a_mix(in HNum3 a, in HNum3 b, in HNum3 t);
HNum3 a_mix(in HNum3 a, in HNum3 b, in float t);
HNum3 a_mix(in HNum3 a, in float b, in HNum3 t);
HNum3 a_mix(in HNum3 a, in float b, in float t);
HNum3 a_mix(in float a, in HNum3 b, in HNum3 t);
HNum3 a_mix(in float a, in HNum3 b, in float t);
HNum3 a_mix(in float a, in float b, in HNum3 t);
/**
* Creates a constant HNum4
* @param val The current value of the constant
*/
HNum4 constH4(in float val);
/**
* Creates a HNum4 corresponding to the variable with the given index
* @param val The current value of the variable
* @param index The variable's index
*/
HNum4 varH4(in float val, in int index);
/**
* Creates a HNum4 corresponding to the variable x (index = 0)
* @param val The current value of the variable
*/
HNum4 varH4x(in float val);
/**
* Creates a HNum4 corresponding to the variable y (index = 1)
* @param val The current value of the variable
*/
HNum4 varH4y(in float val);
/**
* Creates a HNum4 corresponding to the variable z (index = 2)
* @param val The current value of the variable
*/
HNum4 varH4z(in float val);
/**
* Creates a HNum4 corresponding to the variable w (index = 3)
* @param val The current value of the variable
*/
HNum4 varH4w(in float val);
HNum4 add(in HNum4 a, in HNum4 b);
HNum4 add(in HNum4 a, in float b);
HNum4 add(in float a, in HNum4 b);
HNum4 sub(in HNum4 a, in HNum4 b);
HNum4 sub(in HNum4 a, in float b);
HNum4 sub(in float a, in HNum4 b);
HNum4 mult(in HNum4 a, in HNum4 b);
HNum4 mult(in HNum4 a, in float b);
HNum4 mult(in float a, in HNum4 b);
HNum4 neg(in HNum4 a);
HNum4 div(in HNum4 a, in HNum4 b);
HNum4 div(in HNum4 a, in float b);
HNum4 div(in float a, in HNum4 b);
HNum4 inv(in HNum4 a);
HNum4 a_pow(in HNum4 a, in HNum4 b);
HNum4 a_pow(in HNum4 a, in float b);
HNum4 a_pow(in float a, in HNum4 b);
HNum4 a_min(in HNum4 a, in HNum4 b);
HNum4 a_max(in HNum4 a, in HNum4 b);
HNum4 a_exp2(in HNum4 a);
HNum4 a_inversesqrt(in HNum4 a);
HNum4 a_atan(in HNum4 a);
HNum4 a_sqrt(in HNum4 a);
HNum4 a_sinh(in HNum4 a);
HNum4 a_ceil(in HNum4 a);
HNum4 a_tan(in HNum4 a);
HNum4 a_asinh(in HNum4 a);
HNum4 a_asin(in HNum4 a);
HNum4 a_acosh(in HNum4 a);
HNum4 a_abs(in HNum4 a);
HNum4 a_exp(in HNum4 a);
HNum4 a_cosh(in HNum4 a);
HNum4 a_floor(in HNum4 a);
HNum4 a_log(in HNum4 a);
HNum4 a_atanh(in HNum4 a);
HNum4 a_log2(in HNum4 a);
HNum4 a_acos(in HNum4 a);
HNum4 a_tanh(in HNum4 a);
HNum4 a_cos(in HNum4 a);
HNum4 a_sin(in HNum4 a);
HNum4 a_atan2(in HNum4 y, in HNum4 x);
HNum4 a_atan2(in HNum4 y, in float x);
HNum4 a_atan2(in float y, in HNum4 x);
HNum4 a_mix(in HNum4 a, in HNum4 b, in HNum4 t);
HNum4 a_mix(in HNum4 a, in HNum4 b, in float t);
HNum4 a_mix(in HNum4 a, in float b, in HNum4 t);
HNum4 a_mix(in HNum4 a, in float b, in float t);
HNum4 a_mix(in float a, in HNum4 b, in HNum4 t);
HNum4 a_mix(in float a, in HNum4 b, in float t);
HNum4 a_mix(in float a, in float b, in HNum4 t);

//--------------------------------
// Macros
//--------------------------------

#define HESSIAN2(f,x, y,result)  {     result = f(varH2x(x), varH2y(y)); }
//--------------------------------
#define HESSIAN3(f,x, y, z,result)  {     result = f(varH3x(x), varH3y(y), varH3z(z)); }
//--------------------------------
#define HESSIAN4(f,x, y, z, w,result)  {     result = f(varH4x(x), varH4y(y), varH4z(z), varH4w(w)); }

//--------------------------------
// Gradient
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
GNum2 a_atan2(in GNum2 y, in GNum2 x);
GNum2 a_atan2(in GNum2 y, in float x);
GNum2 a_atan2(in float y, in GNum2 x);
GNum2 a_mix(in GNum2 a, in GNum2 b, in GNum2 t);
GNum2 a_mix(in GNum2 a, in GNum2 b, in float t);
GNum2 a_mix(in GNum2 a, in float b, in GNum2 t);
GNum2 a_mix(in GNum2 a, in float b, in float t);
GNum2 a_mix(in float a, in GNum2 b, in GNum2 t);
GNum2 a_mix(in float a, in GNum2 b, in float t);
GNum2 a_mix(in float a, in float b, in GNum2 t);
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
GNum4 a_atan2(in GNum4 y, in GNum4 x);
GNum4 a_atan2(in GNum4 y, in float x);
GNum4 a_atan2(in float y, in GNum4 x);
GNum4 a_mix(in GNum4 a, in GNum4 b, in GNum4 t);
GNum4 a_mix(in GNum4 a, in GNum4 b, in float t);
GNum4 a_mix(in GNum4 a, in float b, in GNum4 t);
GNum4 a_mix(in GNum4 a, in float b, in float t);
GNum4 a_mix(in float a, in GNum4 b, in GNum4 t);
GNum4 a_mix(in float a, in GNum4 b, in float t);
GNum4 a_mix(in float a, in float b, in GNum4 t);

//--------------------------------
// Univariate
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
/**
* Creates a constant derivative number
* @param val The current value of the constant
*/
vec3 constD2(in float val);
/**
* Creates a derivative number stored in a vec3
* @param val The current value of the variable
*/
vec3 varD2(in float val);
vec3 add(in vec3 a, in vec3 b);
vec3 add(in vec3 a, in float b);
vec3 add(in float a, in vec3 b);
vec3 sub(in vec3 a, in vec3 b);
vec3 sub(in vec3 a, in float b);
vec3 sub(in float a, in vec3 b);
vec3 mult(in vec3 a, in vec3 b);
vec3 mult(in vec3 a, in float b);
vec3 mult(in float a, in vec3 b);
vec3 neg(in vec3 a);
vec3 div(in vec3 a, in vec3 b);
vec3 div(in vec3 a, in float b);
vec3 div(in float a, in vec3 b);
vec3 inv(in vec3 a);
vec3 a_pow(in vec3 a, in vec3 b);
vec3 a_pow(in vec3 a, in float b);
vec3 a_pow(in float a, in vec3 b);
vec3 a_min(in vec3 a, in vec3 b);
vec3 a_max(in vec3 a, in vec3 b);
vec3 a_exp2(in vec3 a);
vec3 a_inversesqrt(in vec3 a);
vec3 a_atan(in vec3 a);
vec3 a_sqrt(in vec3 a);
vec3 a_sinh(in vec3 a);
vec3 a_ceil(in vec3 a);
vec3 a_tan(in vec3 a);
vec3 a_asinh(in vec3 a);
vec3 a_asin(in vec3 a);
vec3 a_acosh(in vec3 a);
vec3 a_abs(in vec3 a);
vec3 a_exp(in vec3 a);
vec3 a_cosh(in vec3 a);
vec3 a_floor(in vec3 a);
vec3 a_log(in vec3 a);
vec3 a_atanh(in vec3 a);
vec3 a_log2(in vec3 a);
vec3 a_acos(in vec3 a);
vec3 a_tanh(in vec3 a);
vec3 a_cos(in vec3 a);
vec3 a_sin(in vec3 a);
vec3 a_atan2(in vec3 y, in vec3 x);
vec3 a_atan2(in vec3 y, in float x);
vec3 a_atan2(in float y, in vec3 x);
vec3 a_mix(in vec3 a, in vec3 b, in vec3 t);
vec3 a_mix(in vec3 a, in vec3 b, in float t);
vec3 a_mix(in vec3 a, in float b, in vec3 t);
vec3 a_mix(in vec3 a, in float b, in float t);
vec3 a_mix(in float a, in vec3 b, in vec3 t);
vec3 a_mix(in float a, in vec3 b, in float t);
vec3 a_mix(in float a, in float b, in vec3 t);

//--------------------------------
// Utilities prototypes
//--------------------------------

mat2 a_outerProduct(in vec2 a, in vec2 b);
mat3 a_outerProduct(in vec3 a, in vec3 b);
mat4 a_outerProduct(in vec4 a, in vec4 b);

//--------------------------------
// Implementation
//--------------------------------

//--------------------------------
// Hessian
//--------------------------------

HNum2 constH2(in float val)
{
    return HNum2(val, vec2(0.0), mat2(0.0));
}
//--------------------------------
HNum2 varH2(in float val, in int index)
{   
    vec2 g = vec2(0.0);
    g[index] = 1.0;
    return HNum2(val, g, mat2(0.0));
}
//--------------------------------
HNum2 varH2x(in float val)
{   
    vec2 g = vec2(0.0);
    g[0] = 1.0;
    return HNum2(val, g, mat2(0.0));
}
//--------------------------------
HNum2 varH2y(in float val)
{   
    vec2 g = vec2(0.0);
    g[1] = 1.0;
    return HNum2(val, g, mat2(0.0));
}
//--------------------------------
HNum2 add(in HNum2 a, in HNum2 b)
{
    return HNum2(a.val + b.val , a.g + b.g, a.h + b.h);
}
//--------------------------------
HNum2 add(in HNum2 a, in float b)
{
    return HNum2(a.val + b , a.g, a.h);
}
//--------------------------------
HNum2 add(in float a, in HNum2 b)
{
    return HNum2(a + b.val , b.g, b.h);
}
//--------------------------------
HNum2 sub(in HNum2 a, in HNum2 b)
{
    return HNum2(a.val - b.val , a.g - b.g, a.h - b.h);
}
//--------------------------------
HNum2 sub(in HNum2 a, in float b)
{
    return HNum2(a.val - b , a.g, a.h);
}
//--------------------------------
HNum2 sub(in float a, in HNum2 b)
{
    return HNum2(a - b.val , - b.g, - b.h);
}
//--------------------------------
HNum2 mult(in HNum2 a, in HNum2 b)
{
    return HNum2(a.val * b.val, 
        a.val*b.g + b.val*a.g, 
        a.val*b.h + b.val*a.h + a_outerProduct(b.g,a.g) + a_outerProduct(a.g,b.g)
    );
}
//--------------------------------
HNum2 mult(in HNum2 a, in float b)
{
    return HNum2(a.val * b, b*a.g, b*a.h);
}
//--------------------------------
HNum2 mult(in float a, in HNum2 b)
{
    return HNum2(a * b.val, a*b.g, a*b.h);
}
//--------------------------------
HNum2 neg(in HNum2 a)
{
    return mult(-1.0,a);
}
//--------------------------------
HNum2 div(in HNum2 a, in HNum2 b)
{
    float b1 = b.val;
    float b2 = b1*b1;
    float b3 = b2*b1;

    return HNum2(a.val / b.val , 
        (b.val*a.g - a.val*b.g)/b2, 
        2.0*a.val/b3*a_outerProduct(b.g,b.g) 
        - a.val/b2*b.h
        + a.h/b1 
        - a_outerProduct(b.g/b2, a.g)
        - a_outerProduct(a.g/b2, b.g)
    );
}
//--------------------------------
HNum2 div(in HNum2 a, in float b)
{
    return HNum2(a.val / b, a.g/b, a.h/b);
}
//--------------------------------
HNum2 div(in float a, in HNum2 b)
{
    float b1 = b.val;
    float b2 = b1*b1;
    float b3 = b2*b1;

    return HNum2(a / b.val, 
        -a*b.g/b2, 
        2.0*a/b3*a_outerProduct(b.g,b.g) - a/b2*b.h
    );
}
//--------------------------------
HNum2 inv(in HNum2 a)
{
    return div(1.0, a);
}
//--------------------------------
HNum2 a_pow(in HNum2 a, in HNum2 b)
{
    return a_exp(mult(b,a_log(a)));
}
//--------------------------------
HNum2 a_pow(in HNum2 a, in float b)
{
    // constant exponent -> make special case
    float v = pow(a.val, b); // value f(a(x))
    float da = b*pow(a.val,b-1.0); // first derivative f'(a(x))
    float dda = b*(b-1.0)*pow(a.val,b-2.0); // second derivative f''(a(x))
    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_pow(in float a, in HNum2 b)
{
    return a_exp(mult(b,log(a)));
}
//--------------------------------
HNum2 a_min(in HNum2 a, in HNum2 b)
{
    if(a.val < b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
HNum2 a_max(in HNum2 a, in HNum2 b)
{
    if(a.val > b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
HNum2 a_exp2(in HNum2 a)
{
    float v = exp2(a.val); // value f(a(x))
    float da = log(2.0)*exp2(a.val); // first derivative f'(a(x))
    float dda = log(2.0)*log(2.0)*exp2(a.val); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_inversesqrt(in HNum2 a)
{
    float v = inversesqrt(a.val); // value f(a(x))
    float da = -0.5/pow(sqrt(a.val),3.0); // first derivative f'(a(x))
    float dda = 0.75/pow(sqrt(a.val),5.0); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_atan(in HNum2 a)
{
    float v = atan(a.val); // value f(a(x))
    float da = 1.0/(1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -2.0*a.val/pow(1.0 + a.val * a.val, 2.0); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_sqrt(in HNum2 a)
{
    float v = sqrt(a.val); // value f(a(x))
    float da = 0.5/sqrt(a.val); // first derivative f'(a(x))
    float dda = -0.25/pow(sqrt(a.val),3.0); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_sinh(in HNum2 a)
{
    float v = sinh(a.val); // value f(a(x))
    float da = cosh(a.val); // first derivative f'(a(x))
    float dda = sinh(a.val); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_ceil(in HNum2 a)
{
    float v = ceil(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_tan(in HNum2 a)
{
    float v = tan(a.val); // value f(a(x))
    float da = 1.0 + pow(tan(a.val),2.0); // first derivative f'(a(x))
    float dda = 2.0*tan(a.val)*(1.0 + pow(tan(a.val),2.0)); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_asinh(in HNum2 a)
{
    float v = asinh(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(1.0 + a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_asin(in HNum2 a)
{
    float v = asin(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = a.val/pow(sqrt(1.0 - a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_acosh(in HNum2 a)
{
    float v = acosh(a.val); // value f(a(x))
    float da = 1.0/sqrt(-1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(-1.0 + a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_abs(in HNum2 a)
{
    float v = abs(a.val); // value f(a(x))
    float da = a.val < 0.0 ? -1.0 : 1.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_exp(in HNum2 a)
{
    float v = exp(a.val); // value f(a(x))
    float da = exp(a.val); // first derivative f'(a(x))
    float dda = exp(a.val); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_cosh(in HNum2 a)
{
    float v = cosh(a.val); // value f(a(x))
    float da = sinh(a.val); // first derivative f'(a(x))
    float dda = cosh(a.val); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_floor(in HNum2 a)
{
    float v = floor(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_log(in HNum2 a)
{
    float v = log(a.val); // value f(a(x))
    float da = 1.0/a.val; // first derivative f'(a(x))
    float dda = -1.0/(a.val * a.val); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_atanh(in HNum2 a)
{
    float v = atanh(a.val); // value f(a(x))
    float da = 1.0/(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = 2.0*a.val/pow(1.0 - a.val * a.val,2.0); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_log2(in HNum2 a)
{
    float v = log2(a.val); // value f(a(x))
    float da = 1.0/(a.val * log(2.0)); // first derivative f'(a(x))
    float dda = -1.0/(a.val * a.val * log(2.0)); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_acos(in HNum2 a)
{
    float v = acos(a.val); // value f(a(x))
    float da = -1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(1.0 - a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_tanh(in HNum2 a)
{
    float v = tanh(a.val); // value f(a(x))
    float da = 1.0 - pow(tanh(a.val),2.0); // first derivative f'(a(x))
    float dda = -2.0*tanh(a.val)*(1.0 - pow(tanh(a.val),2.0)); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_cos(in HNum2 a)
{
    float v = cos(a.val); // value f(a(x))
    float da = -sin(a.val); // first derivative f'(a(x))
    float dda = -cos(a.val); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_sin(in HNum2 a)
{
    float v = sin(a.val); // value f(a(x))
    float da = cos(a.val); // first derivative f'(a(x))
    float dda = -sin(a.val); // second derivative f''(a(x))

    return HNum2(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum2 a_atan2(in HNum2 y, in HNum2 x)
{
    const float pi = 3.14159265; 
    // from https://en.wikipedia.org/wiki/Atan2
    if(x.val > 0.0)
    {
        HNum2 n = a_sqrt(add(mult(x,x),mult(y,y)));
        HNum2 inner = div(y, add(n,x));
        
        return mult(2.0,a_atan(inner));
        
    }else if(x.val <= 0.0 && abs(y.val) > 1E-6)
    {
        HNum2 n = a_sqrt(add(mult(x,x),mult(y,y)));
        HNum2 inner = div(sub(n,x),y);
         return mult(2.0,a_atan(inner));
    }else if(x.val < 0.0 && abs(y.val) <= 1E-6)
    {
        return constH2(pi);
    }
    // return 0 for undefined
    return constH2(0.0); 
}
//--------------------------------
HNum2 a_atan2(in HNum2 y, in float x)
{
    return a_atan2(y,constH2(x));
}
//--------------------------------
HNum2 a_atan2(in float y, in HNum2 x)
{
    return a_atan2(constH2(y),x);
}
//--------------------------------
HNum2 a_mix(in HNum2 a, in HNum2 b, in HNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum2 a_mix(in HNum2 a, in HNum2 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

//--------------------------------
HNum2 a_mix(in HNum2 a, in float b, in HNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum2 a_mix(in HNum2 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

//--------------------------------
HNum2 a_mix(in float a, in HNum2 b, in HNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum2 a_mix(in float a, in HNum2 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

//--------------------------------
HNum2 a_mix(in float a, in float b, in HNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

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
HNum3 a_atan2(in HNum3 y, in HNum3 x)
{
    const float pi = 3.14159265; 
    // from https://en.wikipedia.org/wiki/Atan2
    if(x.val > 0.0)
    {
        HNum3 n = a_sqrt(add(mult(x,x),mult(y,y)));
        HNum3 inner = div(y, add(n,x));
        
        return mult(2.0,a_atan(inner));
        
    }else if(x.val <= 0.0 && abs(y.val) > 1E-6)
    {
        HNum3 n = a_sqrt(add(mult(x,x),mult(y,y)));
        HNum3 inner = div(sub(n,x),y);
         return mult(2.0,a_atan(inner));
    }else if(x.val < 0.0 && abs(y.val) <= 1E-6)
    {
        return constH3(pi);
    }
    // return 0 for undefined
    return constH3(0.0); 
}
//--------------------------------
HNum3 a_atan2(in HNum3 y, in float x)
{
    return a_atan2(y,constH3(x));
}
//--------------------------------
HNum3 a_atan2(in float y, in HNum3 x)
{
    return a_atan2(constH3(y),x);
}
//--------------------------------
HNum3 a_mix(in HNum3 a, in HNum3 b, in HNum3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum3 a_mix(in HNum3 a, in HNum3 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

//--------------------------------
HNum3 a_mix(in HNum3 a, in float b, in HNum3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum3 a_mix(in HNum3 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

//--------------------------------
HNum3 a_mix(in float a, in HNum3 b, in HNum3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum3 a_mix(in float a, in HNum3 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

//--------------------------------
HNum3 a_mix(in float a, in float b, in HNum3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum4 constH4(in float val)
{
    return HNum4(val, vec4(0.0), mat4(0.0));
}
//--------------------------------
HNum4 varH4(in float val, in int index)
{   
    vec4 g = vec4(0.0);
    g[index] = 1.0;
    return HNum4(val, g, mat4(0.0));
}
//--------------------------------
HNum4 varH4x(in float val)
{   
    vec4 g = vec4(0.0);
    g[0] = 1.0;
    return HNum4(val, g, mat4(0.0));
}
//--------------------------------
HNum4 varH4y(in float val)
{   
    vec4 g = vec4(0.0);
    g[1] = 1.0;
    return HNum4(val, g, mat4(0.0));
}
//--------------------------------
HNum4 varH4z(in float val)
{   
    vec4 g = vec4(0.0);
    g[2] = 1.0;
    return HNum4(val, g, mat4(0.0));
}
//--------------------------------
HNum4 varH4w(in float val)
{   
    vec4 g = vec4(0.0);
    g[3] = 1.0;
    return HNum4(val, g, mat4(0.0));
}
//--------------------------------
HNum4 add(in HNum4 a, in HNum4 b)
{
    return HNum4(a.val + b.val , a.g + b.g, a.h + b.h);
}
//--------------------------------
HNum4 add(in HNum4 a, in float b)
{
    return HNum4(a.val + b , a.g, a.h);
}
//--------------------------------
HNum4 add(in float a, in HNum4 b)
{
    return HNum4(a + b.val , b.g, b.h);
}
//--------------------------------
HNum4 sub(in HNum4 a, in HNum4 b)
{
    return HNum4(a.val - b.val , a.g - b.g, a.h - b.h);
}
//--------------------------------
HNum4 sub(in HNum4 a, in float b)
{
    return HNum4(a.val - b , a.g, a.h);
}
//--------------------------------
HNum4 sub(in float a, in HNum4 b)
{
    return HNum4(a - b.val , - b.g, - b.h);
}
//--------------------------------
HNum4 mult(in HNum4 a, in HNum4 b)
{
    return HNum4(a.val * b.val, 
        a.val*b.g + b.val*a.g, 
        a.val*b.h + b.val*a.h + a_outerProduct(b.g,a.g) + a_outerProduct(a.g,b.g)
    );
}
//--------------------------------
HNum4 mult(in HNum4 a, in float b)
{
    return HNum4(a.val * b, b*a.g, b*a.h);
}
//--------------------------------
HNum4 mult(in float a, in HNum4 b)
{
    return HNum4(a * b.val, a*b.g, a*b.h);
}
//--------------------------------
HNum4 neg(in HNum4 a)
{
    return mult(-1.0,a);
}
//--------------------------------
HNum4 div(in HNum4 a, in HNum4 b)
{
    float b1 = b.val;
    float b2 = b1*b1;
    float b3 = b2*b1;

    return HNum4(a.val / b.val , 
        (b.val*a.g - a.val*b.g)/b2, 
        2.0*a.val/b3*a_outerProduct(b.g,b.g) 
        - a.val/b2*b.h
        + a.h/b1 
        - a_outerProduct(b.g/b2, a.g)
        - a_outerProduct(a.g/b2, b.g)
    );
}
//--------------------------------
HNum4 div(in HNum4 a, in float b)
{
    return HNum4(a.val / b, a.g/b, a.h/b);
}
//--------------------------------
HNum4 div(in float a, in HNum4 b)
{
    float b1 = b.val;
    float b2 = b1*b1;
    float b3 = b2*b1;

    return HNum4(a / b.val, 
        -a*b.g/b2, 
        2.0*a/b3*a_outerProduct(b.g,b.g) - a/b2*b.h
    );
}
//--------------------------------
HNum4 inv(in HNum4 a)
{
    return div(1.0, a);
}
//--------------------------------
HNum4 a_pow(in HNum4 a, in HNum4 b)
{
    return a_exp(mult(b,a_log(a)));
}
//--------------------------------
HNum4 a_pow(in HNum4 a, in float b)
{
    // constant exponent -> make special case
    float v = pow(a.val, b); // value f(a(x))
    float da = b*pow(a.val,b-1.0); // first derivative f'(a(x))
    float dda = b*(b-1.0)*pow(a.val,b-2.0); // second derivative f''(a(x))
    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_pow(in float a, in HNum4 b)
{
    return a_exp(mult(b,log(a)));
}
//--------------------------------
HNum4 a_min(in HNum4 a, in HNum4 b)
{
    if(a.val < b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
HNum4 a_max(in HNum4 a, in HNum4 b)
{
    if(a.val > b.val)
    {
        return a;
    }
    return b;
}
//--------------------------------
HNum4 a_exp2(in HNum4 a)
{
    float v = exp2(a.val); // value f(a(x))
    float da = log(2.0)*exp2(a.val); // first derivative f'(a(x))
    float dda = log(2.0)*log(2.0)*exp2(a.val); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_inversesqrt(in HNum4 a)
{
    float v = inversesqrt(a.val); // value f(a(x))
    float da = -0.5/pow(sqrt(a.val),3.0); // first derivative f'(a(x))
    float dda = 0.75/pow(sqrt(a.val),5.0); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_atan(in HNum4 a)
{
    float v = atan(a.val); // value f(a(x))
    float da = 1.0/(1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -2.0*a.val/pow(1.0 + a.val * a.val, 2.0); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_sqrt(in HNum4 a)
{
    float v = sqrt(a.val); // value f(a(x))
    float da = 0.5/sqrt(a.val); // first derivative f'(a(x))
    float dda = -0.25/pow(sqrt(a.val),3.0); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_sinh(in HNum4 a)
{
    float v = sinh(a.val); // value f(a(x))
    float da = cosh(a.val); // first derivative f'(a(x))
    float dda = sinh(a.val); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_ceil(in HNum4 a)
{
    float v = ceil(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_tan(in HNum4 a)
{
    float v = tan(a.val); // value f(a(x))
    float da = 1.0 + pow(tan(a.val),2.0); // first derivative f'(a(x))
    float dda = 2.0*tan(a.val)*(1.0 + pow(tan(a.val),2.0)); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_asinh(in HNum4 a)
{
    float v = asinh(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(1.0 + a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_asin(in HNum4 a)
{
    float v = asin(a.val); // value f(a(x))
    float da = 1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = a.val/pow(sqrt(1.0 - a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_acosh(in HNum4 a)
{
    float v = acosh(a.val); // value f(a(x))
    float da = 1.0/sqrt(-1.0 + a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(-1.0 + a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_abs(in HNum4 a)
{
    float v = abs(a.val); // value f(a(x))
    float da = a.val < 0.0 ? -1.0 : 1.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_exp(in HNum4 a)
{
    float v = exp(a.val); // value f(a(x))
    float da = exp(a.val); // first derivative f'(a(x))
    float dda = exp(a.val); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_cosh(in HNum4 a)
{
    float v = cosh(a.val); // value f(a(x))
    float da = sinh(a.val); // first derivative f'(a(x))
    float dda = cosh(a.val); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_floor(in HNum4 a)
{
    float v = floor(a.val); // value f(a(x))
    float da = 0.0; // first derivative f'(a(x))
    float dda = 0.0; // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_log(in HNum4 a)
{
    float v = log(a.val); // value f(a(x))
    float da = 1.0/a.val; // first derivative f'(a(x))
    float dda = -1.0/(a.val * a.val); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_atanh(in HNum4 a)
{
    float v = atanh(a.val); // value f(a(x))
    float da = 1.0/(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = 2.0*a.val/pow(1.0 - a.val * a.val,2.0); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_log2(in HNum4 a)
{
    float v = log2(a.val); // value f(a(x))
    float da = 1.0/(a.val * log(2.0)); // first derivative f'(a(x))
    float dda = -1.0/(a.val * a.val * log(2.0)); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_acos(in HNum4 a)
{
    float v = acos(a.val); // value f(a(x))
    float da = -1.0/sqrt(1.0 - a.val * a.val); // first derivative f'(a(x))
    float dda = -a.val/pow(sqrt(1.0 - a.val * a.val),3.0); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_tanh(in HNum4 a)
{
    float v = tanh(a.val); // value f(a(x))
    float da = 1.0 - pow(tanh(a.val),2.0); // first derivative f'(a(x))
    float dda = -2.0*tanh(a.val)*(1.0 - pow(tanh(a.val),2.0)); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_cos(in HNum4 a)
{
    float v = cos(a.val); // value f(a(x))
    float da = -sin(a.val); // first derivative f'(a(x))
    float dda = -cos(a.val); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_sin(in HNum4 a)
{
    float v = sin(a.val); // value f(a(x))
    float da = cos(a.val); // first derivative f'(a(x))
    float dda = -sin(a.val); // second derivative f''(a(x))

    return HNum4(v , da * a.g,  da * a.h + dda * a_outerProduct(a.g,a.g));
}
//--------------------------------
HNum4 a_atan2(in HNum4 y, in HNum4 x)
{
    const float pi = 3.14159265; 
    // from https://en.wikipedia.org/wiki/Atan2
    if(x.val > 0.0)
    {
        HNum4 n = a_sqrt(add(mult(x,x),mult(y,y)));
        HNum4 inner = div(y, add(n,x));
        
        return mult(2.0,a_atan(inner));
        
    }else if(x.val <= 0.0 && abs(y.val) > 1E-6)
    {
        HNum4 n = a_sqrt(add(mult(x,x),mult(y,y)));
        HNum4 inner = div(sub(n,x),y);
         return mult(2.0,a_atan(inner));
    }else if(x.val < 0.0 && abs(y.val) <= 1E-6)
    {
        return constH4(pi);
    }
    // return 0 for undefined
    return constH4(0.0); 
}
//--------------------------------
HNum4 a_atan2(in HNum4 y, in float x)
{
    return a_atan2(y,constH4(x));
}
//--------------------------------
HNum4 a_atan2(in float y, in HNum4 x)
{
    return a_atan2(constH4(y),x);
}
//--------------------------------
HNum4 a_mix(in HNum4 a, in HNum4 b, in HNum4 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum4 a_mix(in HNum4 a, in HNum4 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

//--------------------------------
HNum4 a_mix(in HNum4 a, in float b, in HNum4 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum4 a_mix(in HNum4 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

//--------------------------------
HNum4 a_mix(in float a, in HNum4 b, in HNum4 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
HNum4 a_mix(in float a, in HNum4 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

//--------------------------------
HNum4 a_mix(in float a, in float b, in HNum4 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}


//--------------------------------
// Gradient
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
//--------------------------------
GNum2 a_atan2(in GNum2 y, in GNum2 x)
{
    const float pi = 3.14159265; 
    // from https://en.wikipedia.org/wiki/Atan2
    if(x.val > 0.0)
    {
        GNum2 n = a_sqrt(add(mult(x,x),mult(y,y)));
        GNum2 inner = div(y, add(n,x));
        
        return mult(2.0,a_atan(inner));
        
    }else if(x.val <= 0.0 && abs(y.val) > 1E-6)
    {
        GNum2 n = a_sqrt(add(mult(x,x),mult(y,y)));
        GNum2 inner = div(sub(n,x),y);
         return mult(2.0,a_atan(inner));
    }else if(x.val < 0.0 && abs(y.val) <= 1E-6)
    {
        return constG2(pi);
    }
    // return 0 for undefined
    return constG2(0.0); 
}
//--------------------------------
GNum2 a_atan2(in GNum2 y, in float x)
{
    return a_atan2(y,constG2(x));
}
//--------------------------------
GNum2 a_atan2(in float y, in GNum2 x)
{
    return a_atan2(constG2(y),x);
}
//--------------------------------
GNum2 a_mix(in GNum2 a, in GNum2 b, in GNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum2 a_mix(in GNum2 a, in GNum2 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

//--------------------------------
GNum2 a_mix(in GNum2 a, in float b, in GNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum2 a_mix(in GNum2 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

//--------------------------------
GNum2 a_mix(in float a, in GNum2 b, in GNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum2 a_mix(in float a, in GNum2 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

//--------------------------------
GNum2 a_mix(in float a, in float b, in GNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

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
//--------------------------------
GNum4 a_atan2(in GNum4 y, in GNum4 x)
{
    const float pi = 3.14159265; 
    // from https://en.wikipedia.org/wiki/Atan2
    if(x.val > 0.0)
    {
        GNum4 n = a_sqrt(add(mult(x,x),mult(y,y)));
        GNum4 inner = div(y, add(n,x));
        
        return mult(2.0,a_atan(inner));
        
    }else if(x.val <= 0.0 && abs(y.val) > 1E-6)
    {
        GNum4 n = a_sqrt(add(mult(x,x),mult(y,y)));
        GNum4 inner = div(sub(n,x),y);
         return mult(2.0,a_atan(inner));
    }else if(x.val < 0.0 && abs(y.val) <= 1E-6)
    {
        return constG4(pi);
    }
    // return 0 for undefined
    return constG4(0.0); 
}
//--------------------------------
GNum4 a_atan2(in GNum4 y, in float x)
{
    return a_atan2(y,constG4(x));
}
//--------------------------------
GNum4 a_atan2(in float y, in GNum4 x)
{
    return a_atan2(constG4(y),x);
}
//--------------------------------
GNum4 a_mix(in GNum4 a, in GNum4 b, in GNum4 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum4 a_mix(in GNum4 a, in GNum4 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

//--------------------------------
GNum4 a_mix(in GNum4 a, in float b, in GNum4 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum4 a_mix(in GNum4 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

//--------------------------------
GNum4 a_mix(in float a, in GNum4 b, in GNum4 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
GNum4 a_mix(in float a, in GNum4 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

//--------------------------------
GNum4 a_mix(in float a, in float b, in GNum4 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}


//--------------------------------
// Univariate
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

//--------------------------------
vec3 constD2(in float val)
{
    vec3 c = vec3(0.0);
    c[0] = val;
    return c;
}
//--------------------------------
vec3 varD2(in float val)
{   
    return vec3(val, 1.0, 0.0);
}
//--------------------------------
vec3 add(in vec3 a, in vec3 b)
{
    return a + b;
}
//--------------------------------
vec3 add(in vec3 a, in float b)
{
    a[0] += b;
    return a;
}
//--------------------------------
vec3 add(in float a, in vec3 b)
{
    b[0] += a;
    return b;
}
//--------------------------------
vec3 sub(in vec3 a, in vec3 b)
{
    return a - b;
}
//--------------------------------
vec3 sub(in vec3 a, in float b)
{
    a[0] -= b;
    return a;
}
//--------------------------------
vec3 sub(in float a, in vec3 b)
{
    b *= -1.0;
    b[0] += a;
    return b;
}
//--------------------------------
vec3 mult(in vec3 a, in vec3 b)
{
    float v = a[0] * b[0];
	float da = a[0]*b[1] + b[0]*a[1];
	float dda = a[2]*b[0] + 2.0*a[1]*b[1] + a[0]*b[2];

    return vec3(v,da,dda);
}
//--------------------------------
vec3 mult(in vec3 a, in float b)
{
    return a*b;
}
//--------------------------------
vec3 mult(in float a, in vec3 b)
{
    return a*b;
}
//--------------------------------
vec3 neg(in vec3 a)
{
    return mult(-1.0,a);
}
//--------------------------------
vec3 div(in vec3 a, in vec3 b)
{
    float v = a[0] / b[0];
	float b2 = b[0] * b[0];
float da = a[1]/b[0] - a[0]*b[1]/b2;
	float b3 = b2 * b[0];
float dda = a[2]/b[0] - 2.0*a[1]*b[1]/b2 - a[0]*b[2]/b2 + 2.0*a[0]*(b[1]*b[1])/b3;

    return vec3(v,da,dda);
}
//--------------------------------
vec3 div(in vec3 a, in float b)
{
    return a / b;
}
//--------------------------------
vec3 div(in float a, in vec3 b)
{
    float v = a / b[0];
	float b2 = b[0] * b[0];
float da = - a*b[1]/b2;
	float b3 = b2 * b[0];
float dda = - a*b[2]/b2 + 2.0*a*(b[1]*b[1])/b3;

    return vec3(v, da, dda);
}
//--------------------------------
vec3 inv(in vec3 a)
{
    return div(1.0, a);
}
//--------------------------------
vec3 a_pow(in vec3 a, in vec3 b)
{
    return a_exp(mult(b,a_log(a)));
}
//--------------------------------
vec3 a_pow(in vec3 a, in float b)
{
    float v = pow(a[0], b); // value f(a(x))
	float da = b*pow(a[0],b-1.0); // first derivative f'(a(x))
	float dda = b*(b-1.0)*pow(a[0],b-2.0); // second derivative f''(a(x))

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_pow(in float a, in vec3 b)
{
    return a_exp(mult(b,log(a)));
}
//--------------------------------
vec3 a_min(in vec3 a, in vec3 b)
{
    if(a[0] < b[0])
    {
        return a;
    }
    return b;
}
//--------------------------------
vec3 a_max(in vec3 a, in vec3 b)
{
    if(a[0] > b[0])
    {
        return a;
    }
    return b;
}
//--------------------------------
vec3 a_exp2(in vec3 a)
{
    float v = exp2(a[0]);
	float da = log(2.0)*exp2(a[0]);
	float dda = log(2.0)*log(2.0)*exp2(a[0]);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_inversesqrt(in vec3 a)
{
    float v = inversesqrt(a[0]);
	float da = -0.5/pow(sqrt(a[0]),3.0);
	float dda = 0.75/pow(sqrt(a[0]),5.0);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_atan(in vec3 a)
{
    float v = atan(a[0]);
	float da = 1.0/(1.0 + a[0] * a[0]);
	float dda = -2.0*a[0]/pow(1.0 + a[0] * a[0], 2.0);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_sqrt(in vec3 a)
{
    float v = sqrt(a[0]);
	float da = 0.5/sqrt(a[0]);
	float dda = -0.25/pow(sqrt(a[0]),3.0);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_sinh(in vec3 a)
{
    float v = sinh(a[0]);
	float da = cosh(a[0]);
	float dda = sinh(a[0]);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_ceil(in vec3 a)
{
    float v = ceil(a[0]);
	float da = 0.0;
	float dda = 0.0;

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_tan(in vec3 a)
{
    float v = tan(a[0]);
	float da = 1.0 + pow(tan(a[0]),2.0);
	float dda = 2.0*tan(a[0])*(1.0 + pow(tan(a[0]),2.0));

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_asinh(in vec3 a)
{
    float v = asinh(a[0]);
	float da = 1.0/sqrt(1.0 + a[0] * a[0]);
	float dda = -a[0]/pow(sqrt(1.0 + a[0] * a[0]),3.0);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_asin(in vec3 a)
{
    float v = asin(a[0]);
	float da = 1.0/sqrt(1.0 - a[0] * a[0]);
	float dda = a[0]/pow(sqrt(1.0 - a[0] * a[0]),3.0);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_acosh(in vec3 a)
{
    float v = acosh(a[0]);
	float da = 1.0/sqrt(-1.0 + a[0] * a[0]);
	float dda = -a[0]/pow(sqrt(-1.0 + a[0] * a[0]),3.0);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_abs(in vec3 a)
{
    float v = abs(a[0]);
	float da = a[0] < 0.0 ? -1.0 : 1.0;
	float dda = 0.0;

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_exp(in vec3 a)
{
    float v = exp(a[0]);
	float da = exp(a[0]);
	float dda = exp(a[0]);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_cosh(in vec3 a)
{
    float v = cosh(a[0]);
	float da = sinh(a[0]);
	float dda = cosh(a[0]);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_floor(in vec3 a)
{
    float v = floor(a[0]);
	float da = 0.0;
	float dda = 0.0;

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_log(in vec3 a)
{
    float v = log(a[0]);
	float da = 1.0/a[0];
	float dda = -1.0/(a[0] * a[0]);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_atanh(in vec3 a)
{
    float v = atanh(a[0]);
	float da = 1.0/(1.0 - a[0] * a[0]);
	float dda = 2.0*a[0]/pow(1.0 - a[0] * a[0],2.0);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_log2(in vec3 a)
{
    float v = log2(a[0]);
	float da = 1.0/(a[0] * log(2.0));
	float dda = -1.0/(a[0] * a[0] * log(2.0));

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_acos(in vec3 a)
{
    float v = acos(a[0]);
	float da = -1.0/sqrt(1.0 - a[0] * a[0]);
	float dda = -a[0]/pow(sqrt(1.0 - a[0] * a[0]),3.0);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_tanh(in vec3 a)
{
    float v = tanh(a[0]);
	float da = 1.0 - pow(tanh(a[0]),2.0);
	float dda = -2.0*tanh(a[0])*(1.0 - pow(tanh(a[0]),2.0));

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_cos(in vec3 a)
{
    float v = cos(a[0]);
	float da = -sin(a[0]);
	float dda = -cos(a[0]);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_sin(in vec3 a)
{
    float v = sin(a[0]);
	float da = cos(a[0]);
	float dda = -sin(a[0]);

    return vec3(v, da * a[1], da * a[2] + dda * a[1]*a[1]);
}
//--------------------------------
vec3 a_atan2(in vec3 y, in vec3 x)
{
    const float pi = 3.14159265; 
    // from https://en.wikipedia.org/wiki/Atan2
    if(x[0] > 0.0)
    {
        vec3 n = a_sqrt(add(mult(x,x),mult(y,y)));
        vec3 inner = div(y, add(n,x));
        
        return mult(2.0,a_atan(inner));
        
    }else if(x[0] <= 0.0 && abs(y[0]) > 1E-6)
    {
        vec3 n = a_sqrt(add(mult(x,x),mult(y,y)));
        vec3 inner = div(sub(n,x),y);
         return mult(2.0,a_atan(inner));
    }else if(x[0] < 0.0 && abs(y[0]) <= 1E-6)
    {
        return constD2(pi);
    }
    // return 0 for undefined
    return constD2(0.0); 
}
//--------------------------------
vec3 a_atan2(in vec3 y, in float x)
{
    return a_atan2(y,constD2(x));
}
//--------------------------------
vec3 a_atan2(in float y, in vec3 x)
{
    return a_atan2(constD2(y),x);
}
//--------------------------------
vec3 a_mix(in vec3 a, in vec3 b, in vec3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
vec3 a_mix(in vec3 a, in vec3 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

//--------------------------------
vec3 a_mix(in vec3 a, in float b, in vec3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
vec3 a_mix(in vec3 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

//--------------------------------
vec3 a_mix(in float a, in vec3 b, in vec3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

//--------------------------------
vec3 a_mix(in float a, in vec3 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

//--------------------------------
vec3 a_mix(in float a, in float b, in vec3 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}



//--------------------------------
// Implementation utils
//--------------------------------

mat2 a_outerProduct(in vec2 a, in vec2 b)
{
    return mat2(a * b[0], a * b[1]);
}

//--------------------------------
mat3 a_outerProduct(in vec3 a, in vec3 b)
{
    return mat3(a * b[0], a * b[1], a * b[2]);
}

//--------------------------------
mat4 a_outerProduct(in vec4 a, in vec4 b)
{
    return mat4(a * b[0], a * b[1], a * b[2], a * b[3]);
}


#endif // AUTODIFF_H_