#ifndef GRAD_H_
#define GRAD_H_

// This file contains methods to compute the gradient of a scalar valued
// function (2-4 variables) using automatic forward differentiation

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
GNum2 a_ipow(in GNum2 x, in int n);
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
GNum3 a_ipow(in GNum3 x, in int n);
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
GNum4 a_ipow(in GNum4 x, in int n);
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
GNum2 a_ipow(in GNum2 x, in int n)
{
    // based on https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    if (n < 0)
    {   
        x = div(1.0,x);
        n = -n;
    }
    if (n == 0) 
    {
        return constG2(1.0);
    }
    GNum2 y = constG2(1.0);
    while (n > 1)
    {
        if (n % 2 == 0)
        {   
            x = mult(x,x);
            
        }
        else
        {    
            y = mult(x, y);
            x = mult(x, x);
        }

        n = n / 2;
    }
    
    return mult(x, y);
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
GNum3 a_ipow(in GNum3 x, in int n)
{
    // based on https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    if (n < 0)
    {   
        x = div(1.0,x);
        n = -n;
    }
    if (n == 0) 
    {
        return constG3(1.0);
    }
    GNum3 y = constG3(1.0);
    while (n > 1)
    {
        if (n % 2 == 0)
        {   
            x = mult(x,x);
            
        }
        else
        {    
            y = mult(x, y);
            x = mult(x, x);
        }

        n = n / 2;
    }
    
    return mult(x, y);
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
GNum4 a_ipow(in GNum4 x, in int n)
{
    // based on https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    if (n < 0)
    {   
        x = div(1.0,x);
        n = -n;
    }
    if (n == 0) 
    {
        return constG4(1.0);
    }
    GNum4 y = constG4(1.0);
    while (n > 1)
    {
        if (n % 2 == 0)
        {   
            x = mult(x,x);
            
        }
        else
        {    
            y = mult(x, y);
            x = mult(x, x);
        }

        n = n / 2;
    }
    
    return mult(x, y);
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


#endif // GRAD_H_