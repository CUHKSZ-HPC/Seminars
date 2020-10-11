# SIMD Demo

## Description

These codes are SIMD Demo presented on Oct. 11, 2020 seminar using AVX2 intrinsic functions.  
vec_scalar.cpp is vector multiply scalar demo  
vec_vec.cpp is vector dot product vector demo  
mat_vec.cpp is matrix multiply vector demo

## How to compile it

In order to enable #pragma option, you should use intel c/c++ compiler  
icc -mavx2 -mfma -o test code.cpp -O2  


