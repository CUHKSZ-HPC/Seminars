#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<immintrin.h>

const long MAX_N = 2048;
const long ITER = 30000;
const long MAX_NN = MAX_N*ITER;

void init_vec(float* a, int size) {
	for (int i = 0; i < size; i++) a[i] = 1.0;
	return;
}


//use pragma unroll
// inline void _avx_mul(float*a, float c, int size) {
// 	__m256 vec;
// 	__m256 scalar = _mm256_set1_ps(c);
// 	#pragma unroll(4)
// 	for (int i = 0; i < size; i += 8) {
// 		vec = _mm256_load_ps(a + i);
// 		vec = _mm256_mul_ps(vec, scalar);
// 		_mm256_store_ps(a + i, vec);
// 	}
// 	return;
// }

inline void _avx_mul(float*a, float c, int size) {
	__m256 vec0,vec1,vec2,vec3;
	__m256 scalar = _mm256_set1_ps(c);
	for (int i = 0; i < size; i += 32) {
		vec0 = _mm256_load_ps(a + i);
		vec1 = _mm256_load_ps(a + i + 8);
		vec2 = _mm256_load_ps(a + i + 16);
		vec3 = _mm256_load_ps(a + i + 24);
		vec0 = _mm256_mul_ps(vec0, scalar);
		vec1 = _mm256_mul_ps(vec1, scalar);
		vec2 = _mm256_mul_ps(vec2, scalar);
		vec3 = _mm256_mul_ps(vec3, scalar);
		_mm256_store_ps(a + i, vec0);
		_mm256_store_ps(a + i + 8, vec1);
		_mm256_store_ps(a + i + 16, vec2);
		_mm256_store_ps(a + i + 32, vec3);
	}
	return;
}

inline void _avx_mul_not_unroll(float*a, float c, int size){
	__m256 vec;
	__m256 scalar = _mm256_set1_ps(c);
	for (int i = 0; i < size; i += 8) {
		vec = _mm256_load_ps(a + i);
		vec = _mm256_mul_ps(vec, scalar);
		_mm256_store_ps(a + i, vec);
	}
	return;
}

inline void mul(float*a, float c, int size) {
	#pragma novector
	for (int i = 0; i < size; i++) {
		a[i] = a[i] * c;
	}
	return;
}

inline void auto_mul(float*a, float c, int size) {

	#pragma vector always
	for (int i = 0; i < size; i++) {
		a[i] = a[i] * c;
	}
	return;
}



int main() {
	float* a;
	a = (float*) aligned_alloc(32, MAX_N*sizeof(float));
	clock_t start, end;
	double Gflops,time_cost;
	Gflops = MAX_N * ITER * 100;

	init_vec(a, MAX_N);
	start = clock();
	for(int i=0;i<ITER*100;i++)  _avx_mul(a, 1.1, MAX_N);
	end = clock();
	time_cost = (double)(end - start) / CLOCKS_PER_SEC;
	printf("with avx    time:%.4f  Gflops:%.2lf\n", time_cost , Gflops / (1000000000*time_cost) );

	for(int i=0;i<ITER*100;i++)  _avx_mul_not_unroll(a, 1.1, MAX_N);
	end = clock();
	time_cost = (double)(end - start) / CLOCKS_PER_SEC;
	printf("with avx not unrolled:    time:%.4f  Gflops:%.2lf\n", time_cost , Gflops / (1000000000*time_cost) );

	init_vec(a, MAX_N);
	start = clock();
	for (int i = 0; i < ITER*100; i++)  mul(a, 1.1, MAX_N);
	end = clock();
	time_cost = (double)(end - start) / CLOCKS_PER_SEC;
	printf("without avx:    time:%.4f  Gflops:%.2lf\n", time_cost , Gflops / (1000000000*time_cost) );

	init_vec(a, MAX_N);
	start = clock();
	for (int i = 0; i < ITER*100; i++)  auto_mul(a, 1.1, MAX_N);
	end = clock();
	time_cost = (double)(end - start) / CLOCKS_PER_SEC;
	printf("optimized by compiler:    time:%.4f  Gflops:%.2lf\n", time_cost , Gflops / (1000000000*time_cost) );
	free(a);

	float* b;
	b = (float*) malloc(MAX_NN*sizeof(float));
	init_vec(b,MAX_NN);
	start = clock();
	for(int i = 0; i < 100; i++)
	_avx_mul(b,1.1,MAX_NN);
	end = clock();
	time_cost = (double)(end - start) / CLOCKS_PER_SEC;
	printf("cold data with avx:  time:%.4f  Gflops:%.2lf\n", time_cost , Gflops / (1000000000*time_cost));

	init_vec(b,MAX_NN);
	start = clock();
	for (int i = 0; i < 100; i++)  mul(b,1.1,MAX_NN);
	end = clock();
	time_cost = (double)(end - start) / CLOCKS_PER_SEC;
	printf("cold data without avx:   time:%.4f  Gflops:%.2lf\n", time_cost , Gflops / (1000000000*time_cost));
	free(b);

	return 0;
}
