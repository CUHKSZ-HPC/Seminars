#include<cstdio>
#include<time.h>
#include<stdlib.h>
#include<immintrin.h>

const long long MAX_N = 4096;
const long long ITER = 100000;




void init(float* a,long long size){
    for(int i=0;i<size;i++) a[i] = (float) rand();
}



float auto_dot(const float* a, const float* b, long long size){
    double product = 0;
    #pragma vector always
    for(int i=0;i<size;i++){
        product+=a[i]*b[i];
    }
    return product;
}

float dot(const float* a, const float* b, long long size){
    double product = 0;
    #pragma novector
    for(int i=0;i<size;i++){
        product+=a[i]*b[i];
    }
    return product;
}



float fhsum(__m256 x){
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

float _avx_dot(const float* a, const float* b, long long size){
    //float product = 0;
    __m256 vec1,vec2,sum;
    sum = _mm256_setzero_ps();
    #pragma unroll(4)
    for(int i=0;i<size;i+=8){
        vec1 = _mm256_load_ps(a+i);
        vec2 = _mm256_load_ps(b+i);
        sum = _mm256_fmadd_ps(vec1,vec2,sum);
    }
    //reduce
    return fhsum(sum);
}


int main(){
    clock_t start;
    float* a = (float*) aligned_alloc(32,MAX_N*sizeof(float));
    float* b = (float*) aligned_alloc(32,MAX_N*sizeof(float));
    volatile float ans;
    double Gflops = 2*MAX_N * ITER *100,time_cost;
    init(a,MAX_N);
    init(b,MAX_N);
    start = clock();
    for(int i=0;i<ITER*100;i++)  ans = _avx_dot(a,b,MAX_N);
    time_cost = (double) (clock() - start)/CLOCKS_PER_SEC;
    printf("with avx2:  time  %.5lf  GFLOPS:%.5lf\n", time_cost, Gflops/(1000000000*time_cost));


    init(a,MAX_N);
    init(b,MAX_N);
    start = clock();
    for(int i=0;i<ITER*100;i++) ans = dot(a,b,MAX_N);
    time_cost = (double) (clock() - start)/CLOCKS_PER_SEC;
    printf("without avx2:  time  %.5lf  GFLOPS:%.5lf\n", time_cost, Gflops/(1000000000*time_cost));

    init(a,MAX_N);
    init(b,MAX_N);
    start = clock();
    for(int i=0;i<ITER*100;i++) ans = auto_dot(a,b,MAX_N);
    time_cost = (double) (clock() - start)/CLOCKS_PER_SEC;
    printf("compiler optimized :  time  %.5lf  GFLOPS:%.5lf\n", time_cost, Gflops/(1000000000*time_cost));


    return 0;
}