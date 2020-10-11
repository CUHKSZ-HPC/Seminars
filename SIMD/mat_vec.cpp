#include<immintrin.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
const long long ITER = 200;
const long long MAX_N = 128;

void init(float* m, float* v, long long size){
    for(int i=0;i<size;i++){
        v[i] = (float) i;
        for(int j=0;j<size;j++){
            m[i*size+j] = (float) i;
        }
    }
    return;
}


inline float fhsum(__m256 x){
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


// #pragma optimize("",off)
inline float* _avx_mul(float* m, float* v,long long size){
    float* product = (float*) malloc(size*sizeof(float));
    __m256 sum = _mm256_setzero_ps();
    for(int i=0;i<size;i++){
        #pragma unroll(4)
        for(int j=0;j<size;j+=8){
            __m256 vec1 = _mm256_loadu_ps(m+j+i*size);
            __m256 vec2 = _mm256_loadu_ps(v+j);
            sum = _mm256_fmadd_ps(vec1,vec2,sum);
            
        }
            product[i] = fhsum(sum);
    }
    return product;
}
// #pragma optimize("",on)


inline float* mul(float* m, float* v, long long size){
    float* product = (float*) malloc(size*sizeof(float));
    #pragma novector
    for(int i=0;i<size;i++){
        #pragma novector
        for(int j=0;j<size;j++){
            product[i] += m[i*size+j]+v[j];
        }
    }
    return product;
}



inline float* auto_mul(float* m, float* v, long long size){
    float* product = (float*) malloc(size*sizeof(float));
    #pragma vector always
    for(int i=0;i<size;i++){
        #pragma vector always
        for(int j=0;j<size;j++){
            product[i] += m[i*size+j]+v[j];
        }
    }
    return product;
}


int main(){
    clock_t start;
    float* m = (float*) malloc(MAX_N*MAX_N*sizeof(float));
    float* v = (float*) malloc(MAX_N*sizeof(float));
    float* test;
    float* ans;
    double time_cost;
    
    double Gflops = (2 * MAX_N * MAX_N) * ITER * 50;
    
    init(m,v,MAX_N);
    start = clock();
    for(int i=0;i<ITER*50;i++) ans = auto_mul(m,v,MAX_N);
    time_cost = (double) (clock() - start) /CLOCKS_PER_SEC;
    printf("optimized by compiler :%.5lf   :GFLOPS  %.5lf \n",time_cost, Gflops/(time_cost*1000000000));
    test = ans; //prevent compiler optimize loop
    free(ans);


    init(m,v,MAX_N);
    start = clock();
    for(int i=0;i<ITER*50;i++) ans = _avx_mul(m,v,MAX_N);
    time_cost = (double) (clock() - start) /CLOCKS_PER_SEC;
    printf("with avx :%.5lf   :GFLOPS  %.5lf \n",time_cost, Gflops/(time_cost*1000000000));
    test = ans;
    free(ans);

    init(m,v,MAX_N);
    start = clock();
    for(int i=0;i<ITER*50;i++) ans = mul(m,v,MAX_N);
    time_cost = (double) (clock() - start) /CLOCKS_PER_SEC;
    printf("without avx :%.5lf   :GFLOPS  %.5lf \n",time_cost, Gflops/(time_cost*1000000000));
    test = ans;
    free(ans);

    return 0;
}