#ifndef MEMORYHELPER_H
#define MEMORYHELPER_H

// Local
#include "MetaHelper.h"
#include "vectorization.h"

//Default implementation work for non-vectorized case
template<typename T, class VecT>
class VectorizedMemOp {
 public:
  static VecT load( const T* ptr ) {
    return *ptr;
  }
  static void store( float* ptr, VecT value) {
    *ptr = value;
  }
};

#ifdef USE_AVX
template<>
class VectorizedMemOp<float,__m128> {
 public:
  static __m128 load( const float* ptr ) {
    return _mm_load_ps( ptr );
  }
  static void store( float* ptr, __m128 value) {
    _mm_store_ps( ptr, value );
  }
};
template<>
class VectorizedMemOp<double,__m128d> {
 public:
  static __m128d load( const double* ptr ) {
    return _mm_load_pd( ptr );
  }
  static void store( double* ptr, __m128d value) {
    _mm_store_pd( ptr, value );
  }
};
#elif defined USE_AVX2
template<>
class VectorizedMemOp<float,__m256> {
 public:
  static __m256 load( const float* ptr ) {
    return _mm256_load_ps( ptr );
  }
  static void store( float* ptr, __m256 value) {
    _mm256_store_ps( ptr, value );
  }
};
template<>
class VectorizedMemOp<double,__m256d> {
 public:
  static __m256d load( const double* ptr ) {
    return _mm256_load_pd( ptr );
  }
  static void store( double* ptr, __m256d value) {
    _mm256_store_pd( ptr, value );
  }
};
#elif defined USE_NEON
template<>
class VectorizedMemOp<float,float32x4_t> {
 public:
  static float32x4_t load( const float* ptr ) {
    return vld1q_f32( ptr );
  }
  static void store( float* ptr, float32x4_t value) {
    vst1q_f32( ptr, value );
  }
};
template<>
class VectorizedMemOp<double,float64x2_t> {
 public:
 static float64x2_t load( const double* ptr ) {
    return vld1q_f64( ptr );
  }
  static void store( double* ptr, float64x2_t value) {
    vst1q_f64( ptr, value );
  }
};
#endif
#endif //MEMORYHELPER_H

