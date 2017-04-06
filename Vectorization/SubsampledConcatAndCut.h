#ifndef SUBSAMPLEDCONCATANDCUT_H
#define SUBSAMPLEDCONCATANDCUT_H

// Local
#include "MetaHelper.h"
#include "vectorization.h"

template<typename T, class VecT, int SHIFT>
struct SubsampledConcatAndCut {
  static VecT  Concat( VecT a, VecT b, VecT c) {
    assert(false);
  }
  static VecT  Concat( VecT a, VecT b) {
    assert(false);
  }
};

#ifdef USE_AVX
template<>
struct SubsampledConcatAndCut<float,__m128,0> {
  static __m128  Concat( __m128 a, __m128 b, __m128 c) {
    return _mm_blend_ps( _mm_permute_ps(a,216),_mm_permute_ps(b,141),
      0b00001100);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m128,1> {
  static __m128  Concat( __m128 a, __m128 b, __m128 c) {
    return _mm_blend_ps( _mm_permute_ps(a,141),_mm_permute_ps(b,216),
      0b00001100);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m128,2> {
  static __m128  Concat( __m128 a, __m128 b, __m128 c) {
    auto x = _mm_blend_ps( _mm_permute_ps(a,210),_mm_permute_ps(b,225),
      0b00001110);
    return _mm_blend_ps( x,_mm_permute_ps(c,57),0b00001000);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m128,3> {
  static __m128  Concat( __m128 a, __m128 b, __m128 c) {
    auto x = _mm_blend_ps( _mm_permute_ps(a,147),_mm_permute_ps(b,180),
      0b00001110);
    return _mm_blend_ps( x,_mm_permute_ps(c,120),0b00001000);
  }
};
template<>
struct SubsampledConcatAndCut<double,__m128d,0> {
  static __m128d  Concat( __m128d a, __m128d b) {
    return _mm_blend_pd( a,_mm_permute_pd(b,1),0b00000010);
  }
};
template<>
struct SubsampledConcatAndCut<double,__m128d,1> {
  static __m128d  Concat( __m128d a, __m128d b) {
    return _mm_blend_pd( _mm_permute_pd(a,1),b,0b00000010);
  }
};
#elif defined USE_AVX2

#elif defined USE_NEON

#endif
#endif //SUBSAMPLEDCONCATANDCUT_H
