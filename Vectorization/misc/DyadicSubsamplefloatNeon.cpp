//STL
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <cassert>

//Boost
#include <boost/align/aligned_allocator.hpp>

//Intel AVX intrinsics
#include "immintrin.h"

struct MemOp {
  static float32x4_t load( const float* ptr ) {
    return vld1q_f32( ptr );
  }
  static void store( float* ptr, float32x4_t value) {
    vst1q_f32( ptr, value );
  };
  static float64x2_t load( const double* ptr ) {
    return vld1q_f64( ptr );
  }
  static void store( double* ptr, float64x2_t value) {
    vst1q_f64( ptr, value );
  }
};

template<typename T, class VecT, int SHIFT>
struct SubsampledConcatAndCut {
  static VecT  Concat( VecT a, VecT b, VecT c) {
    assert(false);
  }
  static VecT  Concat( VecT a, VecT b) {
    assert(false);
  }
};

template<>
struct SubsampledConcatAndCut<float,float32x4_t,0> {
  static float32x4_t  Concat( float32x4_t a, float32x4_t b, float32x4_t c) {
    return _mm_shuffle_ps(a,b,0b10001000);
  }
};
template<>
struct SubsampledConcatAndCut<float,float32x4_t,1> {
  static float32x4_t  Concat( float32x4_t a, float32x4_t b, float32x4_t c) {
    return _mm_shuffle_ps(a,b,0b11011101);
  }
};
template<>
struct SubsampledConcatAndCut<float,float32x4_t,2> {
  static float32x4_t  Concat( float32x4_t a, float32x4_t b, float32x4_t c) {
    auto x = _mm_permute_ps(_mm_shuffle_ps(a,b,0b10000010),0b01111000);
    return _mm_blend_ps(x, _mm_permute_ps(c,57),0b00001000);
  }
};
template<>
struct SubsampledConcatAndCut<float,float32x4_t,3> {
  static float32x4_t  Concat( float32x4_t a, float32x4_t b, float32x4_t c) {
    auto x = _mm_permute_ps(_mm_shuffle_ps(a,b,0b11010011),0b01111000);
    return _mm_blend_ps(x, _mm_permute_ps(c,120),0b00001000);
  }
};
template<>
struct SubsampledConcatAndCut<double,float64x2_t,0> {
  static float64x2_t  Concat( float64x2_t a, float64x2_t b) {
    return _mm_shuffle_pd(a,b,0b00000000);
  }
};
template<>
struct SubsampledConcatAndCut<double,float64x2_t,1> {
  static float64x2_t  Concat( float64x2_t a, float64x2_t b) {
    return _mm_shuffle_pd(a,b,0b00001111);
  }
};

typedef  boost::alignment::aligned_allocator<int,sizeof(float32x4_t)>
  PackAllocator;
typedef std::vector<float,PackAllocator> fvector;
typedef std::vector<double,PackAllocator> dvector;

//g++ -O3 -mavx -std=c++14 -DPARAM=0 ./DyadicSubsample128float.cpp -o ./test
//for ((i=0; i<4; i++)); do g++ -O3 -mavx -std=c++14 ./DyadicSubsample128float.cpp -DPARAM=$i -o ./test; ./test >> Dyadicfloat128.txt; done;
int main(int argc, char* argvi[]) {

  auto print = [](auto i) { std::cout<<i; };

  fvector v={1,2,3,4,5,6,7,8,9,10,11,12};

  auto a = MemOp::load(v.data());
  auto b = MemOp::load(v.data()+4);
  auto c = MemOp::load(v.data()+8);


  std::cout<<"-- i = "<<PARAM<<" --"<<std::endl;
  MemOp::store(v.data(),SubsampledConcatAndCut<float,float32x4_t,PARAM>::Concat(a,b,c));
  std::for_each(v.data(),v.data()+4,print);
  std::cout << std::endl;

  dvector v2={1,2,3,4};

  auto a2 = MemOp::load(v2.data());
  auto b2 = MemOp::load(v2.data()+2);

  MemOp::store(v2.data(),SubsampledConcatAndCut<double,float64x2_t,PARAM%2>::Concat(a2,b2));
  std::for_each(v2.data(),v2.data()+2,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

