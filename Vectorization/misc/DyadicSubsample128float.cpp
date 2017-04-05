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
  static __m128d load( const double* ptr ) {
    return _mm_load_pd( ptr );
  }
  static void store( double* ptr, __m128d value) {
    _mm_store_pd( ptr, value );
  }
  static __m128 load( const float* ptr ) {
    return _mm_load_ps( ptr );
  }
  static void store( float* ptr, __m128 value) {
    _mm_store_ps( ptr, value );
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

typedef  boost::alignment::aligned_allocator<int,sizeof(__m128)>
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
  MemOp::store(v.data(),SubsampledConcatAndCut<float,__m128,PARAM>::Concat(a,b,c));
  std::for_each(v.data(),v.data()+4,print);
  std::cout << std::endl;

  dvector v2={1,2,3,4};

  auto a2 = MemOp::load(v2.data());
  auto b2 = MemOp::load(v2.data()+2);

  MemOp::store(v2.data(),SubsampledConcatAndCut<double,__m128d,PARAM%2>::Concat(a2,b2));
  std::for_each(v2.data(),v2.data()+2,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

