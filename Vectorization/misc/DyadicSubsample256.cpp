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
  static __m256d load( const double* ptr ) {
    return _mm256_load_pd( ptr );
  }
  static void store( double* ptr, __m256d value) {
    _mm256_store_pd( ptr, value );
  }
  static __m256 load( const float* ptr ) {
    return _mm256_load_ps( ptr );
  }
  static void store( float* ptr, __m256 value) {
    _mm256_store_ps( ptr, value );
  }
};

template<typename T, class VecT, int SHIFT>
struct SubsampledConcatAndCut {
  static VecT  Concat( VecT a, VecT b, VecT c) {
    assert(false);
  }
};

template<>
struct SubsampledConcatAndCut<float,__m256,0> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    auto x = _mm256_blend_ps(_mm256_permute_ps(a,0b11011000),
      _mm256_permute_ps(b,0b10001101),0b11001100);
    return (__m256)_mm256_permute4x64_epi64((__m256i)x,216);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,1> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    return a;
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,2> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    return a;
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,3> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    return a;
  }
};
template<>
struct SubsampledConcatAndCut<double,__m256d,0> {
  static __m256d  Concat( __m256d a, __m256d b, __m256d c) {
    return a;
  }
};
template<>
struct SubsampledConcatAndCut<double,__m256d,1> {
  static __m256d  Concat( __m256d a, __m256d b, __m256d c) {
    return a;
  }
};

typedef  boost::alignment::aligned_allocator<int,sizeof(__m256)>
  PackAllocator;
typedef std::vector<float,PackAllocator> fvector;
typedef std::vector<double,PackAllocator> dvector;

//g++ -O3 -mavx -std=c++14 -DPARAM=0 ./DyadicSubsample128float.cpp -o ./test
//for ((i=0; i<8; i++)); do g++ -O3 -mavx -std=c++14 ./DyadicSubsample256.cpp -DPARAM=$i -o ./test; ./test >> Dyadic256.txt; done;
int main(int argc, char* argvi[]) {

  auto print = [](auto i) { std::cout<<i; };

  fvector v={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};

  auto a = MemOp::load(v.data());
  auto b = MemOp::load(v.data()+8);
  auto c = MemOp::load(v.data()+16);


  std::cout<<"-- i = "<<PARAM<<" --"<<std::endl;
  MemOp::store(v.data(),SubsampledConcatAndCut<float,__m256,PARAM>::Concat(a,b,c));
  std::for_each(v.data(),v.data()+8,print);
  std::cout << std::endl;

  dvector v2={1,2,3,4,5,6,7,8,9,10,11,12};

  auto a2 = MemOp::load(v2.data());
  auto b2 = MemOp::load(v2.data()+4);
  auto c2 = MemOp::load(v2.data()+8);

  MemOp::store(v2.data(),SubsampledConcatAndCut<
    double,__m256d,PARAM%2>::Concat(a2,b2,c2));
  std::for_each(v2.data(),v2.data()+2,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

