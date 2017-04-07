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

template<int Begin, int End, int Val, class Enable = void>
struct ctrange { };

template<int Begin, int End, int Val>
struct ctrange<Begin, End, Val,
  typename std::enable_if<Val >= Begin && Val < End>::type> {
  using enabled = void;
};

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

template<typename T, typename vecT, int Val, class enable=void>
struct AVX256ConcatandCut {
  static vecT Concat(vecT left, vecT right) {
    assert(("Vectorized Shift AVX256 cannot account for shift > 256 bits",
          false));
    return left;
  }
};

template<int Val>
struct AVX256ConcatandCut<float, __m256, Val,
    typename ctrange<0, 1, Val>::enabled> {
  static __m256 Concat(__m256 left, __m256 right) {
    return left;
  }
};
template<int Val>
struct AVX256ConcatandCut<float, __m256, Val,
    typename ctrange<1, 4, Val>::enabled> {
  static __m256 Concat(__m256 left, __m256 right) {
    return (__m256)_mm256_alignr_epi8(
      (__m256i)_mm256_permute2f128_ps(left,right,33),
      (__m256i)left, Val*sizeof(float));
  }
};
template<int Val>
struct AVX256ConcatandCut<float, __m256, Val,
    typename ctrange<4, 5, Val>::enabled> {
  static __m256 Concat(__m256 left, __m256 right) {
    return _mm256_permute2f128_ps(left,right,33);
  }
};
template<int Val>
struct AVX256ConcatandCut<float, __m256, Val,
    typename ctrange<5, 8, Val>::enabled> {
  static __m256 Concat(__m256 left, __m256 right) {
    return (__m256)_mm256_alignr_epi8((__m256i)right,
      (__m256i)_mm256_permute2f128_ps(left,right,33), (Val-4)*sizeof(float));
  }
};
template<int Val>
struct AVX256ConcatandCut<float, __m256, Val,
    typename ctrange<8, 9, Val>::enabled> {
  __m256 static Concat(__m256 left, __m256 right) {
    return right;
  }
};

template<typename T, class VecT, int RIGHT_SHIFT>
class VectorizedConcatAndCut {
 public:
  static VecT Concat( VecT left, VecT right ) {
    return left;
  }
};
template<int Val>
struct AVX256ConcatandCut<double, __m256d, Val,
    typename ctrange<0, 1, Val>::enabled> {
  static __m256d Concat(__m256d left, __m256d right) {
    return left;
  }
};
template<int Val>
struct AVX256ConcatandCut<double, __m256d, Val,
    typename ctrange<1, 2, Val>::enabled> {
  static __m256d Concat(__m256d left, __m256d right) {
    return (__m256d)_mm256_alignr_epi8(
      (__m256i)_mm256_permute2f128_pd(left,right,33),
      (__m256i)left, Val*sizeof(double));
  }
};
template<int Val>
struct AVX256ConcatandCut<double, __m256d, Val,
    typename ctrange<2, 3, Val>::enabled> {
  static __m256d Concat(__m256d left, __m256d right) {
    return _mm256_permute2f128_pd(left,right,33);
  }
};
template<int Val>
struct AVX256ConcatandCut<double, __m256d, Val,
    typename ctrange<3, 4, Val>::enabled> {
  static __m256d Concat(__m256d left, __m256d right) {
    return (__m256d)_mm256_alignr_epi8((__m256i)right,
      (__m256i)_mm256_permute2f128_pd(left,right,33), (Val-2)*sizeof(double));
  }
};
template<int Val>
struct AVX256ConcatandCut<double, __m256d, Val,
    typename ctrange<4, 5, Val>::enabled> {
  static __m256d Concat(__m256d left, __m256d right) {
    return right;
  }
};

template<int RIGHT_SHIFT>
class VectorizedConcatAndCut<float,__m256,RIGHT_SHIFT> {
 public:
  //Optimized specific intrinsic for concat / shift / cut in AVX
  static __m256 Concat( __m256 left, __m256 right ) {
    return AVX256ConcatandCut<float,__m256,RIGHT_SHIFT>::Concat(left,right);
  }
};
template<int RIGHT_SHIFT>
class VectorizedConcatAndCut<double,__m256d,RIGHT_SHIFT> {
 public:
  //Optimized specific intrinsic for concat / shift / cut in AVX
  static __m256d Concat( __m256d left, __m256d right ) {
    return AVX256ConcatandCut<double,__m256d,RIGHT_SHIFT>::Concat(left,right);
  }
};

/*template<>
struct SubsampledConcatAndCut<float,__m256,0> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    return (__m256)_mm256_permute4x64_epi64((__m256i)
      _mm256_shuffle_ps(a,b,0b10001000),216);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,1> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    return (__m256)_mm256_permute4x64_epi64((__m256i)
      _mm256_shuffle_ps(a,b,0b11011101),216);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,2> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    a=_mm256_permutevar8x32_ps(a,
      _mm256_set_epi32(0,0,0,0,0,6,4,2));
    b=_mm256_permutevar8x32_ps(b,
      _mm256_set_epi32(0,6,4,2,0,0,0,0));
    a=_mm256_blend_ps(a,b,0b01111000);
    return _mm256_blend_ps(a,_mm256_permutevar8x32_ps(c,
      _mm256_set_epi32(0,0,0,0,0,0,0,0)),0b10000000);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,3> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    a=_mm256_permutevar8x32_ps(a,
      _mm256_set_epi32(0,0,0,0,0,7,5,3));
    b=_mm256_permutevar8x32_ps(b,
      _mm256_set_epi32(0,7,5,3,1,0,0,0));
    a=_mm256_blend_ps(a,b,0b01111000);
    return _mm256_blend_ps(a,_mm256_permutevar8x32_ps(c,
      _mm256_set_epi32(1,0,0,0,0,0,0,0)),0b10000000);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,4> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    auto x = (__m256) _mm256_permute2x128_si256(
      (__m256i)_mm256_permute_ps(a,0b11011000),
      (__m256i)_mm256_permute_ps(c,0b10001101),97);
    auto y = (__m256)_mm256_permute4x64_epi64((__m256i)
      _mm256_permute_ps(b,0b10001101),180);
    return _mm256_blend_ps(x,y,0b00111100);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,5> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    auto x = (__m256) _mm256_permute2x128_si256(
      (__m256i)_mm256_permute_ps(a,0b10001101),
      (__m256i)_mm256_permute_ps(c,0b11011000),97);
    auto y = (__m256)_mm256_permute4x64_epi64((__m256i)
      _mm256_permute_ps(b,0b11011000),180);
    return _mm256_blend_ps(x,y,0b00111100);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,6> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    a=_mm256_permutevar8x32_ps(a,
      _mm256_set_epi32(0,0,0,0,0,0,0,6));
    b=_mm256_permutevar8x32_ps(b,
      _mm256_set_epi32(0,0,0,6,4,2,0,0));
    a=_mm256_blend_ps(a,b,0b00011110);
    return _mm256_blend_ps(a,_mm256_permutevar8x32_ps(c,
      _mm256_set_epi32(4,2,0,0,0,0,0,0)),0b11100000);
  }
};
template<>
struct SubsampledConcatAndCut<float,__m256,7> {
  static __m256  Concat( __m256 a, __m256 b, __m256 c) {
    a=_mm256_permutevar8x32_ps(a,
      _mm256_set_epi32(0,0,0,0,0,0,0,7));
    b=_mm256_permutevar8x32_ps(b,
      _mm256_set_epi32(0,0,0,7,5,3,1,0));
    a=_mm256_blend_ps(a,b,0b00011110);
    return _mm256_blend_ps(a,_mm256_permutevar8x32_ps(c,
      _mm256_set_epi32(5,3,1,0,0,0,0,0)),0b11100000);
  }
};
template<>
struct SubsampledConcatAndCut<double,__m256d,0> {
  static __m256d  Concat( __m256d a, __m256d b, __m256d c) {
    return (__m256d) _mm256_permute2x128_si256(
      _mm256_permute4x64_epi64((__m256i)a,216),
      _mm256_permute4x64_epi64((__m256i)b,141),48);
  }
};
template<>
struct SubsampledConcatAndCut<double,__m256d,1> {
  static __m256d  Concat( __m256d a, __m256d b, __m256d c) {
    return (__m256d) _mm256_permute2x128_si256(
      _mm256_permute4x64_epi64((__m256i)a,141),
      _mm256_permute4x64_epi64((__m256i)b,216),48);
  }
};
template<>
struct SubsampledConcatAndCut<double,__m256d,2> {
  static __m256d  Concat( __m256d a, __m256d b, __m256d c) {
    auto x = _mm256_permute2x128_si256((__m256i)a,(__m256i)c,97);
    return _mm256_blend_pd((__m256d)_mm256_permute4x64_epi64(x,180),
      (__m256d)_mm256_permute4x64_epi64((__m256i)b,225),0b0110);
  }
};
template<>
struct SubsampledConcatAndCut<double,__m256d,3> {
  static __m256d  Concat( __m256d a, __m256d b, __m256d c) {
    auto x = _mm256_permute2x128_si256((__m256i)a,(__m256i)c,97);
    return _mm256_blend_pd((__m256d)_mm256_permute4x64_epi64(x,225),
      (__m256d)_mm256_permute4x64_epi64((__m256i)b,180),0b0110);
  }
};*/

typedef  boost::alignment::aligned_allocator<int,sizeof(__m256)>
  PackAllocator;
typedef std::vector<float,PackAllocator> fvector;
typedef std::vector<double,PackAllocator> dvector;

//g++ -O3 -mavx -std=c++14 -DPARAM=0 ./DyadicSubsample128float.cpp -o ./test
//for ((i=0; i<8; i++)); do g++ -O3 -mavx -std=c++14 ./DyadicSubsample256.cpp -DPARAM=$i -o ./test; ./test >> Dyadic256.txt; done;
int main(int argc, char* argvi[]) {

  auto print = [](auto i) { std::cout<<i; };

  fvector v={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

  auto a = MemOp::load(v.data());
  auto b = MemOp::load(v.data()+8);

  std::cout<<"-- i = "<<PARAM<<" --"<<std::endl;
  MemOp::store(v.data(),VectorizedConcatAndCut<float,__m256,PARAM>::Concat(a,b));
  std::for_each(v.data(),v.data()+8,print);
  std::cout << std::endl;

  dvector v2={1,2,3,4,5,6,7,8};

  auto a2 = MemOp::load(v2.data());
  auto b2 = MemOp::load(v2.data()+4);

  MemOp::store(v2.data(),VectorizedConcatAndCut<double,__m256d,PARAM%5>::Concat(a2,b2));
  std::for_each(v2.data(),v2.data()+4,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

