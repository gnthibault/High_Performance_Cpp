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
class VectorizedShift {
 public:
  //Defaulted implementation for scalar type: no shift
  constexpr static VecT LeftShift( VecT input ) {
    return SHIFT != 0 ? 0 : input;
  }
  //Defaulted implementation for scalar type: no shift
  constexpr static VecT RightShift( VecT input ) {
    return SHIFT != 0 ? 0 : input;
  }
};

template<typename T, class VecT, int RIGHT_SHIFT>
class VectorizedConcatAndCut {
 public:
  //Defaulted implementation is to rely on individual shifting
  //and add
  static VecT Concat( VecT left, VecT right ) {
    //Perform shift on both operand
    //The left one should be right shifted and right one should be left shifted
    VecT r = VectorizedShift<T,VecT,RightShiftBytes>::RightShift(left);
    VecT l = VectorizedShift<T,VecT,LeftShiftBytes>::LeftShift(right);
    //return sum of the two
    return r + l;
  }
 protected:
  constexpr static int VecSize = sizeof(VecT)/sizeof(T);
  constexpr static int RightShiftBytes = RIGHT_SHIFT*sizeof(T);
  constexpr static int LeftShiftBytes = (VecSize-RIGHT_SHIFT)*sizeof(T);
};

template<int SHIFT>
class VectorizedShift<float,__m128,SHIFT> {
 public:
  constexpr static __m128 LeftShift( __m128 input ) {
    return (__m128)_mm_slli_si128( (__m128i)input, SHIFT );
  }
  constexpr static __m128 RightShift( __m128 input ) {
    return (__m128)_mm_srli_si128( (__m128i)input, SHIFT );
  }
};

template<int SHIFT>
class VectorizedShift<double,__m128d,SHIFT> {
 public:
  constexpr static __m128d LeftShift( __m128d input ) {
    return (__m128d)_mm_slli_si128( (__m128i)input, SHIFT );
  }
  constexpr static __m128d RightShift( __m128d input ) {
    return (__m128d)_mm_srli_si128( (__m128i)input, SHIFT );
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

  fvector v={1,2,3,4,5,6,7,8};

  auto a = MemOp::load(v.data());
  auto b = MemOp::load(v.data()+4);

  std::cout<<"-- i = "<<PARAM<<" --"<<std::endl;
  MemOp::store(v.data(),VectorizedConcatAndCut<float,__m128,PARAM>::Concat(a,b));
  std::for_each(v.data(),v.data()+4,print);
  std::cout << std::endl;

  dvector v2={1,2,3,4};

  auto a2 = MemOp::load(v2.data());
  auto b2 = MemOp::load(v2.data()+2);

  MemOp::store(v2.data(),VectorizedConcatAndCut<double,__m128d,PARAM%3>::Concat(a2,b2));
  std::for_each(v2.data(),v2.data()+2,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

