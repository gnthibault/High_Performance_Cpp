/*
 * vectorization.h
 *
 *  Created on: 16 mars 2016
 *      Author: gnthibault
 */

#ifndef VECTORIZATION_H_
#define VECTORIZATION_H_

// STL
#include <cassert>
#include <exception>
#include <vector>

// Boost
#include <boost/align/aligned_allocator.hpp>

// Local
#include "MetaHelper.h"

/*
 * Include for x86 intrinsics
 * Documentation for the various intrinsics can be found on
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/
 * https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/ARM-NEON-Intrinsics.html
 */
#ifdef USE_SSE 			//compile using g++ -std=c++11 -msse2 -O3
  #include "xmmintrin.h"
  #include "emmintrin.h"
#elif defined USE_AVX 	//compile using g++ -std=c++11 -march=core-avx2 -O3
  #include "immintrin.h"
#elif defined USE_AVX512 //compile using g++ -std=c++11 -mfma -mavx512f -O3 or
                         // -march=knl or -march=skylake-avx512
  #include "zmmintrin.h"
#elif defined USE_NEON 	//compile using g++-arm-linux-gnu.x86_64 -std=c++11 -mfpu=neon -O3
  #include <arm_neon.h>
#endif

/**
Check vectorization with:
gcc -g -c test.c
objdump -d -M intel -S test.o */


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

#ifdef USE_SSE
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
#elif defined USE_AVX
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
#endif

//Default implementation work for non-vectorized case
template<typename T, class VecT>
class VectorSum {
 public:
  static T ReduceSum( VecT value ) {
    return value;
  }
};

#ifdef USE_SSE
/*
 * Used in the dot product example
 * Sum all 4 float elements of a 128 bits vector into 1 value
 * To do so, we use a butterfly like summation pattern
 */
template<>
class VectorSum<float,__m128> {
 public:
  static float ReduceSum( __m128 value ) {
    /*
     * Shuffle the input vector such that we have 1,0,3,2
     * This is equivalent to a pairwise swap where the first
     * two elements are swapped with the next two
     */
    __m128 shufl = _mm_shuffle_ps(value,value, _MM_SHUFFLE(1,0,3,2));

    //Sum both values
    shufl = _mm_add_ps(value, shufl);
    //shufl = |3|2|1|0| + |1|0|3|2| = |3+1|2+0|1+3|0+2|

    /*
     * Second shuffle 2,3,0,1
     * This is equivalent to 1 by 1 swap between every
     * two neighboring elements from the first swap
     */
    __m128 shufl2 = _mm_shuffle_ps(shufl,shufl, _MM_SHUFFLE(2,3,0,1));
    //shufl2 = |2+0|3+1|0+2|1+3|


    //Sum both values
    shufl = _mm_add_ps(shufl, shufl2);
    //shufl = |3+1|2+0|1+3|0+2| + |2+0|3+1|0+2|1+3|

    //Copy the lower single-precision (32-bit) floating-point element of a to dst.
    return _mm_cvtss_f32( shufl );
    //We also could have used to extract the 0th element:
    //return _mm_extract_ps (shufl a, 0);
  }
};
#endif

//Perform left and right shift
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

/*
 * Concatenate two vectors that are given as input
 * then right shift the results of RIGHT_SHIFT elements
 */
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

#ifdef USE_SSE
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
#elif defined USE_AVX
template<int Val, class enable=void>
struct AVX256ConcatandCut {
  static __m256 Concat(__m256 left, __m256 right) {
    assert(("Vectorized Shift AVX256 cannot account for shift > 256 bits",
          false));
    return left;
  }
};

template<int Val>
struct AVX256ConcatandCut<Val, typename ctrange<0, 1, Val>::enabled> {
  static __m256 Concat(__m256 left, __m256 right) {
    return left;
  }
};
template<int Val>
struct AVX256ConcatandCut<Val, typename ctrange<1, 4, Val>::enabled> {
  static __m256 Concat(__m256 left, __m256 right) {
    right=(__m256)_mm256_permute2x128_si256((__m256i)left,(__m256i)right,33);
    return (__m256)_mm256_alignr_epi8((__m256i)right,(__m256i)left,
        Val*sizeof(int));
  }
};
template<int Val>
struct AVX256ConcatandCut<Val, typename ctrange<4, 5, Val>::enabled> {
  static __m256 Concat(__m256 left, __m256 right) {
    return (__m256)_mm256_permute2x128_si256((__m256i)left,(__m256i)right,33);
  }
};
template<int Val>
struct AVX256ConcatandCut<Val, typename ctrange<5, 8, Val>::enabled> {
  static __m256 Concat(__m256 left, __m256 right) {
    left=(__m256)_mm256_permute2x128_si256((__m256i)left,(__m256i)right,33);
    return (__m256)_mm256_alignr_epi8((__m256i)right,(__m256i)left,
       (Val-4)*sizeof(int));
  }
};

template<int Val>
struct AVX256ConcatandCut<Val, typename ctrange<8, 9, Val>::enabled> {
  __m256 static Concat(__m256 left, __m256 right) {
    return right;
  }
};

template<int RIGHT_SHIFT>
class VectorizedConcatAndCut<float,__m256,RIGHT_SHIFT> {
 public:
  //Optimized specific intrinsic for concat / shift / cut in AVX
  static __m256 Concat( __m256 left, __m256 right ) {
    return AVX256ConcatandCut<RIGHT_SHIFT>::Concat(left,right);
  }
};
#elif defined USE_NEON
template<int RIGHT_SHIFT>
class VectorizedConcatAndCut<float,float32x4_t,RIGHT_SHIFT> {
 public:
  //Optimized specific intrinsic for concat / shift / cut in AVX
  static float32x4_t Concat( float32x4_t left, float32x4_t right ) {
    return vextq_f32( left, right, RIGHT_SHIFT) ;
  }
};
#endif


// forward-declaration to allow use in Iter
template<typename T> class SimdVec;

//Specialize Packed types when they exist
template<typename T> struct PackedType { typedef T type; };//Default packed type is... not packed

#ifdef USE_SSE
	template<> struct PackedType<float> { using type = __m128; };
#elif defined USE_AVX
	template<> struct PackedType<float> { using type = __m256; };
#elif defined USE_NEON
	template<> struct PackedType<float> { using type = float32x4_t; };
#endif
template<typename T> using PackType = typename PackedType<T>::type;
template<typename T> using PackAllocator =
  boost::alignment::aligned_allocator<T,sizeof(PackType<T>)>;

#endif /* VECTORIZATION_H_ */
