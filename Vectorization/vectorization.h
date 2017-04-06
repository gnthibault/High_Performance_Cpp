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

/*
 * Include for x86 intrinsics
 * Documentation for the various intrinsics can be found on
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/
 * https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/ARM-NEON-Intrinsics.html
 */
#ifdef USE_AVX 			//compile using g++ -std=c++11 -mavx -O3
  #include "xmmintrin.h"
  #include "emmintrin.h"
#elif defined USE_AVX2 	//compile using g++ -std=c++11 -march=core-avx2 -O3
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


//Specialize Packed types when they exist
template<typename T> struct PackedType { typedef T type; };//Default packed type is... not packed

#ifdef USE_AVX
  template<> struct PackedType<float> { using type = __m128; };
  template<> struct PackedType<double> { using type = __m128d; };
#elif defined USE_AVX2
  template<> struct PackedType<float> { using type = __m256; };
  template<> struct PackedType<double> { using type = __m256d; };
#elif defined USE_NEON
  template<> struct PackedType<float> { using type = float32x4_t; };
  template<> struct PackedType<double> { using type = float64x2_t; };
#endif
template<typename T> using PackType = typename PackedType<T>::type;
template<typename T> using PackAllocator =
  boost::alignment::aligned_allocator<T,sizeof(PackType<T>)>;

#endif /* VECTORIZATION_H_ */
