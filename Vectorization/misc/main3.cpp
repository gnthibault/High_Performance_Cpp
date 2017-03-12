//STL
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <cassert>

//Boost
#include <boost/align/aligned_allocator.hpp>

//Intel AVX2 intrinsics
#include "immintrin.h"

template<int Begin, int End, int Val, class Enable = void>
struct ctrange { };

template<int Begin, int End, int Val>
struct ctrange<Begin, End, Val,
  typename std::enable_if<Val >= Begin && Val < End>::type> {
  using enabled = void;
};

template<int Val, class enable=void>
struct Param {
  __m256i static doIt(__m256i left, __m256i right) {
    assert(("Vectorized Shift AVX256 cannot account for shift > 256 bits",
          false));
    return left;
  }
};

template<int Val>
struct Param<Val, typename ctrange<0, 1, Val>::enabled> {
  __m256i static doIt(__m256i left, __m256i right) {
    return left;
  }
};
template<int Val>
struct Param<Val, typename ctrange<1, 4, Val>::enabled> {
  __m256i static doIt(__m256i left, __m256i right) {
    right=_mm256_permute2x128_si256(left,right,33);
    return _mm256_alignr_epi8(right,left,Val*sizeof(int));
  }
};
template<int Val>
struct Param<Val, typename ctrange<4, 5, Val>::enabled> {
  __m256i static doIt(__m256i left, __m256i right) {
    return _mm256_permute2x128_si256(left,right,33);
  }
};
template<int Val>
struct Param<Val, typename ctrange<5, 8, Val>::enabled> {
  __m256i static doIt(__m256i left, __m256i right) {
    left=_mm256_permute2x128_si256(left,right,33);
    return _mm256_alignr_epi8(right,left,(Val-4)*sizeof(int));
  }
};

template<int Val>
struct Param<Val, typename ctrange<8, 9, Val>::enabled> {
  __m256i static doIt(__m256i left, __m256i right) {
    return right;
  }
};


struct MemOp {
  static __m256i load(const __m256i* ptr ) {
    return _mm256_load_si256(ptr);
  };
  static void store(__m256i* ptr, __m256i value) {
    _mm256_store_si256(ptr, value);
  };
};

typedef  boost::alignment::aligned_allocator<int,sizeof(__m256i)>
  PackAllocator;
typedef std::vector<int,PackAllocator> vector;

//g++ -O3 -mavx2 -std=c++14 ./main3.cpp -o ./test
//for ((i=0; i<16; i++)); do g++ -O3 -mavx2 -std=c++14 ./main3.cpp -DSHIFT=$i -o ./test; ./test >> shift.txt; done;
int main(int argc, char* argvi[]) {

  auto print = [](int i) { std::cout<<i; };

  vector v={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

  auto* a = reinterpret_cast<__m256i*>(v.data());
  auto* b = a+1; 

  auto left = MemOp::load(a);
  auto right = MemOp::load(b);

  std::cout<<"-- i = "<<SHIFT<<" --"<<std::endl;
  MemOp::store(a,Param<SHIFT>::doIt(left,right));
  std::for_each(v.data(),v.data()+8,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}


/*left = (__mm256)_mm256_permute2x128_si256((__m256i)left,
        (__m256i)right, 0);
    return (__m256)_mm256_alignr_epi8((__m256i)left,
        (__m256i)right, (unsigned int)RIGHT_SHIFT*sizeof(float));
*/
