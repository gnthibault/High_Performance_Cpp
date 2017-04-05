//STL
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <cassert>

//Boost
#include <boost/align/aligned_allocator.hpp>

//Intel SSE intrinsics
#include "immintrin.h"

struct MemOp {
  static __m128 load( const float* ptr ) {
    return _mm_load_ps( ptr );
  }
  static void store( float* ptr, __m128 value) {
    _mm_store_ps( ptr, value );
  }
};

typedef  boost::alignment::aligned_allocator<int,sizeof(__m128)>
  PackAllocator;
typedef std::vector<float,PackAllocator> vector;

//g++ -O3 -mavx2 -std=c++14 ./main3.cpp -o ./test
//for ((i=0; i<256; i++)); do g++ -O3 -msse -std=c++14 ./permute.cpp -DSHIFT=$i -o ./test; ./test >> permute128.txt; done;
int main(int argc, char* argvi[]) {

  auto print = [](int i) { std::cout<<i; };

  vector v={1,2,3,4};

  auto a = MemOp::load(v.data());

  std::cout<<"-- i = "<<SHIFT<<" --"<<std::endl;
  MemOp::store(v.data(),_mm_permute_ps(a,SHIFT));
  std::for_each(v.data(),v.data()+4,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}


/*left = (__mm256)_mm256_permute2x128_si256((__m256i)left,
        (__m256i)right, 0);
    return (__m256)_mm256_alignr_epi8((__m256i)left,
        (__m256i)right, (unsigned int)RIGHT_SHIFT*sizeof(float));
*/
