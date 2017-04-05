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
  static __m128d load( const double* ptr ) {
    return _mm_load_pd( ptr );
  }
  static void store( double* ptr, __m128d value) {
    _mm_store_pd( ptr, value );
  }
};

typedef  boost::alignment::aligned_allocator<int,sizeof(__m128)>
  PackAllocator;
typedef std::vector<double,PackAllocator> vector;

//g++ -O3 -mavx2 -std=c++14 ./main3.cpp -o ./test
//for ((i=0; i<256; i++)); do g++ -O3 -mavx -std=c++14 ./permuted128.cpp -DSHIFT=$i -o ./test; ./test >> permuted128.txt; done;
int main(int argc, char* argvi[]) {

  auto print = [](int i) { std::cout<<i; };

  vector v={1,2};

  auto a = MemOp::load(v.data());

  std::cout<<"-- i = "<<SHIFT<<" --"<<std::endl;
  MemOp::store(v.data(),_mm_permute_pd(a,SHIFT));
  std::for_each(v.data(),v.data()+2,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}


/*left = (__mm256)_mm256_permute2x128_si256((__m256i)left,
        (__m256i)right, 0);
    return (__m256)_mm256_alignr_epi8((__m256i)left,
        (__m256i)right, (unsigned int)RIGHT_SHIFT*sizeof(float));
*/
