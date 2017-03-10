//STL
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
//Boost
#include <boost/align/aligned_allocator.hpp>

//Intel AVX2 intrinsics
#include "immintrin.h"

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

//g++ -O3 -mavx2 -std=c++14 ./main.cpp -o ./test
//for ((i=0; i<16; i++)); do g++ -O3 -mavx2 -std=c++14 ./main.cpp -DSHIFT=$i -o ./test; ./test >> permute.txt; done;
int main(int argc, char* argvi[]) {

  auto print = [](int i) { std::cout<<i; };

  vector v={0,0,0,0,1,1,1,1,
    2,2,2,2,3,3,3,3};

  //std::for_each(v.cbegin(),v.cend(),print);
  //std::cout << "Now, let's tranform that"<<std::endl;
  auto* a = reinterpret_cast<__m256i*>(v.data());
  auto* b = a+1; 

  auto left = MemOp::load(a);
  auto right = MemOp::load(b);

  std::cout<<"-- i = "<<SHIFT<<" --"<<std::endl;
  MemOp::store(a,_mm256_permute2x128_si256(left,right,SHIFT));
  std::for_each(v.data(),v.data()+16,print);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}


/*left = (__mm256)_mm256_permute2x128_si256((__m256i)left,
        (__m256i)right, 0);
    return (__m256)_mm256_alignr_epi8((__m256i)left,
        (__m256i)right, (unsigned int)RIGHT_SHIFT*sizeof(float));
*/
