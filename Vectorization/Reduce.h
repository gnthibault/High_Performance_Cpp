#ifndef REDUCE_H
#define REDUCE_H

//Default implementation work for non-vectorized case
template<typename T, class VecT>
class VectorSum {
 public:
  static T ReduceSum( VecT value ) {
    return value;
  }
};

#ifdef USE_AVX
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

#endif //REDUCE_H
