/*
 * vectorization.h
 *
 *  Created on: 16 mars 2016
 *      Author: gnthibault
 */

#ifndef VECTORIZATION_H_
#define VECTORIZATION_H_

//STL
#include <vector>

//Boost
#include <boost/align/aligned_allocator.hpp>

/*
 * Include for x86 intrinsics
 * Documentation for the various intrinsics can be found on
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/
 */
#include "xmmintrin.h"
#include "emmintrin.h"

//Store a 128bits sse2 vector of float into buffer
void store(float* ptr, __m128 value)
{
	_mm_store_ps( ptr, value );
}

//Store a 128bits sse2 vector of int into buffer
void store(int* ptr, __m128i value)
{
	_mm_store_si128 ((__m128i*) ptr, value );
}

//Load a 128bits sse2 vector of float from a buffer
__m128 load(float* ptr)
{
	return _mm_load_ps( ptr );
}

//Load a 128bits sse2 vector of int from a buffer
__m128i load(int* ptr)
{
	return _mm_load_si128((__m128i*)(ptr));
}

/*
 * Sum all 4 float elements of a 128 bits vector into 1 value
 * To do so, we use a butterfly like summation pattern
 */
float sumFloat(__m128 value)
{
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

	//Extract the right value
	return _mm_cvtss_f32 (shufl);
	//return _mm_extract_ps (shufl a, 0);
}

/*
 * Simple example of intrinsic for multiplication,
 * probably useless by the way because operator *
 * should be translated by this intrinsic by compiler
 */
__m128 mulFloat(__m128 value0, __m128 value1)
{
	return _mm_mul_ps( value0, value1 );
}

/*
 * Simple example of intrinsic for addition,
 * probably useless by the way because operator +
 * should be translated by this intrinsic by compiler
 */
__m128 addFloat(__m128 value0, __m128 value1)
{
	return _mm_add_ps( value0, value1 );
}

/*
 * Shift left operand by LS bytes towards left,
 * then shift right operand by RS bytes towards right
 * then return the sum of the two
 */
template<int LS, int RS>
__m128 shiftAdd(__m128 left, __m128 right)
{
	//Left shift
	__m128 result0 = (__m128)_mm_slli_si128( (__m128i)left, LS );
	//Right shift
	__m128 result1 = (__m128)_mm_srli_si128( (__m128i)right, RS );
	return _mm_add_ps( result0, result1 );
}
/*
// forward-declaration to allow use in Iter
template<typename T> class Sse2Vec<T>;

template<typename T>
class Sse2Iter
{
    public:
    Sse2Iter (const Sse2Vec* p_vec, int pos)
        : _pos( pos )
        , _p_vec( p_vec )
    { }

    // these three methods form the basis of an iterator for use with
    // a range-based for loop
    bool
    operator!= (const Sse2Iter& other) const
    {
        return _pos != other._pos;
    }

    // this method must be defined after the definition of Sse2Vec
    // since it needs to use it
    int operator* () const;

    const Sse2Iter& operator++ ()
    {
        ++_pos;
        // although not strictly necessary for a range-based for loop
        // following the normal convention of returning a value from
        // operator++ is a good idea.
        return *this;
    }

private:
    size_t m_pos;
    const Sse2Vec *m_vec;
};

class Sse2Vec
{
public:
    Sse2Vec()=default;

    Sse2Iter begin() const
    {
        return Sse2Iter( this, 0 );
    }

    Sse2Iter end() const
    {
        return Sse2Iter( this,  );
    }

protected:
    std::vector<T,boost::alignment::aligned_allocator<T> > m_vec;
};

template<typename T>
T Sse2Iter<T>::operator*() const
{
     return _p_vec->get( _pos );
}*/

#endif /* VECTORIZATION_H_ */
