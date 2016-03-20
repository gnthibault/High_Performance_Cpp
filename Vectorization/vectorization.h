/*
 * vectorization.h
 *
 *  Created on: 16 mars 2016
 *      Author: gnthibault
 */

#ifndef VECTORIZATION_H_
#define VECTORIZATION_H_

//Include de la mort
#include "xmmintrin.h"
#include "emmintrin.h"

void store(float* ptr, __m128 value)
{
	_mm_store_ps( ptr, value );
}

void store(int* ptr, __m128i value)
{
	_mm_store_si128 ((__m128i*) ptr, value );
}

__m128 load(float* ptr)
{
	return _mm_load_ps( ptr );
}

__m128i load(int* ptr)
{
	return _mm_load_si128((__m128i*)(ptr));
}

float sumFloat(__m128 value)
{
	// Shuffle the input vector such that we have 1,0,3,2
	__m128 shufl = _mm_shuffle_ps(value,value, _MM_SHUFFLE(1,0,3,2));

	//Sum both values
	shufl = _mm_add_ps(value, shufl);

	//Second shuffle 2,3,0,1
	__m128 shufl2 = _mm_shuffle_ps(shufl,shufl, _MM_SHUFFLE(2,3,0,1));

	//Sum both values
	shufl = _mm_add_ps(shufl, shufl2);

	//Extract the right value
	return _mm_cvtss_f32 (shufl);
	//return _mm_extract_ps (shufl a, 0);
}

__m128 mulFloat(__m128 value0, __m128 value1)
{
	return _mm_mul_ps( value0, value1 );
}

__m128 addFloat(__m128 value0, __m128 value1)
{
	return _mm_add_ps( value0, value1 );
}

template<int LS, int RS>
__m128 shiftAdd(__m128 left, __m128 right)
{
	//Left shift
	__m128 result0 = (__m128)_mm_slli_si128( (__m128i)left, LS );
	//Right shift
	__m128 result1 = (__m128)_mm_srli_si128( (__m128i)right, RS );
	return _mm_add_ps( result0, result1 );
}

#endif /* VECTORIZATION_H_ */
