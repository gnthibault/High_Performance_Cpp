/*
 * main.cpp
 *
 *  Created on: 16 mars 2016
 *      Author: gnthibault
 */


//STD
#include <cstdlib>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

//boost
#include <boost/align/aligned_allocator.hpp>

//Local
#define USE_SSE
#include "../vectorization.h"

#define NRUN 100
#define SIZE 524288
//#define SIZE 8

//g++ ./main.cpp -O3 -std=c++11 -msse2 -ffast-math -fopenmp -o test
int main( int argc, char* argv[] )
{
	std::vector<float,boost::alignment::aligned_allocator<float> > floatVec0(SIZE,2);
	std::vector<float,boost::alignment::aligned_allocator<float> > floatVec1(SIZE,2);

	for( int i = 0; i < SIZE; i++ )
	{
		floatVec0.at(i) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));;
		floatVec1.at(i) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));;
	}

	auto start = std::chrono::steady_clock::now();
	auto stop = std::chrono::steady_clock::now();
	auto diff = stop-start;
	double msec=std::numeric_limits<double>::max();

	float reference = 0;
	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		reference = std::inner_product(floatVec0.begin(), floatVec0.end(), floatVec1.begin(), 0.0f);
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count());
	}
	double refMsec = msec;
	std::cout << "Runtime for std::inner version is "<< msec << " msec "<< std::endl;
	msec=std::numeric_limits<double>::max();

	float resultat = 0;
	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		__m128 accumulator = {0.0f,0.0f,0.0f,0.0f};

		//Mixing both vectorization and thread level parallelization
		#pragma omp parallel for reduction(+:accumulator)
		for( int i=0; i<SIZE; i+= 128/(sizeof(float)*8) )// 128 bits sse2 / (float = 4 octets * 8 bits par octets)
		{
			accumulator +=	VectorizedMemOp<float,__m128>::load(floatVec0.data()+i) *
					VectorizedMemOp<float,__m128>::load(floatVec1.data()+i);
		}
		resultat = VectorSum<float,__m128>::ReduceSum( accumulator );
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count());
	}
	std::cout << "Speedup for vectorized version is "<< refMsec/msec << std::endl;
	msec=std::numeric_limits<double>::max();

	std::cout << "Resultat attendu : "<< reference << " Resultat Obtenu : "<< resultat << std::endl;

	//std::for_each( floatDst.cbegin(), floatDst.cend(), [](const float& val){std::cout<<val<<std::endl;});

	return EXIT_SUCCESS;
}
