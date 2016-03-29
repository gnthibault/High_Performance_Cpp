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
#include <string.h> //memcpy

//boost
#include <boost/align/aligned_allocator.hpp>

//Local
#define USE_SSE
#include "../vectorization.h"
#include "SimdVec.h"

#define NRUN 100

/*
 * Choose size such that size*4 bytes * 2 < processor cache size
 * so that 2 vector of float(4bytes) can fit inside the cache
 */
#define SIZE 262144

/*
 * This code perform no computation, instead, it shows how to load and store packs of data, here
 * we used the single floating point type and the SSE2 vectorized instructions
 */

//g++ ./main.cpp -std=c++11 -O3 -msse2 -o test
int main( int argc, char* argv[] )
{
	/*
	 * In order to be able to use vectorization, one should first ensure that memory is aligned, because
	 * load and store of pack of data from and to vectorized registers is impossible for unaligned adresses
	 * and may result in segfault (or not if you are lucky and your operating system always gives you aligned
	 * memory.
	 */
	std::vector<float,boost::alignment::aligned_allocator<float> > floatVec(SIZE,1.f); //SIZE*4octets = 2Mo
	std::vector<float,boost::alignment::aligned_allocator<float> > floatDst(SIZE,0.f); //SIZE*4octets = 2Mo

	//Initialize timing tools
	auto start = std::chrono::steady_clock::now();
	auto stop = std::chrono::steady_clock::now();
	auto diff = stop-start;
	double msec=std::numeric_limits<double>::max();

	//Initialize verification tool
	bool isOK = true;

	//Good old memcpy
	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		memcpy( floatDst.data(), floatVec.data(), floatDst.size()*sizeof(float));
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::micro>(diff).count());
		//Check if copy was performed
		isOK &= std::all_of(floatDst.cbegin(),floatDst.cend(),[](float in){return in == 1.f;});
		//Reset vector to 0 value
		std::fill( floatDst.begin(), floatDst.end(), 0.f);
	}
	std::cout << "Runtime for memcpy version is "<< msec << " µsec "<< std::endl;
	msec=std::numeric_limits<double>::max();

	//The more versatile std::copy
	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		std::copy( floatVec.cbegin(), floatVec.cend(), floatDst.begin() );
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::micro>(diff).count());
		isOK &= std::all_of(floatDst.cbegin(),floatDst.cend(),[](float in){return in == 1.f;});
		std::fill( floatDst.begin(), floatDst.end(), 0.f);
	}
	std::cout << "Runtime for std::copy is "<< msec << " µsec "<< std::endl;
	msec=std::numeric_limits<double>::max();

	//Our homemade vectorized memcpy
	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		#pragma unroll
		for( int i=0; i<SIZE; i+= 128/(sizeof(float)*8) )// 128bits sse2/(1float=4bytes*8bits)
		{
			store( floatDst.data()+i, load(floatVec.data()+i) );
		}
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::micro>(diff).count());
		isOK &= std::all_of(floatDst.cbegin(),floatDst.cend(),[](float in){return in == 1.f;});
		std::fill( floatDst.begin(), floatDst.end(), 0.f);
	}
	std::cout << "Runtime for vectorized handwritten version is "<< msec << " µsec "<< std::endl;
	double throughput = ((double)SIZE*sizeof(float))/(1024.0*1024.0*1024.0);
	throughput /= (msec*1e-6);
	std::cout << "Memory Throughput in this case is "<< throughput <<" GBytes/sec"<<std::endl;
	msec=std::numeric_limits<double>::max();


	//A more functional way to do this copy, using a handwritten container
	Sse2Vec<float> sse2VecSrc( SIZE, 1 );
	Sse2Vec<float> sse2VecDst( SIZE, 0 );
	for(int k = 0; k< NRUN; k++)
	{
		auto dst = sse2VecDst.begin();
		start = std::chrono::steady_clock::now();
		for( const auto src : sse2VecSrc )
		{
			dst.set( src );
			dst++;
		}
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::micro>(diff).count());
		//Check if copy went well
		isOK &= std::all_of(sse2VecDst.cscalarbegin(), sse2VecDst.cscalarend(),
				[](float in){return in == 1.f;} );
		//Reset inner vector to 0 value
		std::fill( sse2VecDst.scalarbegin(), sse2VecDst.scalarend(), 0.f);
	}
	std::cout << "Runtime for vectorized functional version is "<< msec << " µsec "<< std::endl;

	if(isOK)
	{
		std::cout << "All copy were correctly performed" << std::endl;
	}else
	{
		std::cout << "WARNING: not all copy were OK !" << std::endl;
	}


	return EXIT_SUCCESS;
}
