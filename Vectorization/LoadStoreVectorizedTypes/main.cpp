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
#include <string.h> //memcpy

//boost
#include <boost/align/aligned_allocator.hpp>

//Local
#include "../vectorization.h"

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
	std::vector<float,boost::alignment::aligned_allocator<float> > floatVec(SIZE,1.0); //SIZE*4octets = 2Mo
	std::vector<float,boost::alignment::aligned_allocator<float> > floatDst(SIZE,0.0); //SIZE*4octets = 2Mo

	//Initialize timing tools
	auto start = std::chrono::steady_clock::now();
	auto stop = std::chrono::steady_clock::now();
	auto diff = stop-start;
	double msec=std::numeric_limits<double>::max();

	//Good old memcpy
	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		memcpy( floatDst.data(), floatVec.data(), floatDst.size()*sizeof(float));
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::micro>(diff).count());
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
	}
	std::cout << "Runtime for std::copy is "<< msec << " µsec "<< std::endl;
	msec=std::numeric_limits<double>::max();

	//Our homemade vectorized memcpy
	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		#pragma unroll
		for( int i=0; i<SIZE; i+= 128/(sizeof(float)*8) )// 128 bits sse2 / (float = 4 octets * 8 bits par octets)
		{
			store( floatDst.data()+i, load(floatVec.data()+i) );
		}
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::micro>(diff).count());
	}
	std::cout << "Runtime for vectorized handwritten version is "<< msec << " µsec "<< std::endl;
	double throughput = ((double)SIZE*sizeof(float))/(1024.0*1024.0*1024.0);
	throughput /= (msec*1e-6);
	std::cout << "Memory Throughput in this case is "<< throughput <<" GBytes/sec"<<std::endl;
	msec=std::numeric_limits<double>::max();

	return EXIT_SUCCESS;
}
