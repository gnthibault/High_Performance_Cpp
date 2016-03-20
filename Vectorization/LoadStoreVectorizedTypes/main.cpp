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
//#define SIZE 524288
#define SIZE 1048576

//g++ ./main.cpp -std=c++11 -O3 -msse2 -o test
int main( int argc, char* argv[] )
{
	std::vector<float,boost::alignment::aligned_allocator<float> > floatVec(SIZE,1.0); //SIZE*4octets = 2Mo
	std::vector<float,boost::alignment::aligned_allocator<float> > floatDst(SIZE,0.0); //SIZE*4octets = 2Mo

	//std::vector<int> intVec(SIZE); //SIZE*4octets = 2Mo
	//std::vector<int> intDst(SIZE); //SIZE*4octets = 2Mo


	auto start = std::chrono::steady_clock::now();
	auto stop = std::chrono::steady_clock::now();
	auto diff = stop-start;
	double msec=std::numeric_limits<double>::max();

	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		memcpy( floatDst.data(), floatVec.data(), floatDst.size()*sizeof(float));
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count());
	}
	std::cout << "Runtime for memcpy version is "<< msec << " msec "<< std::endl;
	msec=std::numeric_limits<double>::max();


	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		std::copy( floatVec.cbegin(), floatVec.cend(), floatDst.begin() );
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count());
	}
	std::cout << "Runtime for std::copy is "<< msec << " msec "<< std::endl;
	msec=std::numeric_limits<double>::max();

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
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count());
	}
	std::cout << "Runtime for vectorized handwritten version is "<< msec << " msec "<< std::endl;
	double debit = 1000.0*((double)SIZE*sizeof(float))/(1024.0*1024.0*1024.0);
	std::cout << "Pass Band is "<< debit <<" Goctets/sec"<<std::endl;
	msec=std::numeric_limits<double>::max();

	return EXIT_SUCCESS;
}
