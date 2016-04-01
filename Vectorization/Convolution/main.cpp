/*
 * main.cpp
 *
 *  Created on: 16 mars 2016
 *      Author: gnthibault
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

//local
#define USE_SSE
#include "../vectorization.h"

#define SIZEX 256
#define SIZEY 256
#define KERX 1
#define KERY 1
#define NRUN 10

/*
 * This code is a simple example of how to use vectorization to perform
 * a 3*3 mean filter on an image.
 * Performance gap is generally very important due to the memory bound
 * nature of the convolution operation
 */

// Example of how to profile a code using either cachegrind or callgrind utility
// g++ ./main.cpp -std=c++11 -g -o test && valgrind --tool=cachegrind --cachegrind-out-file=out ./test
// cg_annotate out /Absolute/path/main.cpp

// or
// g++ ./main.cpp -g -o test && valgrind --tool=callgrind ./test


// Live monitoring of cpu usage, with advanced analysis tools: perf top
//perf record ./test
//perf report

//This version handles the bounds
bool PerformWorkSequentially( const std::vector<float>& vec, std::vector<float>& out )
{
	for(int j=0; j<SIZEY; j++ )
	{
		for(int i = 0; i<SIZEX; i++ )
		{
			float sum = 0;
			for(int j2 = -KERY; j2<=KERY; j2++ )
			{
				for(int i2 = -KERX; i2<=KERX; i2++ )
				{
					int idX = i+i2;
					int idY = j+j2;
					if( ( idX >= 0 ) && ( idX < SIZEX ) &&
						( idY >= 0 ) && ( idY < SIZEY ) )
					{
						out[i+j*SIZEX] += vec[idX+idY*SIZEX];
						sum = sum+1;
					}
				}
			}
			out[i+j*SIZEX] /= sum;
		}
	}
	return true;
}

//This version does not handle the bounds
bool PerformWorkVectorized( std::vector<float>& vec, std::vector<float>& out )
{
	//#pragma omp parallel for
	for(int j=1; j<SIZEY-2; j++ )
	{
		//First vector at left
		__m128 L0 = VectorizedMemOp<float,__m128>::load(&vec[(j-1)*SIZEX]);
		__m128 L1 = VectorizedMemOp<float,__m128>::load(&vec[(j)*SIZEX]);
		__m128 L2 = VectorizedMemOp<float,__m128>::load(&vec[(j+1)*SIZEX]);

		//Second column
		__m128 C0 = VectorizedMemOp<float,__m128>::load(&vec[(j-1)*SIZEX+4]);
		__m128 C1 = VectorizedMemOp<float,__m128>::load(&vec[(j)*SIZEX+4]);
		__m128 C2 = VectorizedMemOp<float,__m128>::load(&vec[(j+1)*SIZEX+4]);

		for(int i = 4; i<SIZEX; i+=4 )
		{
			//Second column
			__m128 R0 = VectorizedMemOp<float,__m128>::load(&vec[(j-1)*SIZEX+i+4]);
			__m128 R1 = VectorizedMemOp<float,__m128>::load(&vec[(j)*SIZEX+i+4]);
			__m128 R2 = VectorizedMemOp<float,__m128>::load(&vec[(j+1)*SIZEX+i+4]);

			//3 steps to compute the sum
			__m128 Sum0 = C0;
			__m128 Sum1 = C1;
			__m128 Sum2 = C2;

			//Step 1: compute left part
			Sum0 += VectorizedConcatAndCut<float,__m128,3>::Concat(L0,C0);
			Sum1 += VectorizedConcatAndCut<float,__m128,3>::Concat(L1,C1);
			Sum2 += VectorizedConcatAndCut<float,__m128,3>::Concat(L2,C2);

			//Step 2 : compute right part
			Sum0 += VectorizedConcatAndCut<float,__m128,1>::Concat(C0,R0);
			Sum1 += VectorizedConcatAndCut<float,__m128,1>::Concat(C1,R1);
			Sum2 += VectorizedConcatAndCut<float,__m128,1>::Concat(C2,R2);

			//Now divide by 3*3 and store the result
			Sum0 = Sum0+Sum1+Sum2;
			Sum0 = Sum0/9;
			VectorizedMemOp<float,__m128>::store(&out[j*SIZEX+i],Sum0);

			//At the end of the computation, we need to
			//swap the 2 first lines
			L0 = C0;
			L1 = C1;
			L2= C2;

			//Second column
			C0 = R0;
			C1 = R1;
			C2 = R2;
		}
	}
	return true;
}

//g++ ./main.cpp -o test
int main()
{
	std::vector<float> vec(SIZEX*SIZEY,1.);
	std::vector<float> out(SIZEX*SIZEY,0.);

	auto start = std::chrono::steady_clock::now();
	auto stop = std::chrono::steady_clock::now();
	auto diff = stop - start;
	double msec=std::numeric_limits<double>::max();

	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		PerformWorkSequentially(vec, out);
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count());
	}
	std::cout << "Runtime for sequential version is "<< msec << " msec "<< std::endl;
	msec=std::numeric_limits<double>::max();

	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		PerformWorkVectorized(vec, out);
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count());
	}
	std::cout << "Runtime for vectorized version is "<< msec << " msec "<< std::endl;
	msec=std::numeric_limits<double>::max();

	//Optionally check
	/*std::fill( out.begin(), out.end(), 0.);
	PerformWorkVectorized(vec, out);
	for( int j = 0; j<SIZEY; j++ )
	{
		for(int i=0; i<SIZEX; i++ )
		{
			std::cout << out[i+j*SIZEX];
		}
		std::cout << std::endl;
	}*/

	return EXIT_SUCCESS;
}
