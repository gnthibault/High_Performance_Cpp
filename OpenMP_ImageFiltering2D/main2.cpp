/*
 * main2.cpp
 *
 *  Created on: 15 mars 2016
 *      Author: gnthibault
 */

/*
 * main.cpp
 *
 *  Created on: 15 mars 2016
 *      Author: gnthibault
 */

#include <omp.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <numeric>
#include <chrono>

#define SIZEX 1024
#define SIZEY 1024
#define KERX 1
#define KERY 1
#define NRUN 100

//Compile using
//g++ ./main2.cpp -O3 -std=c++11 -fopenmp -o test

//Execute using
//OMP_NUM_THREADS=4 ./test

/*
 * This code intend to benchmark various flavour of the small image
 * processing application seen in main.cpp.
 * We compare the naive sequential version, with multiple other version
 * to see what speed up can be gained
 */

/*
 * Same code as seen in main.cpp
 */
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

/*
 * Same code as in PerformWorkSequentially, but the very first loop,
 * over the least frequently varying index (j) is parallelized
 * using openmp
 */
bool PerformWorkNaiveOMP( const std::vector<float>& vec, std::vector<float>& out )
{
	#pragma omp parallel for
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

/*
 * Same code as in PerformWorkNaiveOMP, this time, we used the keyword
 * collapse(2) to tell openmp that it can create workloads that divide both
 * the first and the second level of loop into chunks.
 * In practice, for our application, it means that the openmp workloads will
 * operate on small chunks of 2D data, instead of chunks of full lines.
 */
bool PerformWorkCollapseOMP( const std::vector<float>& vec, std::vector<float>& out )
{
	#pragma omp parallel for collapse(2)
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

/*
 * As in PerformWorkNaiveOMP, we used openmp only on the first loop,
 * but this time, we exploited the redundancy of the memory reads over
 * one line. Noticing that, for every pixels, we need 3 elements of the previous
 * column, and of the current column, we can factorize the memory reads with the previous
 * iteration, and, doing so, reduce the number of memory operations.
 * On memory bound problems, this approach generally yield good results
 *
 * It is important to notice that, for simplicity, we will no longer address
 * the bounds. But for very large image size, this should not really matters
 * from a performance point of view.
 */
bool PerformWorkCacheOMP( const std::vector<float>& vec, std::vector<float>& out )
{
	#pragma omp parallel for
	for(int j=1; j<SIZEY-1; j++ )
	{
		//First 2 columns
		float a0 = vec[(j-1)*SIZEX];
		float a1 = vec[(j)*SIZEX];
		float a2 = vec[(j+1)*SIZEX];

		//Second column
		float b0 = vec[(j-1)*SIZEX+1];
		float b1 = vec[(j)*SIZEX+1];
		float b2 = vec[(j+1)*SIZEX+1];

		for(int i = 1; i<SIZEX; i++ )
		{
			//third column
			float c0 = vec[(j-1)*SIZEX+i+1];
			float c1 = vec[(j)*SIZEX+i+1];
			float c2 = vec[(j+1)*SIZEX+i+1];

			//Computations : make a simple sum
			out[i+j*SIZEX] = (a0+a1+a2+b0+b1+b2+c0+c1+c2)/9.0f;

			//At the end of the computation, we need to
			//swap the 2 first lines
			a0 = b0;
			a1 = b1;
			a2 = b2;

			//Second column
			b0 = c0;
			b1 = c1;
			b2 = c2;
		}
	}
	return true;
}

/*
 * Same code as in PerformWorkCacheOMP, but this time, we factorized both
 * memory reads and arithmetic operations (the addition).
 * Doing so should yield even better results, but only if the problem is
 * compute bound for this architecture ...
 */
bool PerformWorkCache2OMP( const std::vector<float>& vec, std::vector<float>& out )
{
	#pragma omp parallel for
	for(int j=1; j<SIZEY-1; j++ )
	{
		//First 2 columns
		float a0 = vec[(j-1)*SIZEX]+vec[(j)*SIZEX]+vec[(j+1)*SIZEX];

		//Second column
		float b0 = vec[(j-1)*SIZEX+1]+vec[(j)*SIZEX+1]+vec[(j+1)*SIZEX+1];

		for(int i = 1; i<SIZEX; i++ )
		{
			//third column
			float c0 = vec[(j-1)*SIZEX+i+1]+vec[(j)*SIZEX+i+1]+vec[(j+1)*SIZEX+i+1];

			//Computations : make a simple sum
			out[i+j*SIZEX] = (a0+b0+c0)/9.0f;

			//At the end of the computation, we need to
			//swap the 2 first lines
			a0 = b0;

			//Second column
			b0 = c0;
		}
	}
	return true;
}

int main()
{
	std::vector<float> vec(SIZEX*SIZEY,1.);
	std::vector<float> out(SIZEX*SIZEY,0.);

	auto start = std::chrono::steady_clock::now();
	auto stop = std::chrono::steady_clock::now();
	auto diff = stop - start;
	double msec = 0;

	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		PerformWorkSequentially(vec, out);
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec += std::chrono::duration<double, std::milli>(diff).count();
		//Reset to zero before the next run
		std::fill( out.begin(), out.end(), 0.);
	}
	double seqMsec = msec/(double)NRUN;
	std::cout << "Runtime for sequential version is "<< seqMsec << " msec "<< std::endl;
	msec = 0;

	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		PerformWorkNaiveOMP(vec, out);
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec += std::chrono::duration<double, std::milli>(diff).count();
		//Reset to zero before the next run
		std::fill( out.begin(), out.end(), 0.);
	}
	double Nsec = msec/(double)NRUN;
	std::cout << "Acceleration for Naive OMP  is "<< seqMsec/Nsec << std::endl;
	msec = 0;

	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		PerformWorkCollapseOMP(vec, out);
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec += std::chrono::duration<double, std::milli>(diff).count();
		//Reset to zero before the next run
		std::fill( out.begin(), out.end(), 0.);
	}
	Nsec = msec/(double)NRUN;
	std::cout << "Acceleration for collapse OMP  is "<< seqMsec/Nsec << std::endl;
	msec = 0;

	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		PerformWorkCacheOMP(vec, out);
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec += std::chrono::duration<double, std::milli>(diff).count();
		//Reset to zero before the next run
		std::fill( out.begin(), out.end(), 0.);
	}
	Nsec = msec/(double)NRUN;
	std::cout << "Acceleration for Cache OMP  is "<< seqMsec/Nsec << std::endl;
	msec = 0;

	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();
		PerformWorkCache2OMP(vec, out);
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		msec += std::chrono::duration<double, std::milli>(diff).count();
		//Reset to zero before the next run
		std::fill( out.begin(), out.end(), 0.);
	}
	Nsec = msec/(double)NRUN;
	std::cout << "Acceleration for Cache 2 OMP  is "<< seqMsec/Nsec << std::endl;
	msec = 0;

	//Optionally check
	/*PerformWorkCache2OMP(vec, out);
	for( int j = 0; j<SIZEY; j++ )
	{
		for(int i=0; i<SIZEX; i++ )
		{
			std::cout << out[i+j*SIZEX];
		}
		std::cout << std::endl;
	}*/

	return 0;
}
