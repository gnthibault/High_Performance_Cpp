#include <omp.h>
#include <iostream>
#include <stdio.h>
#include <iomanip>      // std::setprecision
#include <algorithm>
#include <functional>
#include <numeric>

//Compile using
//g++ ./main.cpp -O3 -o test -fopenmp

//Run for instance using 4 Threads using
//OMP_NUM_THREADS=4 ./test

/*
 * First Naive exemple of how to use openmp
 * to perform an integral in order to compute pi.
 *
 * The function to be integrated over [0,1] is f(x) = \frac{1}{1+x^2}
 * because (arctan(x))' = f(x) which gives the relationship
 * \int_{0}^{1} \frac{1}{1+x^2}dx = arctan(1)-arctan(0)
 * and arctan(0) = 0 and arctan(1) = \frac{\pi}{4}
 *
 * Multiplying this expression by 4 gives a way to compute \pi:
 */

int main()
{
	static const int nb_steps = 100000000;
	double pi, step = 1./nb_steps;
	double sum=0;

	//We cut the integral into as many pieces as they are threads in our OMP threadpool
	//Thanks to Chasles' Relation, the sum will give us the right integral value
	//over all the domain
	#pragma omp parallel shared(sum) //Not mandatory but emphasize on the role of sum
	{
		size_t nt = omp_get_num_threads();
		size_t id = omp_get_thread_num();
		size_t wSize = nb_steps/nt;

		//Here we explicitly compute the bounds of the integral
		size_t nStart = id*wSize;
		size_t nStop = nStart+wSize;

		double lsum=0;
		for(int i=nStart;i<nStop;i++)
		{
			//We use the middle-point numerical integration strategy
			double x = (i+0.5)*step;
			lsum += 4./(1.+x*x);
		}

		//Most of the processor architecture support simple atomic operations
		//that is needed here in order to synchronize multiple threads
		//access to a variable for read/write operations
		#pragma omp atomic
		sum+=lsum;
	}
	//step is the width of the rectangle in our numerical integration stragegy
	pi = sum*step;
	std::cout << "Pi has value "<<std::setprecision(10)<<pi<< std::endl;
	return 0;
}
