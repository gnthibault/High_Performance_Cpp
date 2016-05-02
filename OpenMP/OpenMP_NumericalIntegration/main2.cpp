/*
 * main2.cpp
 *
 *  Created on: 15 mars 2016
 *      Author: gnthibault
 */

//STL
#include <omp.h>
#include <iostream>
#include <stdio.h>
#include <iomanip>      // std::setprecision
#include <algorithm>
#include <functional>
#include <numeric>
#include <chrono>


//Boost
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

//Compile using
//g++ ./main2.cpp -O3 -o test -fopenmp -std=c++11

//Run for instance using 4 Threads using
//OMP_NUM_THREADS=4 ./test

/*
 * This new version is way more compact than the one in main.cpp
 * it uses the keyword reduction(+:sum) to forces the compiler to
 * understand that we want a "map reduce" pattern where we define
 * the map inside the omp parallel for, and were the reduction
 * operator, that should be associative is +, and is used over
 * the variable sum
 */
#define NRUN 20

/*
 * We will use this struct in order to
 * define a more functional way to express
 * the summation
 */
template<typename T>
struct Integrator
{
	Integrator( T step ): m_step(step){};

	T operator()(int i) const
	{
		T x = (i+0.5)*m_step;
		return 4./(1.+x*x);
	}

	const T m_step;
};

int main()
{
	static const int nb_steps = 100000000;
	double pi, sum = 0.0, step = 1./nb_steps;

	//Initialize chrono helpers
	auto start = std::chrono::steady_clock::now();
	auto stop = std::chrono::steady_clock::now();
	auto diff = stop - start;
	double msec=std::numeric_limits<double>::max();	//Set mininum runtime to ridiculously high value
	Integrator<double> op(step);

	//Perform multiple run in order to get minimal runtime
	for(int k = 0; k< NRUN; k++)
	{
		sum = 0;
		start = std::chrono::steady_clock::now();

		//With this construction, no more need to use the explicit integral bound calculation !

		/*A few options are	available for parallelism: schedule( [dynamic,static,auto], [K,] )
		 * where K is the optional workload size in number of iterations.
		 * When using static, every thread is given "workload size" number of iterations every time
		 * and when it has finished, it start the next worload that should begin P iterations after
		 * the last one, where P = (Number of threads - 1) * workload size.
		 * Using dynamic is usefull when workload processing time can be data dependent, then work
		 * stealing is enabled
		 */
		#pragma omp parallel for reduction(+:sum)
		for(int i=0;i<nb_steps;i++)
		{
			sum += op(i);
		}
		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		//Compute minimum runtime
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count() );
	}

	pi = sum*step;
	std::cout << "Parallel version : Pi has value "<<std::setprecision(10)<<pi<< std::endl;
	std::cout << "Parallel version : Minimal Runtime was "<< msec << " msec "<< std::endl;
	msec=std::numeric_limits<double>::max(); //Reset min runtime to ridiculously high value

	//Perform multiple run in order to get minimal runtime
	for(int k = 0; k< NRUN; k++)
	{
		start = std::chrono::steady_clock::now();

		//Functional way to express the summation
		sum = std::accumulate( 	boost::make_transform_iterator(boost::make_counting_iterator(0), op ),
						boost::make_transform_iterator(boost::make_counting_iterator(nb_steps), op ),
						0.0, std::plus<double>() );

		stop = std::chrono::steady_clock::now();
		diff = stop - start;
		//Compute minimum runtime
		msec = std::min( msec, std::chrono::duration<double, std::milli>(diff).count() );
	}
	pi = sum*step;
	std::cout << "Sequential version : Pi has value "<<std::setprecision(10)<<pi<< std::endl;
	std::cout << "Sequential version : Minimal Runtime was "<< msec << " msec "<< std::endl;
	return EXIT_SUCCESS;
}
