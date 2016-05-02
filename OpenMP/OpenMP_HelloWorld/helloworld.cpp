#include <omp.h>
#include <iostream>
#include <stdio.h>

//Part 1: optional and ugly conditional construction
/*
#if defined( _OPENMP )
	int thread_id()
	{
		return omp_get_thread_num();
	}
#else
	int thread_id()
	{
		return 0;
	}
#endif*/

//Compile with openmp support using
//g++ ./helloworld.cpp -o test -fopenmp

//Decide in compile commande if you want openmp support or not (avoid conditional construction like in part 1
//We do so because -fopenmp defines both a compile option, and link option
//g++ -c ./helloworld.cpp (without OMP support) OR g++ -c ./helloworld.cpp -fopenmp (with OMP support)
//g++ helloworld.o -o test -fopenmp (we still need the link flag)

//We can also modify number of threads using an environement variable
//export OMP_NUM_THREADS=2
//echo $OMP_NUM_THREADS
//Or launch with OMP_NUM_THREADS=2 ./test

int main( int argc, char* argv[])
{
	#pragma omp parallel
	{
		//int id = thread_id();
		int nt = omp_get_num_threads();	//Number of threads in the threadpool
		int id = omp_get_thread_num();	//Current Index inside the threadpool
		//std::cout << "Hello, num "<< id << " , " << nt << std::endl;
		//std::cout << "World, num "<< id << " , " << nt << std::endl;
		printf( "Hello, (%d) / (%d) ",id,nt);
		//fflush(stdout);
		printf( "World, (%d) / (%d)\n",id,nt);
	}
	
	//Then and of the curly braces defines a new openmp context, openmp is said to be transient
	//In the new context, we are no longer multithreaded
	int nt = omp_get_num_threads();
	int id = omp_get_thread_num();
	printf( "Hello, (%d) / (%d) ",id,nt);
	printf( "World, (%d) / (%d)\n",id,nt);
	return 0;
}
