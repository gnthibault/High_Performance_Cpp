/*
 * main.cpp
 *
 *  Created on: 1 avr. 2016
 *      Author: gnthibault
 */

//STL
#include <vector>


//Local
#include "../Convolution.h"


/*
 * Compile time declaration of the simple filter using full specialization
 */
template<> const float MyFilter<float,1,1>::Buf[3] = {1.0f,2.0f,3.0f};
template<> const float MyFilter<float,0,3>::Buf[4] = {1.0f,2.0f,3.0f,4.0f};
template<> const float MyFilter<float,2,2>::Buf[5] = {1.0f,2.0f,3.0f,4.0f,5.0f};
template<> const float MyFilter<float,3,3>::Buf[7] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f};

//build with g++ ./test.cpp -std=c++11 -O3 -o test -DUSE_SSE
int main(int argc, char* argv[])
{

	for( int i = 1; i<47 ; i++)
	{
		std::vector<float> input(i);
		std::vector<float> output(input.size(),0);
		std::vector<float> control(input.size(),0);

		//Fill input vector with ordered values
		std::iota(input.begin(), input.end(),1.0f);

		//Check if results for the naive and vectorized version are equal
		bool isOK = true;
		Convolution< MyFilter<float,1,1> >::Convolve( input.data(), output.data(), input.size() );
		Convolution< MyFilter<float,1,1> >::NaiveConvolve( input.data(), control.data(), 0, input.size(), input.size() );
		isOK &= std::equal(control.begin(), control.end(), output.begin() );

		//Reset values
		std::fill(output.begin(), output.end(), 0);
		std::fill(control.begin(), control.end(), 0);

		Convolution< MyFilter<float,0,3> >::Convolve( input.data(), output.data(), input.size() );
		Convolution< MyFilter<float,0,3> >::NaiveConvolve( input.data(), control.data(), 0, input.size(), input.size() );
		isOK &= std::equal(control.begin(), control.end(), output.begin() );

		//Reset values
		std::fill(output.begin(), output.end(), 0);
		std::fill(control.begin(), control.end(), 0);

		Convolution< MyFilter<float,2,2> >::Convolve( input.data(), output.data(), input.size() );
		Convolution< MyFilter<float,2,2> >::NaiveConvolve( input.data(), control.data(), 0, input.size(), input.size() );
		isOK &= std::equal(control.begin(), control.end(), output.begin() );

		//Reset values
		std::fill(output.begin(), output.end(), 0);
		std::fill(control.begin(), control.end(), 0);

		Convolution< MyFilter<float,3,3> >::Convolve( input.data(), output.data(), input.size() );
		Convolution< MyFilter<float,3,3> >::NaiveConvolve( input.data(), control.data(), 0, input.size(), input.size() );
		isOK &= std::equal(control.begin(), control.end(), output.begin() );

		//Reset values
		std::fill(output.begin(), output.end(), 0);
		std::fill(control.begin(), control.end(), 0);

		if( isOK )
		{
			std::cout << "All tests returned True Value"<<std::endl;
		}else
		{
			std::cout << " WARNING : There may be a bug "<<std::endl;
		}
	}
	return EXIT_SUCCESS;
}
