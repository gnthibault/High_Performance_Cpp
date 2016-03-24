/*
 * main.cpp
 *
 *  Created on: 15 mars 2016
 *      Author: gnthibault
 */

#include <iostream>
#include <algorithm>
#include <functional>
#include <numeric>

#define SIZEX 9
#define SIZEY 9
#define KERX 1
#define KERY 1

//Compile code using
//g++ ./main.cpp -O3 -std=c++11 -o test

/*
 * This code is an extremly simple example of image processing
 * where the input image is assumed to be a 2D vector full of 1's
 * Then a mean filter of size 3*3 is applied over the whole
 * image, the output should then be full of 1.
 *
 * In this first example, the bounds of the image are taken into
 * account
 */

int main()
{
	std::vector<float> vec(SIZEX*SIZEY,1.f);
	std::vector<float> out(SIZEX*SIZEY,0.f);

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

	//Print for initial debug
	for( int j = 0; j<SIZEY; j++ )
	{
		for(int i=0; i<SIZEX; i++ )
		{
			std::cout << out[i+j*SIZEX];
		}
		std::cout << std::endl;
	}

	//Check if output is full of 1's
	bool isOK = std::all_of(out.cbegin(),out.cend(),[](float in){return in == 1.f;});
	std::cout << " Is result OK ? "<< isOK << std::endl;


	return 0;
}
