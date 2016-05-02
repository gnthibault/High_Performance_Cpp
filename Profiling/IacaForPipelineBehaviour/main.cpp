#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>

//Inteal Analasing
#include "/opt/iaca-lin64/include/iacaMarks.h"

//Compile using
//g++ ./main.cpp -o test


/*
 * You can find more information about the pipeline analysis given
 * by iaca on the following webpage:
 * http://kylehegeman.com/blog/2013/12/28/introduction-to-iaca/
 */

//profiling commandline for westmere architecture (WSM)
// /opt/iaca-lin64/bin/iaca -arch WSM -64 ./test
// /opt/iaca-lin64/bin/iaca -arch WSM -64 -analysis LATENCY ./test
// /opt/iaca-lin64/bin/iaca -arch WSM -64 -analysis LATENCY -graph test.dot ./test
// dot -Tpng test.dot1.dot -o test.png

//Commande de compilation alternative
// g++ ./main.cpp -O3 -msse2 -ffast-math -ftree-vectorizer-verbose=2 -o test

#define SIZE 1024

//Latence: temps entre le moment ou la première instruction rentre et la première sortie sort
int main()
{
	//std::vector<float,boost::alignment::aligned_allocator<float> > x(SIZE,1.0);
	//std::vector<float,boost::alignment::aligned_allocator<float> > y(SIZE,1.0);
	std::vector<float> x(SIZE,1.0);
	std::vector<float> y(SIZE,1.0);

	#pragma unroll
	for(int i=0;i<SIZE;i+=1)
	{
		IACA_START
		//2 Cycles pour 1 sortie
		x[i] = 3*x[i]+y[i];

		//9 Cycles pour 4 sorties
		/*x[i] = 3*x[i]+y[i];
		x[i+1] = 3*x[i+1]+y[i+1];
		x[i+2] = 3*x[i+2]+y[i+2];
		x[i+3] = 3*x[i+3]+y[i+3];*/


		//15 Cycles pour 8 sorties
		/*x[i] = 3*x[i]+y[i];
		x[i+1] = 3*x[i+1]+y[i+1];
		x[i+2] = 3*x[i+2]+y[i+2];
		x[i+3] = 3*x[i+3]+y[i+3];
		x[i+4] = 3*x[i+4]+y[i+4];
		x[i+5] = 3*x[i+5]+y[i+5];
		x[i+6] = 3*x[i+6]+y[i+6];
		x[i+7] = 3*x[i+7]+y[i+7];*/

		//15 Cycles pour 8 sorties
		/*float x0 = 3*x[i]+y[i];
		float x1 = 3*x[i+1]+y[i+1];
		float x2 = 3*x[i+2]+y[i+2];
		float x3 = 3*x[i+3]+y[i+3];
		float x4 = 3*x[i+4]+y[i+4];
		float x5 = 3*x[i+5]+y[i+5];
		float x6 = 3*x[i+6]+y[i+6];
		float x7 = 3*x[i+7]+y[i+7];

		x[i]   = x0;
		x[i+1] = x1;
		x[i+2] = x2;
		x[i+3] = x3;
		x[i+4] = x4;
		x[i+5] = x5;
		x[i+6] = x6;
		x[i+7] = x7;*/
	}
	IACA_END

	std::cout << "value "<<std::setprecision(10)<<x[x.size()-1]<< std::endl;
	return 0;
}
