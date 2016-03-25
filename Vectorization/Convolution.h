//STL
#include <vector>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <algorithm>

/*
 * TAP_SIZE_LEFT does only account for number of elements at left,
 * without the current one.
 * TAP_SIZE_RIGHT does only account for number of elements at right,
 * without the current one.
 * Convolution for VEC_SIZE elements will be computed
 */


template<typename T, int VEC_SIZE, int TAP_SIZE_LEFT, int TAP_SIZE_RIGHT>
class Filter
{
public:
	Filter()=default;
public:
	//Total size of the filter, in number of elements
	static const int TapSize =
		TAP_SIZE_LEFT + TAP_SIZE_RIGHT + 1; //+1 = the center pixel
	//How many vector are needed to load a single filter support
	static const int NbVecPerFilt =
		(TapSize+VEC_SIZE-1)/(VEC_SIZE);
	//redefine template parameters
	static const int VecSize = VEC_SIZE;
	static const int TapSizeLeft = TAP_SIZE_LEFT;
	static const int TapSizeRight = TAP_SIZE_RIGHT;
};

template<typename T, int VEC_SIZE, int TAP_SIZE_LEFT, int TAP_SIZE_RIGHT>
class MeanFilter : public Filter<T,VEC_SIZE,TAP_SIZE_LEFT,TAP_SIZE_RIGHT>
{
public:
	MeanFilter()=default;
public:
	static const T Buf[Filter<T,VEC_SIZE,TAP_SIZE_LEFT,TAP_SIZE_RIGHT>::TapSize];
};

/*
 * Compile time declaration of the simple filter. It is very important
 * to notice that, if filter has its size not a multiple of vec size,
 * It should be padded at the end with zeros
 */
template<> const float MeanFilter<float,4,1,1>::Buf[3] = {0.33333f,0.33333f,0.33333f};

/*template<int LS, int RS>
__m128 shiftAdd(__m128 left, __m128 right)
{
	//Left shift
	__m128 result0 = (__m128)_mm_slli_si128( (__m128i)left, LS );
	//Right shift
	__m128 result1 = (__m128)_mm_srli_si128( (__m128i)right, RS );
	return _mm_add_ps( result0, result1 );
}*/

template<typename T, int VEC_SIZE>
void generateNewVec(T* newVec, T* prefetch, int VecLeftIdx, int leftShift)
{
	//handle left vector
	for(int i=0; i<VEC_SIZE-leftShift; i++)
	{
		newVec[i] = prefetch[VEC_SIZE*VecLeftIdx+leftShift+i];
	}

	//handle right vector
	for(int i=0; i<leftShift; i++)
	{
		newVec[VEC_SIZE-leftShift+i] = prefetch[VEC_SIZE*(VecLeftIdx+1)+i];
	}
}

template<class FILT, typename T>
class Convolution
{
public:
	Convolution()=default;
	void Test( const std::vector<T>& input, std::vector<T>& output )
	{
		//How many vectors can be easiliy right processed without trouble loading bounds
		const int RightProcessableVectPerLine =
			((input.size()/FILT::VecSize)*FILT::VecSize //"vector loadable" scalar size (minus the modulo)
			-FILT::TapSizeRight)/FILT::VecSize;	 //right processable area (outside we cannot load the right tap)
		//Vector aligned scalar index to end with (excluded)		
		const int LastIndexToProcess = RightProcessableVectPerLine*FILT::VecSize;
		
		//Buffer containg the prefetch area to be load in vectorized registers		
		T prefetch[PrefetchCardinality*FILT::VecSize];

		//Raw buffer ptr
		const T* in = input.data();

		//1st : fill the FILT::NbVecPerFilt-1 vectors with data
		for( int k = 0; k < (PrefetchCardinality-1)*FILT::VecSize; k+=FILT::VecSize)
		{
			std::copy(in+k,in+k+FILT::VecSize,prefetch+k);
		}

		//Now we must perform regular loop, iterating over vectors
		for( int i = FirstIndexToProcess; i<LastIndexToProcess; i+= FILT::VecSize )
		{
			//Load next prefetch buffer, in the last vector
			std::copy(in+i,in+i+FILT::VecSize,prefetch+(PrefetchCardinality-1)*FILT::VecSize);

			//the first index inside the prefetch buffer
			int prefIdx = PrefetchBeginIdx;

			//Accumulator for the filtering result, initialization comes with first step
			T accumulator[FILT::VecSize];
			//new vector to be crafter by shiftin non overlapping vectors
			T newVec[FILT::VecSize];
			//generate new vector if firstindex to process was not aligned, PrefetchBeginIdx is not null
			generateNewVec<T,FILT::VecSize>(newVec,prefetch,0,PrefetchBeginIdx);
			//perform stage 0 of dot product vectorized			
			for(int k=0; k<FILT::VecSize; k++)
			{
				accumulator[k] = FILT::Buf[0]*newVec[k];
			}


			//For each following index of the support, load the corresponding vector
			for(int k = 1; k < FILT::TapSize; k++)
			{
				int VecLeftIdx = (prefIdx+k)/FILT::VecSize;
				//int VecRightIdx = (prefIdx+k+FILT::VecSize-1)/FILT::VecSize;
				int leftShift = (prefIdx+k)%FILT::VecSize;
				//int rightShift = FILT::VecSize-leftShift;

				//Craft newvec from two vectors
				generateNewVec<T,FILT::VecSize>(newVec,prefetch,VecLeftIdx,leftShift);
				
				//the k^th element of the support for each element of the vector
				for(int l=0; l<FILT::VecSize; l++)
				{
					accumulator[l] += FILT::Buf[k]*newVec[l];
				}
			}

			//Write the result in the output
			std::copy(accumulator,accumulator+FILT::VecSize,output.begin()+i);

			//last : left shift buffer to be updated
			for( int k = 0; k < (PrefetchCardinality-1)*FILT::VecSize; k+=FILT::VecSize)
			{
				std::copy(prefetch+k+FILT::VecSize,
					prefetch+k+2*FILT::VecSize,
					prefetch+k);
			}
		}
	}

protected:
	//To output 1 processed vector, how many vector should we load
	static const int PrefetchCardinality =
		(FILT::TapSizeLeft+
		FILT::VecSize+
		FILT::TapSizeRight+
		FILT::VecSize-1)/(FILT::VecSize);
	static const int FirstIndexToProcess =//Vector aligned scalar index to begin with
		((FILT::TapSizeLeft+
		FILT::VecSize-1)/FILT::VecSize)	//Min number of vector to load left tap
		*FILT::VecSize;		//*FILT::VecSize to get a scalar index
	static const int PrefetchBeginIdx =
		FirstIndexToProcess%FILT::TapSizeLeft;
};		


int main(int argc, char* argv[])
{
	std::vector<float> input(40,1);
	std::vector<float> output(40,0);

	Convolution< MeanFilter<float,4,1,1>, float > conv;
	conv.Test( input, output );

	std::for_each(output.cbegin(),output.cend(),
		[](float in){std::cout << "val= "<< in << std::endl;});
	
	return EXIT_SUCCESS;
}
