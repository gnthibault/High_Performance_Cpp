//STL
#include <vector>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <algorithm>

//Local
#include "vectorization.h"

//C++11 strongly typed enum
enum class Vectorization {SSE,AVX};

/*
 * TAP_SIZE_LEFT does only account for number of elements at left,
 * without the current one.
 * TAP_SIZE_RIGHT does only account for number of elements at right,
 * without the current one.
 * Convolution for VEC_SIZE elements will be computed
 */
template<typename T, int TAP_SIZE_LEFT, int TAP_SIZE_RIGHT>
class Filter
{
public:
	Filter()=default;
public:
	//Typedef main type
	typedef T ScalarType;
	//Typedef vector type
	typedef PackType<T> VectorType;
	//Total size of the filter, in number of elements
	static const int TapSize =
		TAP_SIZE_LEFT + TAP_SIZE_RIGHT + 1; //+1 = the center pixel
	//redefine template parameters
	static const int VecSize = sizeof(VectorType)/sizeof(T);
	//How many vector are needed to load a single filter support
	static const int NbVecPerFilt =
		(TapSize+VecSize-1)/(VecSize);
	static const int TapSizeLeft = TAP_SIZE_LEFT;
	static const int TapSizeRight = TAP_SIZE_RIGHT;
};

/*
 * We define an inheritance of the fully generic filter for the "mean filter"
 */
template<typename T, int TAP_SIZE_LEFT, int TAP_SIZE_RIGHT>
class MeanFilter : public Filter<T,TAP_SIZE_LEFT,TAP_SIZE_RIGHT>
{
public:
	MeanFilter()=default;
public:
	static const T Buf[Filter<T,TAP_SIZE_LEFT,TAP_SIZE_RIGHT>::TapSize];
};

/*
 * Compile time declaration of the simple filter using full specialization
 */
template<> const float MeanFilter<float,1,1>::Buf[3] = {0.33333f,0.33333f,0.33333f};

/*
 * From the prefetch buffer in input, generates a vector that contains
 * the "SUPPORT_IDX" th element of each element of the current "output" vector
 * of the algorithm.
 */
template<typename T, int PREFETCH_BEGIN_IDX, int SUPPORT_IDX>
class ConvolutionShifter
{
public:
	static PackType<T> generateNewVec(T* prefetch)
	{
		//Fetch left part and right part, to be mixed after
		PackType<T> left	= VectorizedMemOp<T,PackType<T> >::load( prefetch+VecLeftIdx );
		PackType<T> right	= VectorizedMemOp<T,PackType<T> >::load( prefetch+VecRightIdx );

		//Perform shift on both operand
		PackType<T> l = VectorizedShift<T,PackType<T>,LeftShift>::LeftShift(left);
		PackType<T> r = VectorizedShift<T,PackType<T>,RightShift>::RightShift(right);

		//Return the generated vector
		return l+r;
	}
private:
	static const int VecSize = sizeof(PackType<T>)/sizeof(T);
	static const int VecLeftIdx = ((PREFETCH_BEGIN_IDX+SUPPORT_IDX)/VecSize)*VecSize;
	static const int VecRightIdx = VecLeftIdx+VecSize;
	static const int LeftShift = ((PREFETCH_BEGIN_IDX+SUPPORT_IDX)%VecSize)*VecSize;
	static const int RightShift = VecSize-LeftShift;
};

template<typename T, class FILT, int PREFETCH_BEGIN_IDX, int SUPPORT_IDX>
class ConvolutionAccumulator
{
public:
	static FILT::VectorType Accumulate(T* prefetch)
	{
		//Recursive call over all previous index of filter support
		FILT::VectorType accumulator = ConvolutionAccumulator<T,FILT,PREFETCH_BEGIN_IDX,SUPPORT_IDX-1>::Accumulate( prefetch );

		//Craft newVec from two vectors
		FILT::VectorType newVec = ConvolutionShifter<T,FILT::VecSize,SUPPORT_IDX,PREFETCH_BEGIN_IDX>::generateNewVec( prefetch );

		//Accumulate
		return accumulator + FILT::Buf[SUPPORT_IDX]*newVec;
	}
};

//Partial template specialization for iteration 0 of the loop
template<typename T, class FILT, int PREFETCH_BEGIN_IDX>
class ConvolutionAccumulator<T,FILT,PREFETCH_BEGIN_IDX,0>
{
public:
	static FILT::VectorType Accumulate(T* prefetch) //accumulator is uninitialized
	{
		//generate new vector if firstindex to process was not aligned, PrefetchBeginIdx is not null
		FILT::VectorType newVec = ConvolutionShifter<T,FILT::VecSize,0,PREFETCH_BEGIN_IDX>::generateNewVec( prefetch );

		//First call: we must initialize the accumulator
		return FILT::Buf[0]*newVec;
	}
};

template<class FILT, typename T>
class Convolution
{
public:
	Convolution()=default;
	void Test( const std::vector<T>& input, std::vector<T>& output )
	{
		//How many vectors can be easily right processed without trouble loading bounds
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

			//Accumulator for the filtering result, initialization comes with first step
			T accumulator[FILT::VecSize];
			//new vector to be crafter by shiftin non overlapping vectors
			T newVec[FILT::VecSize];

			ConvolutionAccumulator<T,FILT,PrefetchBeginIdx,FILT::TapSize>::Accumulate(
					newVec, prefetch, accumulator);

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
