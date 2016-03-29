//STL
#include <cstdlib>
#include <iostream>
#include <functional>
#include <algorithm>

//Local
#include "../vectorization.h"

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
	constexpr static int TapSize =
		TAP_SIZE_LEFT + TAP_SIZE_RIGHT + 1; //+1 = the center pixel
	//redefine template parameters
	constexpr static int VecSize = sizeof(VectorType)/sizeof(T);
	//How many vector are needed to load a single filter support
	constexpr static int NbVecPerFilt =
		(TapSize+VecSize-1)/(VecSize);
	constexpr static int TapSizeLeft = TAP_SIZE_LEFT;
	constexpr static int TapSizeRight = TAP_SIZE_RIGHT;
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
//TODO see __m256 _mm256_blendv_ps (__m256 a, __m256 b, __m256 mask) for AVX instruction
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
		PackType<T> l = VectorizedShift<T,PackType<T>,LeftShiftBytes>::LeftShift(left);
		PackType<T> r = VectorizedShift<T,PackType<T>,RightShiftBytes>::RightShift(right);

		//Return the generated vector
		return l+r;
	}
private:
	constexpr static int VecSize = sizeof(PackType<T>)/sizeof(T);
	constexpr static int VecLeftIdx = ((PREFETCH_BEGIN_IDX+SUPPORT_IDX)/VecSize)*VecSize;
	constexpr static int VecRightIdx = VecLeftIdx+VecSize;
	constexpr static int LeftShift = ((PREFETCH_BEGIN_IDX+SUPPORT_IDX)%VecSize);
	constexpr static int LeftShiftBytes = LeftShift*sizeof(T);
	constexpr static int RightShiftBytes = (VecSize-LeftShift)*sizeof(T);
};

template<typename T, class FILT, int PREFETCH_BEGIN_IDX, int SUPPORT_IDX>
class ConvolutionAccumulator
{
public:
	static typename FILT::VectorType Accumulate(T* prefetch)
	{
		//Recursive call over all previous index of filter support
		typename FILT::VectorType accumulator = ConvolutionAccumulator<T,FILT,PREFETCH_BEGIN_IDX,SUPPORT_IDX-1>::Accumulate( prefetch );

		//Craft newVec from two vectors
		typename FILT::VectorType newVec = ConvolutionShifter<T,PREFETCH_BEGIN_IDX,SUPPORT_IDX>::generateNewVec( prefetch );

		//Accumulate
		return accumulator + FILT::Buf[SUPPORT_IDX]*newVec;
	}
};

//Partial template specialization for iteration 0 of the loop
template<typename T, class FILT, int PREFETCH_BEGIN_IDX>
class ConvolutionAccumulator<T,FILT,PREFETCH_BEGIN_IDX,0>
{
public:
	static typename FILT::VectorType Accumulate(T* prefetch) //accumulator is uninitialized
	{
		//generate new vector if firstindex to process was not aligned, PrefetchBeginIdx is not null
		typename FILT::VectorType newVec = ConvolutionShifter<T,PREFETCH_BEGIN_IDX,0>::generateNewVec( prefetch );

		//First call: we must initialize the accumulator
		return FILT::Buf[0]*newVec;
	}
};

inline int positive_modulo(int i, int n)
{
    return (i % n + n) % n;
}

template<class FILT>
class Convolution
{
public:
	Convolution()=default;
	static void Convolve( const typename FILT::ScalarType* in, typename FILT::ScalarType* out, const int lineSize )
	{
		//How many vectors can be easily right processed without trouble loading bounds
		const int RightProcessableVectPerLine =
			((lineSize/FILT::VecSize)*FILT::VecSize //"vector loadable" scalar size (minus the modulo)
			-FILT::TapSizeRight)/FILT::VecSize;	 //right processable area (outside we cannot load the right tap)
		//Vector aligned scalar index to end with (excluded)		
		const int LastIndexToProcess = RightProcessableVectPerLine*FILT::VecSize;

		if( FirstIndexToProcess >= LastIndexToProcess )
		{
			//The naive implementation for small sizes
			for(int i = 0; i<lineSize;i++)
			{
				for(int k = i-FILT::TapSizeLeft; k <= i+FILT::TapSizeRight; k++)
				{
					out[i] += FILT::Buf[k-i+FILT::TapSizeLeft] * in[positive_modulo(k,lineSize)];
				}
			}
		}else //The vectorized implementation
		{
			//////// handle prefix bound : non vectorized implementation
			for(int i = 0; i<FirstIndexToProcess;i++)
			{
				for(int k = i-FILT::TapSizeLeft; k <= i+FILT::TapSizeRight; k++)
				{
					out[i] += FILT::Buf[k-i+FILT::TapSizeLeft] * in[positive_modulo(k,lineSize)];
				}
			}

			//////// handle vectorizable part

			//Buffer containg the prefetch area to be load in vectorized registers
			typename FILT::ScalarType prefetch[PrefetchCardinality*FILT::VecSize];

			//1st : fill the PrefetchCardinality-1 vectors with data
			std::copy(in,in+(PrefetchCardinality-1)*FILT::VecSize,prefetch);

			//Now we must perform regular loop, iterating over vectors
			#pragma unroll
			for( int i = FirstIndexToProcess; i<LastIndexToProcess; i+= FILT::VecSize )
			{
				//Load next prefetch buffer, in the last vector
				VectorizedMemOp<typename FILT::ScalarType,
					PackType<typename FILT::ScalarType> >::store(
						prefetch+(PrefetchCardinality-1)*FILT::VecSize,
						VectorizedMemOp<typename FILT::ScalarType,
							PackType<typename FILT::ScalarType> >::load(
								in+i+ShiftBetweenProcessedAndLastLoaded ) );

				//Store the result of the convolution
				VectorizedMemOp<typename FILT::ScalarType,
					PackType<typename FILT::ScalarType> >::store(
						out+i,
						ConvolutionAccumulator<typename FILT::ScalarType,
							FILT,PrefetchBeginIdx,FILT::TapSize-1>::Accumulate(prefetch)
				);

				//last : left shift buffer to be updated
				//destination iterator is BEFORE source iterator, use of std::copy is ok
				std::copy( prefetch+FILT::VecSize,
						prefetch+PrefetchCardinality*FILT::VecSize,
						prefetch);
			}

			//////// handle suffix bound : non vectorized implementation
			for(int i = LastIndexToProcess; i<lineSize; i++)
			{
				for(int k = i-FILT::TapSizeLeft; k <= i+FILT::TapSizeRight; k++)
				{
					out[i] += FILT::Buf[k-i+FILT::TapSizeLeft] * in[positive_modulo(k,lineSize)];
				}
			}
		}
	}

protected:
	//To out 1 processed vector, how many vector should we load
	static const int PrefetchCardinality =
		// left tap size part
		((FILT::TapSizeLeft+FILT::VecSize-1)/FILT::VecSize+
		// central area to be processed
		1+
		// right tap part
		(FILT::TapSizeRight+FILT::VecSize-1)/FILT::VecSize);
	//Vector aligned scalar index from output vector to begin with
	static const int FirstIndexToProcess =
		((FILT::TapSizeLeft+
		FILT::VecSize-1)/FILT::VecSize)	//Min number of vector to load left tap
		*FILT::VecSize;		//multiplied by FILT::VecSize to get a scalar index
	//non aligned scalar index from prefetch to begin with
	static const int PrefetchBeginIdx =
		FILT::VecSize-(FILT::TapSizeLeft%FILT::VecSize);
	/*
	 * non aligned scalar index difference between first processed index
	 * and index of last prefetch to be loaded, actually equal to the number
	 * of vector covering the vecSize+TapSizeRight area minus 1 vector
	 */
	static const int ShiftBetweenProcessedAndLastLoaded =
		((( 2*FILT::VecSize + FILT::TapSizeRight + 1)/
				FILT::VecSize)-1)*FILT::VecSize;
};		

//build with g++ ./test.cpp -std=c++11 -O3 -o test -DUSE_SSE
int main(int argc, char* argv[])
{
	std::vector<float> input(4,1);
	std::vector<float> output(input.size(),0);

	Convolution< MeanFilter<float,1,1> >::Convolve( input.data(), output.data(), input.size() );

	std::for_each(output.cbegin(),output.cend(),
		[](float in){std::cout << "val= "<< in << std::endl;});
	
	return EXIT_SUCCESS;
}
