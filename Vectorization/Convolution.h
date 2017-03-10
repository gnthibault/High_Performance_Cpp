//STL
#include <cstdlib>
#include <iostream>
#include <functional>
#include <algorithm>

//Local
#include "vectorization.h"

/*
 * TAP_SIZE_LEFT does only account for number of elements at left,
 * without the current one.
 * TAP_SIZE_RIGHT does only account for number of elements at right,
 * without the current one.
 * Convolution for VEC_SIZE elements will be computed
 */
template<typename T, int TAP_SIZE_LEFT, int TAP_SIZE_RIGHT>
class Filter {
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
 * We define an inheritance of the fully generic filter for an arbitrary
 * MyFilter
 */
template<typename T, int TAP_SIZE_LEFT, int TAP_SIZE_RIGHT>
class MyFilter : public Filter<T,TAP_SIZE_LEFT,TAP_SIZE_RIGHT> {
public:
  MyFilter()=default;
public:
  static const T Buf[Filter<T,TAP_SIZE_LEFT,TAP_SIZE_RIGHT>::TapSize];
};

/*
 * From the prefetch buffer in input, generates a vector that contains
 * the "SUPPORT_IDX" th element of each element of the current "output" vector
 * of the algorithm.
 */
template<typename T, int PREFETCH_BEGIN_IDX, int SUPPORT_IDX>
class ConvolutionShifter {
public:
  static PackType<T> generateNewVec(T* prefetch) {
    //Fetch left part and right part, to be mixed after
    PackType<T> left = VectorizedMemOp<T,PackType<T> >::load(
        prefetch+VecLeftIdx );
    PackType<T> right = VectorizedMemOp<T,PackType<T> >::load(
        prefetch+VecRightIdx );

    //Return the generated vector
    return VectorizedConcatAndCut<T,PackType<T>,RightShift>::Concat(
        left,right);
  }
private:
  constexpr static int VecSize = sizeof(PackType<T>)/sizeof(T);
  //Indexes in the prefetch buffer of vector to be loaded
  constexpr static int VecLeftIdx = ((PREFETCH_BEGIN_IDX+SUPPORT_IDX)/VecSize)
    *VecSize;
  constexpr static int VecRightIdx = VecLeftIdx+VecSize;
  //Shift that should be applied to the 2 neighbouring vector to be blended
  //together
  constexpr static int RightShift = ((PREFETCH_BEGIN_IDX+SUPPORT_IDX)%VecSize);
};

template<typename T, class FILT, int PREFETCH_BEGIN_IDX, int SUPPORT_IDX>
class ConvolutionAccumulator {
public:
  static typename FILT::VectorType Accumulate(T* prefetch) {
    //Recursive call over all previous index of filter support
    typename FILT::VectorType accumulator = ConvolutionAccumulator<T,FILT,
      PREFETCH_BEGIN_IDX,SUPPORT_IDX-1>::Accumulate(prefetch);

    //Craft newVec from two vectors
    typename FILT::VectorType newVec = ConvolutionShifter<T,PREFETCH_BEGIN_IDX,
      SUPPORT_IDX>::generateNewVec( prefetch );

    //Accumulate
    return accumulator + FILT::Buf[SUPPORT_IDX]*newVec;
  }
};

//Partial template specialization for iteration 0 of the loop
template<typename T, class FILT, int PREFETCH_BEGIN_IDX>
class ConvolutionAccumulator<T,FILT,PREFETCH_BEGIN_IDX,0> {
public:
  //accumulator is uninitialized
  static typename FILT::VectorType Accumulate(T* prefetch) {
    //generate new vector if firstindex to process was not aligned,
    //PrefetchBeginIdx is not null
    typename FILT::VectorType newVec=
      ConvolutionShifter<T,PREFETCH_BEGIN_IDX,0>::generateNewVec(prefetch);

    //First call: we must initialize the accumulator
    return FILT::Buf[0]*newVec;
  }
};

inline int positive_modulo(int i, int n) {
  return (i % n + n) % n;
}

template<class FILT>
class Convolution {
public:
  Convolution()=default;
  static void NaiveConvolve(const typename FILT::ScalarType* in,
    typename FILT::ScalarType* out, const int firstIndexIncluded,
    const int lastIndexExcluded, const int lineSize) {

    //The naive implementation for small sizes
    for (int i = firstIndexIncluded; i<lastIndexExcluded;i++) {
      for (int k = i-FILT::TapSizeLeft; k <= i+FILT::TapSizeRight; k++) {
        out[i] += FILT::Buf[k-i+FILT::TapSizeLeft]*
          in[positive_modulo(k,lineSize)];
      }
    }
  }
  
  static void Convolve(const typename FILT::ScalarType* in,
    typename FILT::ScalarType* out, const int lineSize) {
    //How many vectors can be easily right processed without trouble loading
    //bounds
    const int RightProcessableVectPerLine =
      //"vector loadable" scalar size (minus the modulo)
      ((lineSize/FILT::VecSize)*FILT::VecSize
      //right processable area (outside we cannot load the right tap)
      -FILT::TapSizeRight)/FILT::VecSize;
    //Vector aligned scalar index to end with (excluded)		
    const int LastIndexToProcess = RightProcessableVectPerLine*FILT::VecSize;

    if (FirstIndexToProcess >= LastIndexToProcess) {
      NaiveConvolve( in, out, 0, lineSize, lineSize );
    } else { //The vectorized implementation
      //////// handle prefix bound : non vectorized implementation
      NaiveConvolve( in, out, 0, FirstIndexToProcess, lineSize );
      
      //////// handle vectorizable part

      //Buffer containg the prefetch area to be loaded in vectorized registers
      typename FILT::ScalarType prefetch[PrefetchCardinality*FILT::VecSize];

      //1st : fill the PrefetchCardinality-1 vectors with data
      std::copy(in,in+(PrefetchCardinality-1)*FILT::VecSize,prefetch);

      //Now we must perform regular loop, iterating over vectors
      #pragma unroll
      for (int i = FirstIndexToProcess; i<LastIndexToProcess;
        i+= FILT::VecSize ) {
        //Load next prefetch buffer, in the last vector
        VectorizedMemOp<typename FILT::ScalarType,
	  PackType<typename FILT::ScalarType> >::store(
	    prefetch+(PrefetchCardinality-1)*FILT::VecSize,
	    VectorizedMemOp<typename FILT::ScalarType,
	  PackType<typename FILT::ScalarType> >::load(
	    in+i+ShiftBetweenProcessedAndLastLoaded));

        //std::cout<<"Storing at address "<<i<<std::endl;
        //Store the result of the convolution
	VectorizedMemOp<typename FILT::ScalarType,
	  PackType<typename FILT::ScalarType> >::store(
	    out+i,
	    ConvolutionAccumulator<typename FILT::ScalarType,
	    FILT,PrefetchBeginIdx,FILT::TapSize-1>::Accumulate(prefetch));

	//last : left shift buffer to be updated
	//destination iterator is BEFORE source iterator, we can use std::copy
	std::copy( prefetch+FILT::VecSize,
          prefetch+PrefetchCardinality*FILT::VecSize, prefetch);
      }

      //////// handle suffix bound : non vectorized implementation
      NaiveConvolve(in, out, LastIndexToProcess, lineSize, lineSize);
  }
}

protected:
  //To output 1 processed vector, how many vector should we load
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
    FILT::VecSize-1)/FILT::VecSize)  //Min number of vector to load left tap
    *FILT::VecSize;  //multiplied by FILT::VecSize to get a scalar index

  //non aligned scalar index from prefetch to begin with
  static const int PrefetchBeginIdx =
    (FILT::TapSizeLeft%FILT::VecSize) == 0 ? 0 :
    FILT::VecSize-(FILT::TapSizeLeft%FILT::VecSize);

  /*
   * non aligned scalar index difference between first processed index
   * and index of last prefetch to be loaded, actually equal to the number
   * of vector covering the vecSize+TapSizeRight area minus 1 vector
   */
  static const int ShiftBetweenProcessedAndLastLoaded =
    ((( 2*FILT::VecSize + FILT::TapSizeRight - 1)/
    FILT::VecSize)-1)*FILT::VecSize;
};
