// STL
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

// OpenMP
#include <omp.h>

#define CUTOFF 100  // arbitrary

template<typename T>
struct AddOne{
  T operator()(T in) {
    return in+(T)1;
  }
};

template<typename SizeT>
class RecursiveTransformEngine {
 public:
  RecursiveTransformEngine(size_t nbTrial=0, SizeT threshold=(SizeT)1) :
      m_nbTrial(nbTrial),
      m_trialIndex(0),
      m_threshold(threshold),
      m_testThreshold(threshold), 
      m_minThreshold(threshold),
      m_minMsec(std::numeric_limits<double>::max()) {}

 template<typename RdAcItr, typename UnaryOpT>
 void Launch(RdAcItr src, SizeT size, RdAcItr dst, UnaryOpT op) {
    std::chrono::time_point<std::chrono::steady_clock> start;
    if (m_trialIndex<m_nbTrial) {
      start = std::chrono::steady_clock::now();
    }
	#pragma omp parallel
    {
      #pragma omp single nowait
      {
		RecurseTransform(src, size, dst, op);
      }
    }
    if (m_trialIndex<m_nbTrial) {
	  auto stop = std::chrono::steady_clock::now();
	  auto diff = stop - start;
      auto msec = std::chrono::duration<double, std::milli>(diff).count();
      std::cout<<"Current RT With threshold "<<m_testThreshold
        <<" is "<<msec<<std::endl;
	  //Compute minimum runtime
      if ( msec<m_minMsec ) {
	    m_minMsec = msec;
        m_minThreshold = std::min(size,m_testThreshold);
        std::cout<<"Threshold value "<<m_minThreshold<<" hit a new record !!!"<<std::endl;
      }
      if (++m_trialIndex==m_nbTrial) {
        //Trial has ended, make choice
        m_threshold = std::min(size,m_minThreshold);
        std::cout<<"Final choice of "<<m_threshold<<" with best runtime of "<<
          m_minMsec<<std::endl;
      } else {
        //Start another trial iteration
        m_testThreshold = std::min(size,m_testThreshold*2);
        if (m_testThreshold==size) {
          m_trialIndex==m_nbTrial;
        }
        std::cout<<"Now testing a threshold of "<<m_testThreshold<<std::endl;
      }
    }
  }
  
  template<typename RdAcItr, typename UnaryOpT>
  void RecurseTransform(RdAcItr src, SizeT size, RdAcItr dst, UnaryOpT op) {
    //serial version, no random access iterator needed (forward is ok)
    if (size<=m_testThreshold) {
      std::transform(src,src+size,dst,op);
      return;
    }
	SizeT half = size / 2;
	#pragma omp task
	RecurseTransform(src, half, dst, op);
	#pragma omp task
	RecurseTransform(src+half, size-half, dst+half, op);
    return;
  }
 protected:
    double m_minMsec;
    size_t m_nbTrial;
    size_t m_trialIndex;
    SizeT m_threshold;
    SizeT m_testThreshold;
    SizeT m_minThreshold;
};

//Compile with openmp support using
//g++ ./test.cpp -o test -fopenmp
int main( int argc, char* argv[]) {

  std::vector<float> v(1<<24,1);


  // dummy tests
  std::cout<<"First test: maximal recursion depth"<<std::endl;
  RecursiveTransformEngine<size_t> transformer0(0,1);
  auto start = std::chrono::steady_clock::now();
  transformer0.Launch(v.begin(),v.size(),v.begin(),AddOne<float>());
  auto stop = std::chrono::steady_clock::now();
  auto diff = stop - start;
  auto msec = std::chrono::duration<double, std::milli>(diff).count();
  std::cout<<"Duration was : "<<msec<<std::endl;

  // dummy tests
  std::cout<<"First test: minimal recursion depth, should be a bit faster"<<std::endl;
  RecursiveTransformEngine<size_t> transformer1(0,v.size());
  start = std::chrono::steady_clock::now();
  transformer1.Launch(v.begin(),v.size(),v.begin(),AddOne<float>());
  stop = std::chrono::steady_clock::now();
  diff = stop - start;
  msec = std::chrono::duration<double, std::milli>(diff).count();
  std::cout<<"Duration was : "<<msec<<std::endl;
  
  // Now performing real test
  std::fill(v.begin(),v.end(),-1);
  RecursiveTransformEngine<size_t> transformer(25);
  for(int i=0; i<25; i++) {
	transformer.Launch(v.begin(),v.size(),v.begin(),AddOne<float>());
    assert(std::all_of(v.cbegin(),v.cend(),[i](auto in) {return in==i;}));
  }
  return EXIT_SUCCESS;
}
