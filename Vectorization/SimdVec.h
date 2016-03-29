/*
 * SimdVec.h
 *
 *  Created on: 29 mars 2016
 *      Author: gnthibault
 */

#ifndef VECTORIZATION_SIMDVEC_H_
#define VECTORIZATION_SIMDVEC_H_

/*
 * A first attempt of a generic simd aware
 * iterable vector class
 */

//STL
#include <vector>

//boost
#include <boost/align/aligned_allocator.hpp>

//Local
#include "vectorization.h"


/*
 * An iterator must support an operator* method, an operator != method,
 * and an operator++ method
 */
template<typename T>
class SimdIter
{
public:
    SimdIter(SimdVec<T>* vec, size_t idx) : m_idx( idx ), m_vec( vec ) {}

    // these three methods form the basis of an iterator for use with
    // a range-based for loop
    bool operator!=(const SimdIter<T>& other) const
    {
        return m_idx != other.m_idx;
    }

    // this method must be defined after the definition of SimdVec
    // since it needs to use it, here we decide not to return
    // a reference, modification will have to use the set method
    PackType<T> operator* () const;

    //A simple alias for operator*
    PackType<T> get() const { return *(*this); };

    // this method must be defined after the definition of SimdVec
	// since it needs to use it
    void set( PackType<T> val );

    SimdIter<T>& operator++() //prefix
    {
    	// incrementing index accounting for the multiple elements
    	// of the packed type
        m_idx+=(sizeof(PackType<T>)/sizeof(T));
        // although not strictly necessary for a range-based for loop
        // following the normal convention of returning a value from
        // operator++ is a good idea.
        return *this;
    }

    SimdIter<T> operator++(int) //suffix
	{
	   m_idx+=(sizeof(PackType<T>)/sizeof(T));
	   return *this;
	}

private:
    size_t m_idx;
    SimdVec<T> *m_vec;
};
//The const iterator
template<typename T>
class SimdIterConst
{
public:
	SimdIterConst(const SimdVec<T>* vec, size_t idx) : m_idx( idx ), m_vec( vec ) {}

    bool operator!=(const SimdIterConst<T>& other) const
    {
        return m_idx != other.m_idx;
    }
    PackType<T> operator* () const;
    PackType<T> get() const { return *(*this); };
    SimdIterConst<T>& operator++() //prefix
    {
        m_idx+=(sizeof(PackType<T>)/sizeof(T));
        return *this;
    }
    SimdIterConst<T> operator++(int) //suffix
	{
	   m_idx+=(sizeof(PackType<T>)/sizeof(T));
	   return *this;
	}

private:
    size_t m_idx;
    const SimdVec<T> *m_vec;
};

/*
 * An iterable object must feature a begin and a end methods that return
 * iterators to the beginning and end of the "vector"
 */
template<typename T>
class SimdVec
{
public:
    SimdVec(size_t size, T initVal = (T)0)
	{
    	size_t nbElementPerVector = sizeof(PackType<T>)/sizeof(T);
    	//Compute the minimum number of vector that should be used
    	size_t newSize = (size+nbElementPerVector-1)/nbElementPerVector;
        m_vec.resize( newSize*nbElementPerVector, initVal );
	}

    SimdIter<T> begin()
    {
        return SimdIter<T>( this, 0 );
    }
    SimdIterConst<T> cbegin() const
    {
        return SimdIterConst<T>( this, 0 );
    }
    SimdIter<T> end()
    {
        return SimdIter<T>( this, m_vec.size() );
    }
    SimdIterConst<T> cend() const
	{
		return SimdIterConst<T>( this, m_vec.size() );
	}

    //We also authorize non sse2 iterators
    typename std::vector<T,boost::alignment::aligned_allocator<T> >::iterator
	scalarbegin() { return m_vec.begin(); }
    typename std::vector<T,boost::alignment::aligned_allocator<T> >::iterator
	scalarend() { return m_vec.end(); }
    typename std::vector<T,boost::alignment::aligned_allocator<T> >::const_iterator
	cscalarbegin() { return m_vec.cbegin(); }
    typename std::vector<T,boost::alignment::aligned_allocator<T> >::const_iterator
    cscalarend() { return m_vec.cend(); }

    //This is an unsafe get, be carefull about what
    //does the last index return if size was not a multiple
    //of vector size
    PackType<T> get( size_t idx ) const
    {
         return load(m_vec.data()+idx);
    }

    //Unsafe set
    void set( size_t idx, PackType<T> val )
	{
		 store( m_vec.data()+idx, val );
	}

protected:
    std::vector<T,boost::alignment::aligned_allocator<T> > m_vec;
};

template<typename T>
PackType<T> SimdIter<T>::operator*() const
{
     return m_vec->get(m_idx);
}

template<typename T>
PackType<T> SimdIterConst<T>::operator*() const
{
     return m_vec->get(m_idx);
}

template<typename T>
void SimdIter<T>::set(PackType<T> val)
{
     return m_vec->set(m_idx, val);
}



#endif /* VECTORIZATION_SIMDVEC_H_ */
