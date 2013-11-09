#ifndef __ml_VectorLessPointerShuffle__
#define __ml_VectorLessPointerShuffle__

#include <map>
#include <iostream>
#include <functional>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <iostream>
#include "ArgumentException.h"
#include "StateException.h"
#include "matrix.h"

namespace ml
{
template <typename Type>
class VectorLessPointerShuffle : std::binary_function<std::vector<Type>*,std::vector<Type>*,bool>
{
private:
    std::size_t index;
    const std::vector<std::pair<std::size_t, Type> > * FeatureAttributes;
    const std::map<std::size_t, std::map<Type, double> > *
        NominalFeatureProbabilities;
public:
    typedef std::map<std::size_t, std::map<double, double> > 
        feature_probability_type;
        

    VectorLessPointerShuffle()
    {
        throw StateException("DO NOT default construct, sparky.");
    }

    VectorLessPointerShuffle(
                    const std::vector<std::pair<std::size_t, Type> > *
                        FeatureAttr,
                    const std::map<std::size_t, std::map<Type, double> > *
                        probabilities,
                    std::size_t i) : index(i), 
                    FeatureAttributes(FeatureAttr),
                    NominalFeatureProbabilities(probabilities)
    {
    }
    
    VectorLessPointerShuffle(const VectorLessPointerShuffle<Type> & other) 
        : index(other.index),
                FeatureAttributes(other.FeatureAttributes),
                NominalFeatureProbabilities(other.NominalFeatureProbabilities)
    {
    }
   
    VectorLessPointerShuffle & operator = (
            const VectorLessPointerShuffle<Type> & other)
    {
        if (this != &other)
        {
            index = other.index;
            FeatureAttributes = other.FeatureAttributes;
            NominalFeatureProbabilities = 
                other.NominalFeatureProbabilities;
        }
        return *this;
    }

    ~VectorLessPointerShuffle()
    {
        index = -1;
    }

    bool operator () (const std::vector<Type> * a, 
            const std::vector<Type> * b) const
    {

        if (a == NULL || b == NULL || a->size() != b->size())
        {
            throw ArgumentException(
                "Args must be non-null and point to equally-sized vector.");
        }

        if (a->size() == 0)
        {
            throw ArgumentException(
                "Args must be non-empty. This is ridiculous.");
        }

        int i = index % a->size();
        for (int j = 0; j < a->size(); j++) 
        {
            
            double a_val = (*a)[i] == UNKNOWN_VALUE ? 
                (*FeatureAttributes)[i].second : 
                (*a)[i];
            double b_val = (*b)[i] == UNKNOWN_VALUE ? 
                (*FeatureAttributes)[i].second : 
                (*b)[i];

            double a_compare = (*FeatureAttributes)[i].first > 0 ?
         (*(*(*NominalFeatureProbabilities).find(i)).second.find(a_val)).second :
                a_val;
            double b_compare = (*FeatureAttributes)[i].first > 0 ?
         (*(*(*NominalFeatureProbabilities).find(i)).second.find(b_val)).second :
                b_val;

            if ( a_compare < b_compare )
            {
                return true;
            }
            else if ( b_compare < a_compare )
            {
                return false;
            }
            i = (i + 1) % a->size();
        }
        return false;
    }
};
}

#endif // __ml_VectorLessPointerShuffle__
