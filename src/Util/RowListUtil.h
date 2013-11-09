#ifndef __ml_RowListUtil__
#define __ml_RowListUtil__

#include <list>
#include <utility>
#include <vector>
#include <map>
#include <cstdlib>
#include "matrix.h"

namespace ml
{
typedef std::list<std::pair<std::vector<double>*,double> > RowList;
typedef RowList::iterator RowListIterator;
typedef RowList::const_iterator RowListCIterator;
typedef RowList::reverse_iterator RowListRIterator;

void TransposeMatrix(Matrix & from, Matrix & to);
void GetRowList(RowList & l, Matrix & Features, const std::vector<double> & Labels);


double GetEntropy(const std::map<int, std::size_t> & LabelCensus, 
        std::size_t ExampleSize);

void TakeFeatureCensus(const RowList & Examples,
        std::map<std::size_t, std::map<double, std::size_t> > & FeatureCensus,
        const std::vector<std::pair<std::size_t, double> > & FeatureAttributes);

void TakeLabelCensus(const RowList & Examples, 
        std::map<int, std::size_t> & LabelCensus, std::size_t LabelAttributes, 
        int LCValue);

template <typename LabelType, typename CountType>
bool CensusCompare(std::pair<LabelType, CountType> a, 
        std::pair<LabelType, CountType> b)
{
    return a.second < b.second;
}

template <typename InputIterator, typename LabelType>
void GenericTakeLabelCensus(InputIterator first, InputIterator last,
        std::map<LabelType, std::size_t> & LabelCensus, 
        std::size_t LabelAttributes, LabelType LCValue)
{
    typedef typename std::map<LabelType,size_t>::iterator CensusIterator;
    LabelCensus.clear();
    std::pair<CensusIterator, bool> given;
    given = LabelCensus.insert(std::make_pair((LabelType)0, (size_t)0));
    CensusIterator iter = given.first;
    for (int i = 1; i < LabelAttributes; i++)
    {
        iter = LabelCensus.insert(iter, std::make_pair((LabelType)i, (size_t)0));
    }

    InputIterator Example = first;
    while (Example != last)
    {
        int label;
        if (Example->second == UNKNOWN_VALUE)
        {
            label = LCValue;
        }
        else
        {
            label = (LabelType)Example->second;
        }
        LabelCensus[label]++;
        Example++;
    }
}
};

#endif // __ml_RowListUtil__

