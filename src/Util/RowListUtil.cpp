#include "ArgumentException.h"
#include <cmath>
#include <utility>
#include "matrix.h"
#include "RowListUtil.h"
#define LOG_2 0.69314718055994528622676398299518041312694549560546875

using namespace ml;
using namespace std;


void ml::TransposeMatrix(Matrix & from, Matrix & to)
{
    to.setSize(from.cols(), from.rows());
    for (int i = 0; i < from.rows(); i++)
    {
        for (int j = 0; j < from.cols(); j++)
        {
            to[j][i] = from[i][j];
        }
    }
}

void ml::GetRowList(RowList & l, Matrix & Features, 
        const std::vector<double> & Labels)
{
    l.clear();
    if (Features.rows() != Labels.size())
    {
        throw ArgumentException("Labels don't match Features given.");
    }
    for (int i = 0; i < Features.rows(); i++)
    {
        l.push_back(make_pair(&Features[i], Labels[i]));
    }
}

double ml::GetEntropy(const map<int, size_t> & LabelCensus, size_t ExampleSize)
{
    double sum = 0.0;
    for (map<int, size_t>::const_iterator LabelCount = LabelCensus.begin();
            LabelCount != LabelCensus.end();
            LabelCount++)
    {
        double proportion = ((double)LabelCount->second) / ((double)ExampleSize);
        if (proportion <= 0)
        {
            continue;
        }
        sum -= proportion * log(proportion) / LOG_2;
    }
    return sum;
}

void ml::TakeFeatureCensus(const RowList & Examples,
        map<size_t, map<double, size_t> > & FeatureCensus,
        const std::vector< std::pair<size_t, double> > & FeatureAttributes)

{
    typedef typename std::map<double,size_t>::iterator CensusIterator;
    FeatureCensus.clear();
    
    // set non-nominal census to expected, the count to 0, and 
    // otherwise zero-out the nominal census.
    for (int i = 0; i < FeatureAttributes.size(); i++)
    {
        if (FeatureAttributes[i].first != 0)
        {
            for (int j = 0; j < FeatureAttributes[i].first; j++)
            {
                FeatureCensus[i][(double)j] = 0u;
            }
        }
    }

    for (RowListCIterator Row = Examples.begin(); 
            Row != Examples.end();
            Row++)
    {
        for (int i = 0; i < FeatureAttributes.size(); i++)
        {
            if (FeatureAttributes[i].first == 0)
            {
                continue;
            }
            if (Row->first->at(i) == UNKNOWN_VALUE)
            {
                FeatureCensus[i][FeatureAttributes[i].second]++;
            }
            else
            {
                FeatureCensus[i][Row->first->at(i)]++;
            }
        }
    }
}
void ml::TakeLabelCensus(const RowList & Examples,
                         map<int, size_t> & LabelCensus, size_t LabelAttributes,
                         int LCValue)
{
    
    ml::GenericTakeLabelCensus<RowList::const_iterator, int>
        (Examples.begin(), Examples.end(), LabelCensus, LabelAttributes, 
            LCValue);
}
