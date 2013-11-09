#include <limits>
#include <functional>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "ArgumentException.h"
#include <vector>
#include <iterator>
#include "NearestNeighborSet.h"
#include "PointerDistanceComparator.h"
#include <cassert>

using namespace ml;
using namespace std;

typedef vector<map<vector<double>*,double, 
        VectorLessPointerShuffle<double> > >  
        IndexesType;

NearestNeighborSet * NearestNeighborSet::CreateInstance(
                    const RowList & rows, const Rand & r, std::size_t k,
                    const vector<pair<size_t, double> > & FeatureAttr, 
                    const std::map<std::size_t, std::map<double, double> > &
                        probabilities,
                    size_t LabelAttributes, double MostCommonLabel, bool weight,
                    double reduction)
{
    return new NearestNeighborSet(rows, r, k, FeatureAttr, probabilities, 
            LabelAttributes, MostCommonLabel, weight, reduction);
}

void NearestNeighborSet::RetireInstance(NearestNeighborSet * instance)
{
    delete (NearestNeighborSet*) instance;
}

double square(double a)
{
    return a*a;
}

double abs_val(double a)
{
    return a < 0.0 ? -a : a;
}

void GetDifference(
        vector<double> & Difference,
        const vector<pair<size_t, double> > * FeatureAttributes,
        const map<size_t, map<double, double> > * NominalFeatureProbabilities,
        const vector<double> & a,
        const vector<double> & b)
{
    for (int i = 0; i < a.size(); i++)
    {
        double a_val = (a)[i] == UNKNOWN_VALUE ? 
            (*FeatureAttributes)[i].second : 
            (a)[i];
        double b_val = (b)[i] == UNKNOWN_VALUE ? 
            (*FeatureAttributes)[i].second : 
            (b)[i];
        if (a_val == UNKNOWN_VALUE)
        {
            cout << "A LIVE ONE!" << endl;
        }
        if (b_val == UNKNOWN_VALUE)
        {
            cout << "My father was a piano mover!" << endl;
        }

        double a_compare = (*FeatureAttributes)[i].first > 0 ?
     (*(*(*NominalFeatureProbabilities).find(i)).second.find(a_val)).second :
            a_val;
        double b_compare = (*FeatureAttributes)[i].first > 0 ?
     (*(*(*NominalFeatureProbabilities).find(i)).second.find(b_val)).second :
            b_val;

        Difference[i] = b_compare - a_compare;
    }
}


double ManhattanDistance(
        const vector<pair<size_t, double> > * FeatureAttributes,
        const map<size_t, map<double, double> > * NominalFeatureProbabilities,
        const vector<double> & a,
        const vector<double> & b)
{
    if (a.size() != b.size())
    {
        throw ArgumentException("must be equal lengths");
    }
    
    vector<double> Difference(a.size());
    GetDifference(Difference, FeatureAttributes, NominalFeatureProbabilities,
            a, b);
    transform(Difference.begin(), Difference.end(), Difference.begin(), 
            ptr_fun(abs_val));
    return accumulate(Difference.begin(), Difference.end(), 0.0);
}

double EuclideanDistance(
        const vector<pair<size_t, double> > * FeatureAttributes,
        const map<size_t, map<double, double> > * NominalFeatureProbabilities,
        const vector<double> & a,
        const vector<double> & b)
{
    if (a.size() != b.size())
    {
        throw ArgumentException("must be equal lengths");
    }
    
    vector<double> Difference(a.size());
    GetDifference(Difference, FeatureAttributes, NominalFeatureProbabilities,
            a, b);
    transform(Difference.begin(), Difference.end(), Difference.begin(), square);
    return sqrt(accumulate(Difference.begin(), Difference.end(), 0.0));
}


typedef 
    map<vector<double>*,double, 
        PointerDistanceComparator< double(&) (
                const vector<pair<size_t, double> > * ,
                const map<size_t, map<double, double> > * ,
                const vector<double> &,
            const vector<double> &)  > >
        SortedDistanceList;

void NearestNeighborSet::GetKNearest(RowList & Records,
        vector<double> & features, size_t k)
{
    // find the k-nearest points.
    PointerDistanceComparator< double(&) (
            const vector<pair<size_t, double> > * ,
            const map<size_t, map<double, double> > * ,
            const vector<double> &,
            const vector<double> &) > comparator(
               &FeatureAttributes, &NominalFeatureProbabilities, features, 
               ManhattanDistance);
    SortedDistanceList SortedList(comparator);
    SortedList.clear();

    for (IndexesType::iterator Index = Indexes.begin();
            Index != Indexes.end(); Index++)
    { 
        map<vector<double>*,double>::iterator 
            start = Index->find(&features);

        if (start == Index->end())
        {
            start = (Index->insert(make_pair(&features,0.0))).first;
            Index->erase(start);
        }
              
        map<vector<double>*,double>::iterator 
            stop = start;
    
        size_t perimeter = k;
        for (int s = 0; s < perimeter; s++)
        {
            if (start != Index->begin())
            {
                start--;
            }
    
            if (stop != Index->end())
            {
                stop++;
            }
        }
        SortedList.insert(start,stop);
    }
    SortedDistanceList::iterator stop = SortedList.begin();
    
    int count = 0;
    for (; stop != SortedList.end() && count < k; stop++)
    {
        count++;
    }
    Records.clear();
    Records.insert(Records.begin(), SortedList.begin(), stop);
}

void NearestNeighborSet::DeleteEntry( std::vector<double> * features)
{
    for (IndexesType::iterator Index = Indexes.begin();
            Index != Indexes.end();
            Index++)
    {
        Index->erase(features);
    }
}

double NearestNeighborSet::GetNearestVote(const std::vector<double> & features)
{
    vector<double> Features(features);
    size_t FeatureSize = features.size();
    
    RowList Records;
    GetKNearest(Records, Features, K);

    if (LabelAttr == 0)
    {
        double sum = 0.0; 
        for (RowListCIterator Record = Records.begin();
                Record != Records.end(); Record++)
        {
            sum += Record->second;
        }

        return sum / (double)K;
    }
    else
    {

        if (!InverseSquare)
        {
            map<double, size_t> LabelCensus;
            ml::GenericTakeLabelCensus<RowListIterator, double> (
                    Records.begin(), Records.end(), LabelCensus, LabelAttr,
                    CommonLabel);
            double val = max_element(LabelCensus.begin(), LabelCensus.end(), 
                    &ml::CensusCompare<double, size_t>)->first;
            return val;
        }
        else
        {
            map<double, double> LabelCensus;

            for (RowListCIterator::iterator instance = Records.begin();
                    instance != Records.end(); 
                    instance++)
            {
                double denominator = square(EuclideanDistance(
                        &FeatureAttributes,
                        &NominalFeatureProbabilities,
                        (*(instance->first)), features));
                double value;
                if (denominator == 0)
                {
                    value = numeric_limits<double>::infinity();
                }
                else
                {
                    value = 1.0 / denominator;
                }
                LabelCensus[instance->second] += value;
            }
            double val = max_element(LabelCensus.begin(), LabelCensus.end(), 
                    &ml::CensusCompare<double,double>)->first;
            return val;
        }
    }
}

NearestNeighborSet::NearestNeighborSet(const RowList & rows, const Rand & r,
    size_t k, const vector<pair<size_t, double> > & FeatureAttr, 
    const std::map<std::size_t, std::map<double, double> > & probabilities,
    size_t LabelAttributes, double MostCommonLabel, bool weight,
    double reduction) : 
    rand(r), K(k), FeatureAttributes(FeatureAttr), 
    NominalFeatureProbabilities(probabilities), LabelAttr(LabelAttributes), 
    CommonLabel(MostCommonLabel), Indexes(), InverseSquare(weight),
    ReductionTerm(reduction)
{
    int i = 0; 
    Indexes.clear();
    for (int i = 0; i < K; i++)
    {
        Indexes.push_back(map<vector<double>*,double,
                VectorLessPointerShuffle<double> >(
                    VectorLessPointerShuffle<double>(
                        &FeatureAttributes, &NominalFeatureProbabilities, i)));
        for (RowListCIterator row = rows.begin(); 
                row != rows.end();
                row++)
        {
            Indexes[i].insert(*row);
        }

    }
}

NearestNeighborSet::~NearestNeighborSet()
{
    free();
}

void NearestNeighborSet::free()
{
}
