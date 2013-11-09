#include "StateException.h"
#include "InstanceBasedSupervisedLearner.h"
#include "RowListUtil.h"
#include <vector>
#include <utility>

using namespace ml;
using namespace std;

void  InstanceBasedSupervisedLearner::train(Matrix& features, 
        Matrix& labels, Matrix *testSet, Matrix * testLabels)
{
    free();
    
    // copy the matrix so pointers to their rows won't segfault (ensure the 
    // features matrix is persistent by keeping it within the learner itself)
    Features = Matrix(features);
    Matrix Labels(labels);
    Labels.copyPart(labels, 0, 0, 0, labels.cols());
    Features.copyPart(features, 0, 0, 0, features.cols());

    // reduce
    for (int i = 0; i < features.rows(); i++)
    {
        double dice = rand.uniform();
        if (dice >= ReductionTerm)
        {
            Features.addRow(features[i]);
            Labels.addRow(labels[i]);
        }
    }

    Features.shuffleRows(rand, &Labels);


    // gather data about features to pass it to the learner.
    FeatureAttributes.resize(Features.cols());
    for (int i = 0; i < FeatureAttributes.size(); i++)
    {
        size_t count = Features.valueCount(i);
        double expected = count == 0 ? Features.columnMean(i) : 
            Features.mostCommonValue(i);
        FeatureAttributes[i] = make_pair(count, expected);
    }


    Matrix TransposedLabels;
    TransposeMatrix(Labels, TransposedLabels);
    lists.resize(TransposedLabels.rows());
    NNSets.resize(TransposedLabels.rows());

    GetRowList(lists[0], Features, TransposedLabels[0]);
    
    map<size_t, map<double, size_t> > FeatureCensus;
    TakeFeatureCensus(lists[0], FeatureCensus, FeatureAttributes);
    map<size_t, map<double, double> > NominalProbabilities;
    for (map<size_t, map<double, size_t> >::iterator 
            FeatureVoteCount = FeatureCensus.begin();
            FeatureVoteCount != FeatureCensus.end();
            FeatureVoteCount++)
    {
        NominalProbabilities[FeatureVoteCount->first] = map<double, double>();
        for (map<double, size_t>::iterator 
                Counts = (*FeatureVoteCount).second.begin();
                Counts != (*FeatureVoteCount).second.end();
                Counts++)
        {
            NominalProbabilities[FeatureVoteCount->first][Counts->first] = 
                (double)Counts->second / (double)Features.rows();
        }
    }
    NNSets[0] = NearestNeighborSet::CreateInstance(lists[0], rand, K, 
            FeatureAttributes, NominalProbabilities, Labels.valueCount(0), 
            Labels.mostCommonValue(0), InverseSquare, ReductionTerm);
    for (int i = 1; i < TransposedLabels.rows(); i++)
    {
        map<size_t, map<double, size_t> > FeatureCensus;
        GetRowList(lists[i], Features, TransposedLabels[i]);
        TakeFeatureCensus(lists[i], FeatureCensus, FeatureAttributes);
        vector<double> NominalFeatureProbabilities;
        NominalFeatureProbabilities.resize(FeatureAttributes.size());
        NNSets[i] = NearestNeighborSet::CreateInstance(lists[i], rand, K, 
                FeatureAttributes, NominalProbabilities, Labels.valueCount(i), 
                Labels.mostCommonValue(i), InverseSquare, ReductionTerm);
    }
}

void  InstanceBasedSupervisedLearner::predict(
        const std::vector<double>& features, std::vector<double>& labels)
{
    if (NNSets.size() != labels.size())
    {
        throw StateException("This is rediculous.");
    }

    for (int i = 0; i < labels.size(); i++)
    {
        labels[i] = NNSets[i]->GetNearestVote(features);
    }
}

InstanceBasedSupervisedLearner::InstanceBasedSupervisedLearner(const Rand & r, 
    int k, bool weight, double reduction) : rand(r), K(k), lists(), NNSets(), 
    Features(), FeatureAttributes(), InverseSquare(weight), 
    ReductionTerm(reduction)
{
    // FIXME: use TakeFeatureCensus to calculate probablities and pass it on to 
    // the NearestNeighborSet for distance purposes.
}

InstanceBasedSupervisedLearner::~InstanceBasedSupervisedLearner()
{
    free();
}

void InstanceBasedSupervisedLearner::free()
{
    for (int i = 0; i < lists.size(); i++)
    {
        lists[i].clear();
        NearestNeighborSet::RetireInstance(NNSets[i]);
    }
    lists.clear();
    NNSets.clear();
    Features.setSize(0,0);
    FeatureAttributes.clear();
}
