#include "DecisionTreeLearner.h"
#include "RowListUtil.h"
#include <stdexcept>

#include <vector>
#include <sstream>
using namespace std;
using namespace ml;

typedef vector<DecisionTree*>::iterator DTIterator;
typedef vector<DecisionTree*>::const_iterator DTCIterator;
typedef vector<DecisionTree*>::reverse_iterator DTRIterator;

void DecisionTreeLearner::copy(const DecisionTreeLearner & other)
{
    trees = other.trees;
    rand = other.rand;
}

void DecisionTreeLearner::free()
{
    for (DTIterator i = trees.begin(); i != trees.end(); i++)
    {
        DecisionTree::RetireInstance(*i);
    }
    trees.clear();
}

DecisionTreeLearner::DecisionTreeLearner() :
    trees(), rand()
{
}

DecisionTreeLearner::DecisionTreeLearner(const Rand & r) : trees(), rand(r)
{
}


DecisionTreeLearner::DecisionTreeLearner(const DecisionTreeLearner & other) : trees(other.trees), rand(other.rand)
{
}

DecisionTreeLearner & DecisionTreeLearner::operator = (const DecisionTreeLearner & other)
{
    if (&other != this)
    {
        free();
        copy(other);
    }
    return *this;
}

DecisionTreeLearner::~DecisionTreeLearner()
{
    free();
}


void DecisionTreeLearner::train(Matrix & features, Matrix & labels, 
        Matrix *testSet, Matrix * testLabels)
{
    free();

    if (features.rows() != labels.rows())
    {
        stringstream ss;
        ss << "Features and Labels must be equinumerous" << " at " << __LINE__
           << " in " << __FILE__;
        throw std::runtime_error(ss.str());
    }
    std::vector<int> FeatureAttrs;
    FeatureAttrs.resize(features.cols());
    std::vector<int> FeatureCommonValues;
    FeatureCommonValues.resize(features.cols());
    for (int i = 0; i < FeatureAttrs.size(); i++)
    {
        FeatureCommonValues[i] = features.mostCommonValue(i);
        FeatureAttrs[i] = features.valueCount(i);
    }

    trees.resize(labels.cols());
    Matrix TurnedLabels;
    TransposeMatrix(labels, TurnedLabels);

    for (int i = 0; i < trees.size(); i++)
    {
        trees[i] = DecisionTree::CreateInstance(FeatureAttrs, FeatureCommonValues,
                                                labels.valueCount(i), labels.mostCommonValue(i),
                                                features, TurnedLabels[i]);
    }
}

void DecisionTreeLearner::predict(const std::vector<double> & features, std::vector<double> & labels)
{
    for (int i = 0; i < trees.size();
            i++)
    {
        labels[i] = (trees[i])->Classify(features);
    }
}

