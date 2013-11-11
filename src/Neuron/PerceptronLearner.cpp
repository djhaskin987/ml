#include "StateException.h"
#include "SigmoidFire.h"
#include "PerceptronNeuronBank.h"
#include "PerceptronLearner.h"
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <deque>

using namespace std;
using namespace ml;

PerceptronLearner::PerceptronLearner() : trons(), rand()
{
}

PerceptronLearner::PerceptronLearner(const Rand & r, double learn,
                                     double momentum, NeuronBankFactory * fact) : trons(), rand(r),
    LearningRate(learn), MomentumTerm(momentum), factory(fact), trained(false),
        _features(), _labels(), NumInputs(0)

{
}

PerceptronLearner::PerceptronLearner(const PerceptronLearner & other) :
    trons(other.trons), rand(other.rand), LearningRate(other.LearningRate),
    MomentumTerm(other.MomentumTerm), factory(other.factory), trained(other.trained),
    _features(other._features), _labels(other._labels), NumInputs(other.NumInputs)
{
    // copy(other);
}

PerceptronLearner::~PerceptronLearner()
{
    free();
}

PerceptronLearner & PerceptronLearner::operator = (
        const PerceptronLearner & other)
{
    if (&other != this)
    {
        free();
        copy(other);
    }
    return *this;
}

void PerceptronLearner::copy(const PerceptronLearner & other)
{
    trons = other.trons;
    rand = other.rand;
    LearningRate = other.LearningRate;
    MomentumTerm = other.MomentumTerm;
    factory = other.factory;
    trained = other.trained;
    _features = other._features;
    _labels = other._labels;
    NumInputs = other.NumInputs;
}

void PerceptronLearner::free()
{
    for (int i = 0; i < trons.size(); i++)
    {
        factory->Destroy(trons[i]);
    }
    trons.clear();
    trained = false;
    _features.reset();
    _labels.reset();
    NumInputs = 0;
}

shared_ptr<vector<double> >
    PerceptronLearner::getInputs(const std::vector<double> &row)
{
    shared_ptr<vector<double> > inputs(new vector<double>());
    for (int col = 0; col < _features->cols(); col++)
    {
        if (_features->valueCount(col) > 0)
        {
            int val = row[col];
            for (int valIndex = 0;
                    valIndex < _features->valueCount(col);
                    valIndex++)
            {
                double added = val == valIndex ? 1.0 : 0.0;
                inputs->push_back(added);
            }
        }
        else
        {
            double added = row[col];
            inputs->push_back(added);
        }

    }
    if (inputs->size() != NumInputs)
    {
        stringstream ss;
        ss << "Inputs not right!" << endl
           << "Inputs size: " << inputs->size() << endl
           << "  NumInputs: " << NumInputs << endl
           << "  at " << __LINE__ << " in " << __FILE__ << endl;
        throw logic_error(ss.str());
    }
    return inputs;
}

void PerceptronLearner::train(Matrix& features, Matrix& labels,
        Matrix *testSet, Matrix *testLabels)
{
    free();
    _features.reset(new Matrix(features));
    _labels.reset(new Matrix(labels));
    for (int col = 0; col < features.cols(); col++)
    {
        if (features.valueCount(col) > 0)
        {
            NumInputs += features.valueCount(col);
        }
        else
        {
            NumInputs++;
        }
    }
    if (NumInputs < features.cols())
    {
        throw logic_error("NumInputs is too small!");
    }
    int NumClasses = labels.cols();

    if (features.rows() != labels.rows())
    {
        stringstream ss;
        ss << "Features and Labels must be equinumerous" << " at "
           << __LINE__ << " in " << __FILE__;
        throw std::runtime_error(ss.str());
    }

    trons = std::vector<NeuronBank*>(NumClasses);
    for (int i = 0; i < NumClasses; i++)
    {
        int NumOutputs = labels.valueCount(i) > 0 ? labels.valueCount(i) :
            1;
        trons[i] = (*factory)(NumInputs, NumOutputs, &rand,
                              LearningRate, MomentumTerm);
    }
    cout << "\"Epoch\",\"MSE\",\"Misclass\",\"Test MSE\",\"Test Misclass\""
        << endl;

    for (int j = 0; j < labels.cols(); j++)
    {
        int epoch = 0;
        double MSE = 0.0;
        double Improvement = 0.0;
        do
        {
            features.shuffleRows(rand, &labels);
            double OldMSE = MSE;
            double predict = 0.0;
            int off = 0;
            OldMSE = MSE;
            MSE = 0.0;
            for (int i = 0; i < features.rows(); i++)
            {
                shared_ptr<vector<double> >
                    inputs = getInputs(features[i]);
                if (features.valueCount(i) <= 0)
                {
                    trons[j]->Update(*inputs, labels[i][j]);
                }
                else
                predict = trons[j]->Predict(*inputs);
                MSE += trons[j]->MSE();
                if (predict != labels[i][j])
                {
                    off++;
                }
            }
            MSE = MSE / ((double)features.rows());

            double Misclassification = ((double)off) /
                ((double)features.rows());

            double testMisclassification = 0.0;
            int testOff = 0;
            double testPredict = 0.0;
            double TestMSE = 0.0;
            if (testSet != NULL)
            {
                for (int i = 0; i < testSet->rows(); i++)
                {
                    shared_ptr<vector<double> >
                        testInputs = getInputs((*testSet)[i]);
                    testPredict = trons[j]->Predict(*testInputs);
                    TestMSE += trons[j]->TestMSE(*testInputs,
                            (*testLabels)[i][j]);

                    if (testPredict != (*testLabels)[i][j])
                    {
                        testOff++;
                    }
                }
                TestMSE /= (double)testSet->rows();
                testMisclassification = ((double)testOff) /
                ((double)testSet->rows());
            }

            Improvement = abs(OldMSE - MSE);
            cout << '"' << epoch << "\",\""
                 << MSE << "\",\""
                 << Misclassification << "\",\""
                 << TestMSE << "\",\""
                 << testMisclassification << '"' << endl;
            epoch++;
        } while (Improvement > 0.001 * LearningRate);
        std::cout << "Number of total epochs: " << epoch << std::endl;
    }
    trained = true;
}


void PerceptronLearner::predict(const std::vector<double> & features,
                                std::vector<double> & labels)
{
    if (!trained)
    {
        stringstream ss;
        ss << "need to train first" << " at " << __LINE__
           << " in " << __FILE__;
        throw StateException(ss.str());
    }

    if (labels.size() != trons.size())
    {
        throw std::runtime_error("This learner was not trained for this number of outputs.");
    }
    shared_ptr<vector<double> > inputs =
        getInputs(features);
    for (int i = 0; i < labels.size(); i++)
    {
        labels[i] = trons[i]->Predict(*inputs);
    }
}
