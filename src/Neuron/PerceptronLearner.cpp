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
    LearningRate(learn), MomentumTerm(momentum), factory(fact), trained(false)
{
}

PerceptronLearner::PerceptronLearner(const PerceptronLearner & other) :
    trons(other.trons), rand(other.rand), LearningRate(other.LearningRate),
    MomentumTerm(other.MomentumTerm), factory(other.factory), trained(other.trained)
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
}

void PerceptronLearner::free()
{
    for (int i = 0; i < trons.size(); i++)
    {
        factory->Destroy(trons[i]);
    }
    trons.clear();
    trained = false;
}

shared_ptr<vector<double> >
    PerceptronLearner::getInputs(Matrix & matrix, int row)
{
    shared_ptr<vector<double> > inputs(new vector<double>());
    for (int col = 0; col < matrix.cols(); col++)
    {
        if (matrix.valueCount(row) <= 0)
        {
            double added = matrix[row][col];
            inputs->push_back(matrix[row][col]);
        }
        else
        {
            int val = round(matrix[row][col]);
            for (int valIndex = 0;
                    valIndex < matrix.valueCount(col);
                    valIndex++)
            {
                double added = val == valIndex ? 1.0 : 0.0;
                inputs->push_back(added);
            }
        }
    }
    return inputs;
}


void PerceptronLearner::train(Matrix& features, Matrix& labels,
        Matrix *testSet, Matrix *testLabels)
{
    free();

    int NumInputs = 0;
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
        int valueCount = labels.valueCount(i) > 0 ? labels.valueCount(i) :
            1;
        trons[i] = (*factory)(NumInputs, valueCount, &rand,
                              LearningRate, MomentumTerm);
    }

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
                    inputs = getInputs(features, i);
                if (features.valueCount(i) <= 0)
                {
                    trons[j]->Update(*inputs, labels[i][j]);
                }
                else
                predict = trons[j]->Predict(*inputs);
                MSE += trons[j]->MSE();
                if (round(predict) != round(labels[i][j]))
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
                        testInputs = getInputs(*testSet, i);
                    testPredict = trons[j]->Predict(*testInputs);
                    TestMSE += trons[j]->TestMSE(*testInputs,
                            (*testLabels)[i][j]);

                    if (round(testPredict) != round((*testLabels)[i][j]))
                    {
                        testOff++;
                    }
                }
                TestMSE /= (double)testSet->rows();
                testMisclassification = ((double)testOff) /
                ((double)testSet->rows());
            }

            Improvement = abs(OldMSE - MSE);

            std::cout << "MSE: " << MSE;
            std::cout << "\tEpoch: " << epoch ;
            std::cout << "\tMisclass: " << Misclassification
                      << "\tTest MSE: " << TestMSE
                      << "\tTest Misclass: " << testMisclassification
                << std::endl;
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

    for (int i = 0; i < labels.size(); i++)
    {
        labels[i] = trons[i]->Predict(features);
    }
}
