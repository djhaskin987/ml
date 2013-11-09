#include "StateException.h"
#include "SigmoidFire.h"
#include "PerceptronNeuronBank.h"
#include "PerceptronLearner.h"
#include <sstream>
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

void PerceptronLearner::train(Matrix& features, Matrix& labels, 
        Matrix *testSet, Matrix *testLabels)
{
    free();
    int NumFeatures = features.cols();
    int NumOutputs = labels.cols();

    if (features.rows() != labels.rows())
    {
        stringstream ss;
        ss << "Features and Labels must be equinumerous" << " at " 
           << __LINE__ << " in " << __FILE__;
        throw std::runtime_error(ss.str());
    }

    trons = std::vector<NeuronBank*>(NumOutputs);
    for (int i = 0; i < NumOutputs; i++)
    {
        int valueCount = labels.valueCount(i) > 0 ? labels.valueCount(i) :
            1;
        trons[i] = (*factory)(NumFeatures, valueCount, &rand,
                              LearningRate, MomentumTerm);
    }
    

    features.shuffleRows(rand, &labels);

    for (int j = 0; j < labels.cols(); j++)
    {
        deque<double> window;

        int epoch = 0;
        int WindowCount = 0;
        double WindowImprovement = 0.0;
        double MSE = 0.0;
        double ImprovementImprovement = 0.0;
        double ImprovementImprovementEWMA = 0.0;
        double ImprovementEWMA = 0.0;

        while ( WindowCount < 10 )
        {
            double OldMSE = MSE;
            double OldImprovement = WindowImprovement;
            double predict = 0.0;
            int off = 0;
            MSE = 0.0;
            for (int i = 0; i < features.rows(); i++)
            {
                trons[j]->Update(features[i], labels[i][j]);
                predict = trons[j]->Predict(features[i]);
                MSE += trons[j]->MSE();
                if (predict != labels[i][j])
                {
                    off++;
                }
            }
            MSE /= (double)features.rows();
            
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
                    testPredict = trons[j]->Predict((*testSet)[i]);
                    TestMSE += trons[j]->TestMSE((*testSet)[i],
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

            if (epoch == 0) 
            {
                OldMSE = MSE;
            }

            double OldWindowImprovement = WindowImprovement;

            window.push_back(OldMSE - MSE);
           
            WindowImprovement += window.back();
           
            if (window.size() > 10)
            {
                WindowImprovement -= window.front();
                window.pop_front();
            }
            if (epoch <= 1)
            {
                OldWindowImprovement = WindowImprovement;
            }
           
            ImprovementImprovement = WindowImprovement - 
                                        OldWindowImprovement;

            ImprovementEWMA = .9 * ImprovementEWMA + 
                .1 * WindowImprovement;
            ImprovementImprovementEWMA = .9 * ImprovementImprovementEWMA +
                .1 * ImprovementImprovement;
            cout << "IImprovement: " << ImprovementImprovementEWMA 
                 << endl;
            if ( WindowImprovement < .01 &&
                    ImprovementImprovement < 0.0001)
            {
                WindowCount++;
            }
            else
            {
                WindowCount = 0;
            }

            std::cout << "MSE: " << MSE;
            std::cout << "\tEpoch: " << epoch ;
            std::cout << "\tMisclass: " << Misclassification 
                      << "\tTest MSE: " << TestMSE 
                      << "\tTest Misclass: " << testMisclassification
                << "\tEWMA: " << ImprovementEWMA 
                << std::endl;
            epoch++;
        }
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