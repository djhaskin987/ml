#ifndef ml_PerceptronLearner_H
#define ml_PerceptronLearner_H

#include "learner.h"
#include <vector>
#include <memory>
#include "NeuronBank.h"
#include "NeuronBankFactory.h"
#include "rand.h"
namespace ml
{
class PerceptronLearner : public SupervisedLearner
{
private:
    std::vector<NeuronBank*> trons;
    double LearningRate;
    double MomentumTerm;
    Matrix * _features;
    Matrix * _labels;
    int NumInputs;
    Rand rand;
    NeuronBankFactory * factory;
    bool trained;
    void copy(const PerceptronLearner & other);
    void free();
    std::shared_ptr<std::vector<double> >
        getInputs(Matrix & matrix, const std::vector<double> & row,
        int NumInputs);

public:
    PerceptronLearner();
    PerceptronLearner(const Rand & r, double learn, double mometum,
                      NeuronBankFactory * fact);
    PerceptronLearner(const PerceptronLearner & other);
    PerceptronLearner & operator = (const PerceptronLearner & other);
    virtual ~PerceptronLearner();
    virtual void train(Matrix& features, Matrix& labels, Matrix *testSet,
            Matrix *testLabels);
    virtual void predict(const std::vector<double>& features, std::vector<double>& labels);
};
}
#endif // ml_PerceptronLearner_H
