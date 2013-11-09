#ifndef ml_PerceptronLearner_H
#define ml_PerceptronLearner_H

#include "learner.h"
#include <vector>
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
    Rand rand;
    NeuronBankFactory * factory;
    bool trained;
    void copy(const PerceptronLearner & other);
    void free();

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
