#ifndef ml_DecisionTreeLearner_H
#define ml_DecisionTreeLearner_H

#include "learner.h"
#include <vector>
#include "rand.h"
#include "DecisionTree.h"

namespace ml
{
class DecisionTreeLearner : public SupervisedLearner
{
private:
    std::vector<DecisionTree*> trees;
    Rand rand;
    void copy(const DecisionTreeLearner & other);
    void free();

public:
    DecisionTreeLearner();
    DecisionTreeLearner(const Rand & r);
    DecisionTreeLearner(const DecisionTreeLearner & other);
    DecisionTreeLearner & operator = (const DecisionTreeLearner & other);
    virtual ~DecisionTreeLearner();
    virtual void train(Matrix& features, Matrix& labels, 
            Matrix *testSet, Matrix * testLabels);
    virtual void predict(const std::vector<double>& features, std::vector<double>& labels);
};
}
#endif // ml_DecisionTreeLearner_H
