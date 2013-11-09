#ifndef __ml_InstanceBasedSupervisedLearner__
#define __ml_InstanceBasedSupervisedLearner__

#include "NearestNeighborSet.h"
#include "RowListUtil.h"
#include "learner.h"
#include "rand.h"
#include "matrix.h"

namespace ml
{
    class InstanceBasedSupervisedLearner : public SupervisedLearner
    {
        public:
            InstanceBasedSupervisedLearner(const Rand & r, int k, bool weight,
                    double reduction);
            virtual ~InstanceBasedSupervisedLearner();
            virtual void train(Matrix& features, Matrix& labels, 
                    Matrix *testSet, Matrix * testLabels);
            virtual void predict(const std::vector<double>& features, 
                    std::vector<double>& labels);
        private:
            Rand rand;
            int K;
            std::vector<RowList> lists;
            std::vector<NearestNeighborSet*> NNSets;
            Matrix Features;
            std::vector<std::pair<std::size_t, double> > FeatureAttributes;
            bool InverseSquare;
            double ReductionTerm;
            void free();
    };
}

#endif // __ml_InstanceBasedSupervisedLearner__
