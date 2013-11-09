#ifndef __ml_NearestNeighborSet__
#define __ml_NearestNeighborSet__
#include <cstdlib>
#include "VectorLessPointerShuffle.h"
#include "RowListUtil.h"
#include "rand.h"

namespace ml
{
    class NearestNeighborSet
    {
        public:
            static NearestNeighborSet * CreateInstance(
                const RowList & rows, const Rand & r, std::size_t k,
                const std::vector<std::pair<std::size_t, double> > & FeatureAttr,
                const std::map<std::size_t, std::map<double, double> > &
                    probabilities,
                std::size_t LabelAttributes, double MostCommonLabel, 
                bool weight,
                double reduction);

            static void RetireInstance(NearestNeighborSet * instance);
            double GetNearestVote(const std::vector<double> & features);
        private:
            void free();
            void GetKNearest(RowList & Records, 
                    std::vector<double> & features, std::size_t k);
            void DeleteEntry(std::vector<double> * features); 
            virtual ~NearestNeighborSet();
            NearestNeighborSet(const RowList & rows, 
                const Rand & r, std::size_t k, 
                const std::vector<std::pair<std::size_t, double> > & FeatureAttr,
                const std::map<std::size_t, std::map<double, double> > &
                    probabilities,
                std::size_t LabelAttributes, double MostCommonLabel, 
                bool weight, 
                double reduction);
            RowList Rows;
            Rand rand;
            std::size_t K;
            std::vector<std::pair<std::size_t, double> > FeatureAttributes;
            std::map<std::size_t, std::map<double, double> > 
                NominalFeatureProbabilities;
            std::size_t LabelAttr;
            std::size_t CommonLabel;
            std::vector<
                std::map<std::vector<double>*,double, 
                    VectorLessPointerShuffle<double> > >
                        Indexes;
            bool InverseSquare;
            double ReductionTerm;
    };
}

#endif // __ml_NearestNeighborSet__
