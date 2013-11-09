#ifndef __ml_PointerDistanceComparator__
#define __ml_PointerDistanceComparator__

#include <functional>
#include <vector>

namespace ml
{
    template <typename DistanceOperator>
    class PointerDistanceComparator : 
        std::binary_function<std::vector<double>*,
            std::vector<double>*,bool>
    {
        private:
            std::vector<double> point;
            DistanceOperator op;
            const std::vector<std::pair<std::size_t, double> > *
                FeatureAttributes;
            const std::map<size_t, std::map<double, double> > *
                NominalFeatureProbabilities;
        public:
            PointerDistanceComparator(
                const std::vector<std::pair<std::size_t, double> > *
                    FeatureAttr,
                const std::map<size_t, std::map<double, double> > *
                    probabilities,
                const std::vector<double> & pt,
                const DistanceOperator & dist) : point(pt), op(dist),
                FeatureAttributes(FeatureAttr),
                NominalFeatureProbabilities(probabilities)
            {
            }

            PointerDistanceComparator(const PointerDistanceComparator & other) :
                point(other.point), op(other.op), 
                FeatureAttributes(other.FeatureAttributes),
                NominalFeatureProbabilities(other.NominalFeatureProbabilities)
            {
            }

            PointerDistanceComparator 
                operator = (const PointerDistanceComparator & other)
            {
                if (&other != this)
                {
                    point = other.point;
                    op = other.op;
                    FeatureAttributes = other.FeatureAttributes;
                    NominalFeatureProbabilities = 
                        other.NominalFeatureProbabilities;

                }
                return *this;
            }

            ~PointerDistanceComparator()
            {
            }

            bool operator () (const std::vector<double> * a,
                    const std::vector<double> * b)
            {
                double DistanceToA = op(FeatureAttributes, 
                        NominalFeatureProbabilities, point, *a);
                double DistanceToB = op(FeatureAttributes,
                        NominalFeatureProbabilities, point, *b);
                return DistanceToA < DistanceToB;
            }
    };
}

#endif // __ml_PointerDistanceComparator__
