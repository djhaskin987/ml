#ifndef __ml_DTNode__
#define __ml_DTNode__

#include "RowListUtil.h"
#include <set>
namespace ml
{
class DTNode
{
public:
    static DTNode * CreateInstance(const std::vector<int> & FAttrs,
                                   std::vector<int> FCValues,
                                   const int & LAttrs, int LCValue, RowList examples,
                                   std::set<int> attr,
                                   std::map<int, std::size_t> cens, double ent);
    static void RetireInstance(DTNode * instance);

    double Classify(const std::vector<double> & features) const;
    double Decision() const;
    double Entropy() const;

    DTNode(const DTNode & other);
    DTNode & operator = (const DTNode & other);
    virtual ~DTNode();
private:
    explicit DTNode(const std::vector<int> & FAttrs,
                    std::vector<int> FCValues,
                    const int & LAttrs, int LCValue,
                    RowList examples,
                    std::set<int> attr,
                    std::map<int, std::size_t> cens, double ent);

    void copy(const DTNode & other);
    void free();
    RowList RepresentedExamples;
    double decision;
    double entropy;
    const std::vector<int> & FeatureAttrs;
    std::vector<int> FeatureCommonValues;
    const int & LabelAttrs;
    int LabelCommonValue;
    std::vector<DTNode*> children;
    std::set<int> UsedAttributes;
    int SplitAttr;
    double MaxGainAttribute(int & Attribute,
                            std::vector<RowList> & Partitions,
                            std::vector<std::map<int, std::size_t> > & LabelCensa,
                            std::vector<double> & Entropies );
};
}

#endif // __ml_DTNode__
