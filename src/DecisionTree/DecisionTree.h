#ifndef __ml_DecisionTree__
#define __ml_DecisionTree__

#include <vector>
#include "matrix.h"
#include "DTNode.h"

namespace ml
{
class DecisionTree
{
public:
    static void RetireInstance(DecisionTree * instance);
    static DecisionTree *
    CreateInstance(std::vector<int> FAttrs, std::vector<int> FCValues, int LAttrs,
                   int LCValue, Matrix & features,
                   const std::vector<double> & labels);
    double Classify(const std::vector<double> & features);

    /*! \brief performs a deep copy.
     *  \param other DecisionTree to copy.
     */
    DecisionTree(const DecisionTree & other);
    /*! \brief clears the contents of the this DecisionTree
     *      instance and performs a deep copy,
     *      provided that \emph{other} is not the same object (address-wise).
     *  \param other object to copy.
     */
    DecisionTree & operator = (const DecisionTree & other);
    virtual ~DecisionTree();
private:
    explicit DecisionTree(std::vector<int> FAttrs, std::vector<int> FCValues, int LAttrs,
                          int LCValue, Matrix & features, const std::vector<double> & labels);
    void copy(const DecisionTree & other);
    void free();
    DTNode * root;
    std::vector<int> FeatureAttrs;
    int LabelAttrs;
};
}

#endif // __ml_DecisionTree__
