#include <algorithm>
#include <functional>
#include "DecisionTree.h"
#include <set>

using namespace std;
using namespace ml;

double  DecisionTree::Classify(const std::vector<double> & features)
{
    return root->Classify(features);
}

DecisionTree * DecisionTree::CreateInstance(vector<int> FAttrs, vector<int> FCValues, int LAttrs,
        int LCValue, Matrix & features, const std::vector<double> & labels)
{
    return new DecisionTree(FAttrs, FCValues, LAttrs, LCValue, features, labels);
}

void DecisionTree::RetireInstance(DecisionTree * instance)
{
    delete instance;
}

DecisionTree::DecisionTree(vector<int> FAttrs, vector<int> FCValues, int LAttrs, int LCValue, Matrix & features, const std::vector<double> & labels) : FeatureAttrs(FAttrs), LabelAttrs(LAttrs)
{
    // allow for unknown value
    //transform(FAttrs.begin(), FAttrs.end(), FAttrs.begin(), bind2nd(plus<int>(),1));
    //LAttrs++;

    RowList rows;
    for (int i = 0; i < features.rows(); i++)
    {
        rows.push_back(make_pair(&features.row(i),labels[i]));
    }

    map<int, size_t> LabelCensus;
    ml::TakeLabelCensus(rows, LabelCensus, LabelAttrs, LCValue);
    root = DTNode::CreateInstance(FAttrs, FCValues, LAttrs, LCValue, rows, 
            set<int>(), LabelCensus, GetEntropy(LabelCensus, rows.size()));
}

DecisionTree::~DecisionTree()
{
    free();
}

void DecisionTree::free()
{
    DTNode::RetireInstance(root);
}
