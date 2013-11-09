#include "matrix.h"
#include "RowListUtil.h"
#include "StateException.h"
#include <iostream>
#include "DTNode.h"
#include <algorithm>
#include "ArgumentException.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <map>
#include <limits>
using namespace ml;
using namespace std;

DTNode * DTNode::CreateInstance(const vector<int> & FAttrs,
                                vector<int> FCValues,
                                const int & LAttrs, int LCValue, RowList examples,
                                set<int> attr,
                                map<int, size_t> cens, double ent)
{
    return new DTNode(FAttrs, FCValues, LAttrs, LCValue, examples, attr, cens,
                      ent);
}


void DTNode::RetireInstance(DTNode * instance)
{
    delete instance;
}

double  DTNode::Classify(const vector<double> & features) const
{
    if (children.size() == 0)
    {
        return Decision();
    }

    size_t ChildIndex;
    if (features[SplitAttr] == UNKNOWN_VALUE)
    {
        ChildIndex = FeatureCommonValues[SplitAttr];
    }
    else
    {
        ChildIndex = (size_t)features[SplitAttr];
    }

    if (children[ChildIndex] == NULL)
    {
        return Decision();
    }
    else
    {
        return children[ChildIndex]->Classify(features);
    }
}

double DTNode::Decision() const
{
    return decision;
}

double DTNode::Entropy() const
{
    return entropy;
}


double DTNode::MaxGainAttribute(int & Attribute, vector<RowList> & Partitions,
                                vector<map<int, size_t> > & LabelCensa, vector<double> & Entropies )
// assumes decision, entropy, RepresentedExamples, FeatureAttr, and LabelAttr
// are all properly set.
{

    double CurrentGain;
    double MaxGain = -numeric_limits<double>::infinity();

    Attribute = -1;

    Partitions.clear();
    LabelCensa.clear();
    Entropies.clear();


    for (int attr = 0; attr < FeatureAttrs.size(); attr++)
    {
        vector<map<int, size_t> > CurrentLabelCensa;
        vector<double> CurrentEntropies;
        vector<RowList> CurrentPartitions;
        CurrentEntropies.resize(FeatureAttrs[attr]);
        CurrentLabelCensa.resize(FeatureAttrs[attr]);
        CurrentPartitions.resize(FeatureAttrs[attr]);
        for (RowList::iterator Example = RepresentedExamples.begin();
                Example != RepresentedExamples.end();
                Example++)
        {
            int value;
            if (Example->first->at(attr) == UNKNOWN_VALUE)
            {
                value = FeatureCommonValues[attr];
            }
            else
            {
                value = (int)Example->first->at(attr);
            }
            assert(value < FeatureAttrs[attr]);
            CurrentPartitions[value].push_back(*Example);
        }

        double sum = 0.0;
        for (int i = 0; i < FeatureAttrs[attr]; i++)
        {
            if (CurrentPartitions[i].size() > 0)
            {
                map<int, size_t> LabelCensus;
                TakeLabelCensus(CurrentPartitions[i], LabelCensus, LabelAttrs, LabelCommonValue);
                CurrentLabelCensa[i] = LabelCensus;
                CurrentEntropies[i] = GetEntropy(LabelCensus, CurrentPartitions[i].size());
                sum += ((double)(CurrentPartitions[i].size())) /
                       ((double)RepresentedExamples.size()) *
                       CurrentEntropies[i];
            }
        }
        CurrentGain = entropy - sum;
        if (CurrentGain > MaxGain && UsedAttributes.count(attr) == 0u)
        {
            MaxGain = CurrentGain;
            Attribute = attr;
            Partitions = CurrentPartitions;
            LabelCensa = CurrentLabelCensa;
            Entropies = CurrentEntropies;
        }
    }
}



DTNode::DTNode(const vector<int> & FAttrs, vector<int> FCValues,
               const int & LAttrs, int LCValue, RowList examples,
               set<int> attr,
               map<int, size_t> lcens, double ent) :
    RepresentedExamples(examples), decision(0.0), entropy(ent), FeatureAttrs(FAttrs),
    FeatureCommonValues(FCValues), LabelAttrs(LAttrs), LabelCommonValue(LCValue),
    children(), UsedAttributes(attr), SplitAttr(-1)
{
    // maps label id to number of times it was used
    children.clear();
    map<int, size_t> LabelCensus = lcens;
    if (LabelCensus.size() == 0)
    {
        throw ArgumentException("non-trivial census and entropy needed");
    }
    map<int, size_t>::iterator iter;

    // get the most common label
    iter = max_element(LabelCensus.begin(), LabelCensus.end(), 
            &ml::CensusCompare<int, size_t>);
    // set baseline to most common label
    decision = (double)iter->first;
    // if it's a homogeneous set, we're done.
    if (iter->second == RepresentedExamples.size())
    {
        return;
    }
    // if we've already used up all our attributes, we're done.
    else if (UsedAttributes.size() == FeatureAttrs.size())
    {
        return;
    }
    else if (UsedAttributes.size() > FeatureAttrs.size())
    {
        throw StateException("This should NEVER happen.");
    }

    // calculate info gain for each unused attribute
    vector<RowList> Partitions;
    vector<map<int, size_t> > LabelCensa;
    vector<double> Entropies;
    MaxGainAttribute(SplitAttr, Partitions, LabelCensa, Entropies);
    vector<map<int, size_t> >::iterator ChildLabelCensus = LabelCensa.begin();
    vector<double>::iterator ChildEntropy = Entropies.begin();
    children.resize(Partitions.size());
    vector<DTNode*>::iterator Child = children.begin();
    for (vector<RowList>::iterator Partition = Partitions.begin();
            Partition != Partitions.end(); Partition++)
    {
        if (Partition->size() > 0)
        {
            set<int> ChildUsedAttributes(UsedAttributes);
            ChildUsedAttributes.insert(SplitAttr);
            *Child = DTNode::CreateInstance(FeatureAttrs, FeatureCommonValues, LabelAttrs,
                                            LabelCommonValue, *Partition, ChildUsedAttributes, *ChildLabelCensus, *ChildEntropy);
        }
        else
        {
            *Child = NULL;
        }
        Child++;
        ChildLabelCensus++;
        ChildEntropy++;
    }
    //cout << "Split on : " << SplitAttr << endl;
}

DTNode::~DTNode()
{
    free();
}

void DTNode::free()
{
    for (vector<DTNode*>::iterator Child = children.begin();
            Child != children.end();
            Child++)
    {
        if (*Child != NULL)
        {
            DTNode::RetireInstance(*Child);
        }
    }
}
