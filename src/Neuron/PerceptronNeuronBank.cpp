#include "StateException.h"
#include "Perceptron.h"
#include "PerceptronNeuronBank.h"
#include "rand.h"
#include <cmath>
#include <set>
#include <limits>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <utility>

using namespace std;
using namespace ml;

typedef std::vector<std::pair<Neuron*,std::pair<int,int> > >
TronType;

PerceptronNeuronBank::PerceptronNeuronBank(int NumInputs, int NumOutputs, Rand * r,
        double LearningRate, double MomentumTerm,
        std::vector<std::pair<Neuron*,std::pair<int,int> > > ts) :
    learning_rate(LearningRate), momentum_term(MomentumTerm), rand(r),
    inputs(NumInputs), OutputValues(NumOutputs), trons(ts), Count(0),
    Error(0.0), deltas()
{
}

PerceptronNeuronBank::~PerceptronNeuronBank()
{
    free();
}

void PerceptronNeuronBank::free()
{
}

void PerceptronNeuronBank::Update(const std::vector<double> & inputs, double target)
{
    if (Count >= Window())
    {
        return;
    }
    double difference = Error;
    size_t TargetTron = (size_t) target;
    double actual;
    double offset;
    double delta;
    Error = 0.0;
    deltas.clear();
    deltas.assign(trons.size(), 0.0);
    std::vector<double>::iterator d = deltas.begin();
    TronType::iterator i = trons.begin();
    actual = i->first->Fire(inputs);
    if (i->second.first == -1 && i->second.second == -1)
    {
        // continuous output
        offset = target - i->first->Fire(inputs);
        Error += offset*offset;
        delta = offset * actual * (1.0 - actual);
        *d = delta;
        i->first->Update(inputs,delta);
    }
    else
    {
        for ( ; i != trons.end(); i++)
        {
            actual = i->first->Fire(inputs);
            if (i->second.first == TargetTron)
            {
                offset = 0.0 - i->first->Fire(inputs);
                Error += offset*offset;
                delta = offset * actual * (1.0 - actual);
                *d = delta;
                i->first->Update(inputs,delta);
            }
            else if (i->second.second = TargetTron)
            {
                offset = 1.0 - i->first->Fire(inputs);
                Error += offset*offset;
                delta = offset * actual * (1.0 - actual);
                *d = delta;
                i->first->Update(inputs,delta);
            }
            d++;
        }
    }

    Error /= (double) trons.size();
    if (fabs(difference - Error) < .05)
    {
        Count++;
        if (Count >= Window())
        {
            return;
        }
    }
    else
    {
        Count = 0;
    }
}

double PerceptronNeuronBank::MSE() const
{
    return Error;
}

void PerceptronNeuronBank::Deltas(std::vector<double> & ds) const
{
    if (deltas.size() < 1)
    {
        throw StateException("Must call update first.");
    }
    ds = deltas;
}

const Neuron * PerceptronNeuronBank::Tron(int index) const
{
    return trons[index].first;
}

const size_t PerceptronNeuronBank::TronSize() const
{
    return trons.size();
}

void PerceptronNeuronBank::VoteForMe(std::vector<int> & votes, size_t & index, int vote) const
{
    votes[vote]++;
    if (votes[vote] > votes[index])
    {
        index = vote;
    }
}

double PerceptronNeuronBank::Predict(const std::vector<double> & inputs) const
{
    size_t index = 0;
    double current;
    std::vector<int> votes(OutputValues,0);
    std::vector<std::pair<Neuron*, std::pair<int,int> > >::const_iterator i = trons.begin();

    current = i->first->Fire(inputs);

    if (i->second.first == -1 && i->second.second == -1)
    {
        // continuous
        return current;
    }

    for ( ; i != trons.end(); i++)
    {
        current = i->first->Fire(inputs);
        if (current < 0.5)
        {
            VoteForMe(votes, index, i->second.first);
        }
        else
        {
            VoteForMe(votes, index, i->second.second);
        }
    }

    std::vector<int> candidates;
    for (int i = 0; i < votes.size(); i++)
    {
        if (votes[i] >= votes[index])
        {
            candidates.push_back(i);
        }
    }

    if (candidates.size() == 1)
    {
        return (double) index;
    }

    std::vector<double> CandidateScore(candidates.size(), 0.0);
    double MaxVal = -std::numeric_limits<double>::infinity();
    size_t MaxValIndex = -1;
    double CurrentVal;
    for (int i = 0; i < candidates.size(); i++)
    {
        for (TronType::const_iterator j = trons.begin(); j != trons.end(); j++)
        {
            if ((j->second).second == candidates[i])
            {
                CandidateScore[i] += j->first->Vote(inputs);
            }
            else if ((j->second).first == candidates[i])
            {
                CandidateScore[i] -= j->first->Vote(inputs);
            }

            CurrentVal = CandidateScore[i];
            if (CurrentVal > MaxVal)
            {
                MaxVal = CurrentVal;
                MaxValIndex = i;
            }
        }
    }

    return (double) candidates[MaxValIndex];
}

int PerceptronNeuronBank::Window() const
{
    return WINDOW;
}

bool PerceptronNeuronBank::Trained() const
{
    return Count >= Window();
}
