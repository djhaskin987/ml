#include "BackPropNeuronBank.h"
#include "NotImplementedException.h"
#include "ArgumentException.h"
#include "SigmoidFire.h"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace ml;

typedef std::vector<std::vector<Neuron*> >::reverse_iterator BankRackRIterator;
typedef std::vector<Neuron*>::reverse_iterator BankRowRIterator;
typedef std::vector<double>::reverse_iterator RowRIterator;
typedef std::vector<std::vector<double> >::reverse_iterator RackRIterator;

typedef std::vector<std::vector<Neuron*> >::const_iterator BankRackCIterator;
typedef std::vector<Neuron*>::const_iterator BankRowCIterator;
typedef std::vector<double>::const_iterator RowCIterator;
typedef std::vector<std::vector<double> >::const_iterator RackCIterator;

typedef std::vector<std::vector<Neuron*> >::iterator BankRackIterator;
typedef std::vector<Neuron*>::iterator BankRowIterator;
typedef std::vector<double>::iterator RowIterator;
typedef std::vector<std::vector<double> >::iterator RackIterator;


template <class InputIterator, class Type, class BinaryOperator>
Type Reduce(InputIterator first, InputIterator last,
            BinaryOperator reducer, Type seed = Type())
{
    Type returned = seed;
    while (first != last)
    {
        returned = reducer(returned, *first++);
    }
    return returned;
}

size_t NumBits(unsigned int number)
{
    size_t returned = 0;
    unsigned int shifted = number;
    unsigned int bit = 1;
    while (shifted != 0)
    {
        shifted >>= 1;
        bit <<= 1;
        returned++;
    }
    bit >>= 1;
    return returned;
}

NeuronBank * BackPropNeuronBank::CreateInstance(int NumInputs, int NumOutputs,
        Rand * r, double learn, double momentum, bool IsNominal,
        const std::vector<int> & Levels)
{
    return new BackPropNeuronBank(NumInputs, NumOutputs, r,
                                  learn, momentum, IsNominal, Levels);
}

void BackPropNeuronBank::RetireInstance(NeuronBank * instance)
{
    delete (BackPropNeuronBank*) instance;
}

double BackPropNeuronBank::TestMSE(const std::vector<double> & inputs,
        double target)
{
    std::vector< std::vector<double> > OutputBuffer;

    ForwardPropogate(inputs, OutputBuffer);

    RackRIterator OBackIter = OutputBuffer.rbegin();
    RowIterator OIter = OBackIter->begin();

    double error_ = 0.0;
    if (OutputSize == 0)
    {
        BankRowIterator OTIter = OutputTrons.begin();
        if (OutputTrons.end() == OutputTrons.begin())
        {
            throw NotImplementedException("HAVOC!");
        }

        double offset = target - *OIter;
        error_ += offset*offset;
    }
    else
    {
        double TargetInt = target;
        for (BankRowIterator OTIter = OutputTrons.begin();
                OTIter != OutputTrons.end(); OTIter++)
        {
            double MyTarget =
                (target == (double)(OTIter-OutputTrons.begin())) ?
                              1.0 : 0.0;
            double offset = MyTarget - *OIter;
            error_ += offset*offset;
            OIter++;
        }
    }
    error_ /= (double) OutputTrons.size();
    return error_;
}

void BackPropNeuronBank::Update(const std::vector<double> & inputs,
        double target)
{
    if (ErrorCount >= window)
    {
        return;
    }
    double difference = error;
    std::vector< std::vector<double> > OutputBuffer;

    ForwardPropogate(inputs, OutputBuffer);

    std::vector< std::vector<double> > deltas(OutputBuffer);

    RackRIterator OBackIter = OutputBuffer.rbegin();
    RackRIterator DeltaBackIter = deltas.rbegin();
    RowIterator OIter = OBackIter->begin();
    RowIterator DIter = DeltaBackIter->begin();

    error = 0.0;
    if (OutputSize == 0)
    {
        BankRowIterator OTIter = OutputTrons.begin();
        if (OutputTrons.end() == OutputTrons.begin())
        {
            throw NotImplementedException("HAVOC!");
        }

        double offset = target - *OIter;
        error += offset*offset;

        double delta = offset * (*OIter) * (1.0 - (*OIter));
        *DIter = delta;
        (*OTIter)->Update(*(OBackIter+1),delta);
    }
    else
    {
        double TargetInt = target;
        for (BankRowIterator OTIter = OutputTrons.begin();
                OTIter != OutputTrons.end(); OTIter++)
        {
            double MyTarget = (target == (double)(OTIter-OutputTrons.begin())) ?
                              1.0 : 0.0;
            double offset = MyTarget - *OIter;
            error += offset*offset;
            double delta = offset * (*OIter) * (1.0 - (*OIter));
            *DIter = delta;
            (*OTIter)->Update(*(OBackIter+1),delta);
            OIter++;
            DIter++;
        }
    }

    error /= (double) OutputTrons.size();

    if (fabs(difference-error) < LearningRate*LearningRate ||
            error < LearningRate)
    {
        ErrorCount++;
    }
    else
    {
        ErrorCount = 0;
    }
    // BACK PROPOGATION!!!

    // sync iterators
    DeltaBackIter++;
    OBackIter++;
    BankRackRIterator TronsBackIter = MiddleTrons.rbegin();

    // the above should be pointing to the same level.

    double sum;
    // calculate deltas for the top MiddleTrons row
    for (int h = 0; h < TronsBackIter->size(); h++)
    {
        sum = 0.0;
        RowIterator OTronsDeltaIter = (DeltaBackIter-1)->begin();

        for (BankRowIterator OTIter = OutputTrons.begin();
                OTIter != OutputTrons.end();
                OTIter++)
        {
            Neuron * n = *OTIter;
            double weight = n->Weight(h);
            double delta = *OTronsDeltaIter;
            sum += weight * delta;
            OTronsDeltaIter++;
        }

        double oput = (*OBackIter)[h];
        (*DeltaBackIter)[h] = oput * (1.0 - oput) * sum;
        if ((TronsBackIter+1) == MiddleTrons.rend())
        {
            (*TronsBackIter)[h]->Update(inputs,(*DeltaBackIter)[h]);
        }
        else
        {
            (*TronsBackIter)[h]->Update(*(OBackIter+1),(*DeltaBackIter)[h]);
        }
    }

    if ((TronsBackIter+1) == MiddleTrons.rend())
    {
        return;
    }

    // first deltas: the output
    // second deltas: the top middleTrons layer
    /// third (size - 3): the next layer down


    // sync iterators
    DeltaBackIter++;
    OBackIter++;
    TronsBackIter++;

    while (TronsBackIter != (MiddleTrons.rend()-1))
    {
        for (int h = 0; h < TronsBackIter->size(); h++)
        {
            sum = 0.0;
            BankRowIterator TronIter = (TronsBackIter-1)->begin();
            RowIterator DeltaIter = (DeltaBackIter-1)->begin();
            while (TronIter != (TronsBackIter-1)->end())
            {
                Neuron * n = *TronIter;
                double weight = n->Weight(h);
                double delta = *DeltaIter;
                sum += weight * delta;
                TronIter++;
                DeltaIter++;
            }

            double oput = (*OBackIter)[h];
            (*DeltaBackIter)[h] = oput * (1.0 - oput) * sum;
            (*TronsBackIter)[h]->Update(*(OBackIter+1),(*DeltaBackIter)[h]);
        }
        // sync iterators
        DeltaBackIter++;
        OBackIter++;
        TronsBackIter++;
    }

    for (int h = 0; h < TronsBackIter->size(); h++)
    {
        sum = 0.0;
        BankRowIterator TronIter = (TronsBackIter-1)->begin();
        RowIterator DeltaIter = (DeltaBackIter-1)->begin();
        while (TronIter != (TronsBackIter-1)->end())
        {
            Neuron * n = *TronIter;
            double weight = n->Weight(h);
            double delta = *DeltaIter;
            sum += weight * delta;
            TronIter++;
            DeltaIter++;
        }

        double oput = (*OBackIter)[h];
        (*DeltaBackIter)[h] = oput * (1.0 - oput) * sum;
        (*TronsBackIter)[h]->Update(inputs,(*DeltaBackIter)[h]);
    }
}

double BackPropNeuronBank::MSE() const
{
    return error;
}

void BackPropNeuronBank::Deltas(std::vector<double> & ds) const
{
    throw NotImplementedException("Don't call this method.");
}

const Neuron * BackPropNeuronBank::Tron(int index) const
{
    return OutputTrons[index];
}

const size_t BackPropNeuronBank::TronSize() const
{
    return OutputTrons.size();
}

void BackPropNeuronBank::ForwardPropogate(
    const std::vector<double> & inputs,
    std::vector<std::vector<double> > & OutputBuffer) const
{
    OutputBuffer.clear();
    OutputBuffer.assign(MiddleTrons.size()+1, std::vector<double>());
    RackIterator OBIter = OutputBuffer.begin();
    BankRackCIterator MTIter = MiddleTrons.begin();

    OBIter->assign(MTIter->size(),0.0);
    RowIterator FirstOutRow = OBIter->begin();
    for (BankRowCIterator FirstRow = MTIter->begin();
            FirstRow != MTIter->end();
            FirstRow++)
    {
        Neuron * tron = *FirstRow;
        double answer = tron->Fire(inputs);
        *FirstOutRow = answer;
        FirstOutRow++;
    }

    OBIter++;
    MTIter++;

    for ( ; MTIter != MiddleTrons.end(); MTIter++)
    {
        OBIter->assign(MTIter->size(),0.0);
        RowIterator OutputRow = OBIter->begin();
        for (BankRowCIterator tron = MTIter->begin();
                tron != MTIter->end(); tron++)
        {
            Neuron * t = *tron;
            double answer = t->Fire(*(OBIter-1));
            *OutputRow = answer;
            OutputRow++;
        }
        OBIter++;
    }

    OBIter->assign(OutputTrons.size(),0.0);
    RowIterator OutputRow = OBIter->begin();
    for (BankRowCIterator OutRow = OutputTrons.begin();
            OutRow != OutputTrons.end();
            OutRow++)
    {
        Neuron * tron = *OutRow;
        double answer = tron->Fire(*(OBIter-1));
        *OutputRow = answer;
        OutputRow++;
    }

}

double  BackPropNeuronBank::Predict(const std::vector<double> & inputs) const
{
    std::vector< std::vector<double> > OutputBuffer;

    ForwardPropogate(inputs, OutputBuffer);

    std::vector<double> & Output = *(OutputBuffer.end()-1);

    if (OutputSize == 0)
    {
        return *(Output.begin());
    }

    return (double)(max_element(Output.begin(),Output.end())-Output.begin());
}

bool  BackPropNeuronBank::Trained() const
{
    return ErrorCount >= Window();
}

int  BackPropNeuronBank::Window() const
{
    return window;
}

BackPropNeuronBank::BackPropNeuronBank(
    int NumInputs, int NumOutputs, Rand * r, double learn, double momentum,
    bool IsNominal, const std::vector<int> & levels)
    : InputSize(NumInputs), OutputSize(NumOutputs), rand(r),
      LearningRate(learn), MomentumRate(momentum), Nominal(IsNominal),
      Levels(levels), MiddleTrons(), OutputTrons(), error(0.0),
      ErrorCount(0),
      window()
{
    if (Levels.size() < 1)
    {
        stringstream ss;
        ss << "insufficient levels" << " at " << __LINE__
           << " in " << __FILE__;
        throw ArgumentException(ss.str());
    }

    MiddleTrons.resize(Levels.size());
    BankRackIterator iter = MiddleTrons.begin();
    std::vector<int>::iterator level = Levels.begin();
    iter->resize(*level);
    for (BankRowIterator jter = iter->begin(); jter != iter->end(); jter++ )
    {
        *jter =
            Perceptron::CreateInstance<SigmoidFire>(NumInputs, rand, LearningRate, MomentumRate);
    }

    iter++;
    level++;
    while(iter != MiddleTrons.end())
    {
        iter->resize(*level);
        for (BankRowIterator jter = iter->begin(); jter != iter->end(); jter++)
        {
            *jter =
                Perceptron::CreateInstance<SigmoidFire>((iter-1)->size(),
                        rand, LearningRate, MomentumRate);
        }
        iter++;
        level++;
    }

    OutputTrons.resize(OutputSize);
    for (BankRowIterator OIter = OutputTrons.begin();
            OIter != OutputTrons.end(); OIter++)
    {
        *OIter = Perceptron::CreateInstance<SigmoidFire>((iter-1)->size(),
                 rand, LearningRate, MomentumRate);
    }
    window = 20;
}

BackPropNeuronBank::~BackPropNeuronBank()
{
    free();
}


void FreeBankRow(const std::vector<Neuron*> & i)
{
    for_each(i.begin(), i.end(), Perceptron::RetireInstance<SigmoidFire>);
}

void BackPropNeuronBank::free()
{
    BankRackIterator j = MiddleTrons.begin();
    for_each(MiddleTrons.begin(), MiddleTrons.end(), FreeBankRow);

    MiddleTrons.clear();
    FreeBankRow(OutputTrons);
    OutputTrons.clear();
}
