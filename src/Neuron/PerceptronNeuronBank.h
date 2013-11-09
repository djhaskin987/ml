#ifndef __ml_PerceptronNeuronBank__
#define __ml_PerceptronNeuronBank__

#include "Neuron.h"
#include "NeuronBank.h"
#include "Perceptron.h"
#include "rand.h"
#include <utility>

namespace ml
{
class PerceptronNeuronBank : public NeuronBank
{
private:
    static const int WINDOW = 5;
    double learning_rate;
    double momentum_term;
    Rand * rand;
    int inputs;
    int OutputValues;
    std::vector<std::pair<Neuron*,std::pair<int,int> > > trons;
    int Count;
    double Error;
    std::vector<double> deltas;
    void free();

    PerceptronNeuronBank(int NumOutputs, int NumInputs, Rand * r,
                         double LearningRate, double MomentumTerm,
                         std::vector<std::pair<Neuron*,std::pair<int,int> > > ts);
    virtual ~PerceptronNeuronBank();
    void VoteForMe(std::vector<int> & votes, size_t & index, int vote) const;
public:
    /*! \brief returns an instance of this class on the heap.
     */
    template <class PType>
    static NeuronBank * CreateInstance(int NumInputs, int NumOutputs,
                                       Rand * r, double LearningRate, double MomentumTerm);
    /*! \brief retires an instance of this class on the heap.
     *  \param instance the pointer to the instance of this class. It is
     *        expected that instance is in fact an instance of PerceptronNeuronBank.
     */
    template <class PType>
    static void RetireInstance(ml::NeuronBank * instance);

    virtual void Update(const std::vector<double> & inputs, double target);
    virtual void Deltas(std::vector<double> & deltas) const;
    virtual double MSE() const;
    virtual double Predict(const std::vector<double> & inputs) const;
    virtual double TestMSE(const std::vector<double> & inputs,
        double target)
    {
        Predict(inputs);
    }
    virtual const Neuron * Tron(int index) const;
    virtual const size_t TronSize() const;
    virtual bool Trained() const;
    virtual int Window() const;
};

template <class FType>
NeuronBank * PerceptronNeuronBank::CreateInstance(int NumInputs, int NumOutputs,
        Rand * r, double LearningRate, double MomentumTerm)
{
    std::vector<std::pair<Neuron*, std::pair<int, int> > > trons;
    if (NumOutputs == 0)
    {
        trons.push_back(
            std::make_pair(Perceptron::CreateInstance<FType>(NumInputs, r,
                           LearningRate, MomentumTerm), std::make_pair(-1,-1)));
    }
    else
    {
        int size = NumOutputs*(NumOutputs+1) >> 1;
        trons.reserve(size + 1);
        for (int i = 0; i < NumOutputs;  i++)
        {
            for (int j = i + 1; j < NumOutputs; j++)
            {
                trons.push_back(
                    std::make_pair(Perceptron::CreateInstance<FType>(NumInputs, r,
                                   LearningRate, MomentumTerm), std::make_pair(i,j)));
            }
        }
    }
    return ((NeuronBank*)(new PerceptronNeuronBank(NumInputs, NumOutputs, r, LearningRate,
                          MomentumTerm, trons)));
}

template <class FType>
void PerceptronNeuronBank::RetireInstance(NeuronBank * instance)
{
    PerceptronNeuronBank * pinstance =
        (PerceptronNeuronBank*) instance;
    for (int i = 0; i < pinstance->trons.size(); i++)
    {
        Perceptron::RetireInstance<FType>(pinstance->trons[i].first);
    }
    delete instance;
}

}

#endif // __ml_PerceptronNeuronBank__
