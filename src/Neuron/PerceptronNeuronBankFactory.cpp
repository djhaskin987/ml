#include "PerceptronNeuronBankFactory.h"
#include "PerceptronNeuronBank.h"
#include "Perceptron.h"
#include "SigmoidFire.h"

using namespace ml;

NeuronBankFactory* PerceptronNeuronBankFactory::CreateInstance()
{
    return new PerceptronNeuronBankFactory();
}

void PerceptronNeuronBankFactory::RetireInstance(NeuronBankFactory * instance)
{
    delete (PerceptronNeuronBankFactory*) instance;
}

NeuronBank * PerceptronNeuronBankFactory::operator () (int NumInputs, int NumOutputs, Rand * r,
        double learn, double momentum)
{
    return PerceptronNeuronBank::CreateInstance<SigmoidFire>(NumInputs,
            NumOutputs, r, learn, momentum);
}


void PerceptronNeuronBankFactory::Destroy(NeuronBank * instance)
{
    PerceptronNeuronBank::RetireInstance<SigmoidFire>(instance);
}

PerceptronNeuronBankFactory::PerceptronNeuronBankFactory()
{
}

PerceptronNeuronBankFactory::~PerceptronNeuronBankFactory()
{
    free();
}

void PerceptronNeuronBankFactory::free()
{
}

