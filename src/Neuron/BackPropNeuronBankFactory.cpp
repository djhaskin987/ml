#include "BackPropNeuronBankFactory.h"
#include "BackPropNeuronBank.h"

using namespace ml;

NeuronBankFactory* BackPropNeuronBankFactory::CreateInstance(
    bool nominal, const std::vector<int> & levels)
{
    return new BackPropNeuronBankFactory(nominal, levels);
}

void BackPropNeuronBankFactory::RetireInstance(NeuronBankFactory * instance)
{
    delete (BackPropNeuronBankFactory*) instance;
}

NeuronBank * BackPropNeuronBankFactory::operator () (int NumInputs, int NumOutputs, Rand * r,
        double learn, double momentum)
{
    return BackPropNeuronBank::CreateInstance(NumInputs, NumOutputs, r,
            learn, momentum, IsNominal, Levels);
}

void BackPropNeuronBankFactory::Destroy(NeuronBank * instance)
{
    BackPropNeuronBank::RetireInstance(instance);
}

BackPropNeuronBankFactory::BackPropNeuronBankFactory(
    bool nominal, const std::vector<int> & levels) :
    IsNominal(nominal), Levels(levels)
{
}

BackPropNeuronBankFactory::~BackPropNeuronBankFactory()
{
    free();
}

void BackPropNeuronBankFactory::free()
{
}
