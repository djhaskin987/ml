#ifndef __ml_NeuronBankFactory__
#define __ml_NeuronBankFactory__

#include <functional>
#include "NeuronBank.h"
#include "rand.h"

namespace ml
{
class NeuronBankFactory
{
public:
    virtual NeuronBank *  operator () (int NumInputs,
                                       int NumOutputs, Rand * r, double learn,
                                       double momentum) = 0;
    virtual void Destroy(NeuronBank * instance) = 0;
    virtual ~NeuronBankFactory()
    {
    }
};
}

#endif // __ml_NeuronBankFactory__
