#ifndef __ml_PerceptronNeuronBankFactory__
#define __ml_PerceptronNeuronBankFactory__

#include "NeuronBankFactory.h"

namespace ml
{
class PerceptronNeuronBankFactory : public NeuronBankFactory
{
public:
    /*! \brief returns an instance of type PerceptronNeuronBankFactory
     *        (this class) on the heap.
     */
    static NeuronBankFactory * CreateInstance();
    /*! \brief retires an instance of type PerceptronNeuronBankFactory
     *        (this class) on the heap.
     *  \param instance the pointer to the instance of this class. It is
     *        expected that instance is in fact an instance of PerceptronNeuronBankFactory.
     */
    static void RetireInstance(NeuronBankFactory * instance);
    virtual NeuronBank * operator () (int NumInputs, int NumOutputs,
                                      Rand * r, double learn, double momentum);
    virtual void Destroy(NeuronBank * instance);
private:
    void free();
    PerceptronNeuronBankFactory();
    virtual ~PerceptronNeuronBankFactory();
};
}

#endif // __ml_PerceptronNeuronBankFactory__
