#ifndef __ml_BackPropNeuronBankFactory__
#define __ml_BackPropNeuronBankFactory__

#include "NeuronBankFactory.h"
#include <vector>

namespace ml
{
class BackPropNeuronBankFactory : public NeuronBankFactory
{
public:
    /*! \brief returns an instance of type BackPropNeuronBankFactory
     *        (this class) on the heap.
     */
    static NeuronBankFactory * CreateInstance(
        bool nominal, const std::vector<int> & levels);
    /*! \brief retires an instance of type BackPropNeuronBankFactory
     *        (this class) on the heap.
     *  \param instance the pointer to the instance of this class. It is
     *        expected that instance is in fact an instance of BackPropNeuronBankFactory.
     */
    static void RetireInstance(NeuronBankFactory * instance);
    virtual NeuronBank * operator () (int NumInputs, int NumOutputs,
                                      Rand * r, double learn, double momentum);
    virtual void Destroy(NeuronBank * instance);
private:
    void free();
    BackPropNeuronBankFactory(bool nominal,
                              const std::vector<int> & levels);
    virtual ~BackPropNeuronBankFactory();
    bool IsNominal;
    const std::vector<int> Levels;
};
}

#endif // __ml_BackPropNeuronBankFactory__
