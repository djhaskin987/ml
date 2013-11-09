#ifndef __ml_NeuronBank__
#define __ml_NeuronBank__

#include <vector>
#include "Neuron.h"
namespace ml
{
class NeuronBank
{
public:
    virtual void Update(const std::vector<double> & inputs, double target) = 0;
    virtual void Deltas(std::vector<double> & deltas) const = 0;
    virtual double MSE() const= 0;
    virtual double Predict(const std::vector<double> & inputs) const = 0;
    virtual const Neuron * Tron(int index) const = 0;
    virtual double TestMSE(const std::vector<double> & inputs,
        double target) = 0;
    virtual const size_t TronSize() const = 0;
    virtual bool Trained() const = 0;
    virtual int Window() const = 0;
    virtual ~NeuronBank()
    {
    }
};
}
#endif // __ml_NeuronBank__
