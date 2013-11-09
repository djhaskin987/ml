#ifndef ml_Neuron_H
#define ml_Neuron_H

#include <vector>
#include <functional>
#include <cstdlib>

class Neuron
{
public:
    virtual void Update(const std::vector<double> & inputs, double offset) = 0;
    virtual double Fire(const std::vector<double> & inputs) const = 0;
    virtual double Vote(const std::vector<double> & inputs) const = 0;
    virtual double Weight(std::size_t index) const = 0;
    virtual int InputSize() const = 0;
    virtual double LearningRate() const = 0;
};
#endif
