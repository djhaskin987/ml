#ifndef ml_Perceptron_H
#define ml_Perceptron_H

#include "Neuron.h"
#include "Fire.h"
#include "rand.h"

namespace ml
{
class Perceptron : public Neuron
{
private:
    Rand * rand;
    double learning_rate;
    double momentum_term;
    double bias_momentum;
    std::vector<double> momentum;
    std::vector<double> weights;
    double bias;
    ml::Fire * fire;
    typedef std::vector<double>::iterator witer_t;
    void free();
    virtual ~Perceptron();

    Perceptron(int num_inputs, Rand * r, double LearningRate,
               double MomentumTerm, ml::Fire * f);
public:
    virtual void Update(const std::vector<double> & inputs, double offset);
    virtual double Fire(const std::vector<double> & inputs) const;
    virtual double Vote(const std::vector<double> & inputs) const;
    virtual double Weight(size_t index) const;
    virtual int InputSize() const;
    virtual double LearningRate() const;

    template <class FType>
    static Neuron * CreateInstance(int NumInputs, Rand * r,
                                   double LearningRate, double MomentumTerm)
    {

        return new Perceptron(NumInputs, r, LearningRate, MomentumTerm,
                              FType::CreateInstance());
    }

    template <class FType>
    static void RetireInstance(Neuron * instance)
    {
        Perceptron * pinstance = (Perceptron*) instance;
        FType::RetireInstance(pinstance->fire);
        delete instance;
    }
};
}
#endif // ml_Perceptron_H
