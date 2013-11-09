#ifndef __ml_BackPropNeuronBank__
#define __ml_BackPropNeuronBank__

#include "Perceptron.h"
#include "NeuronBank.h"
#include "PerceptronNeuronBank.h"
#include "rand.h"
namespace ml
{
class BackPropNeuronBank : public NeuronBank
{
public:
    /*! \brief returns an instance of type BackPropNeuronBank
     *        (this class) on the heap.
     */
    static NeuronBank * CreateInstance(
        int NumInputs, int NumOutputs, Rand * r, double learn,
        double momentum, bool IsNominal,
        const std::vector<int> & Levels);

    /*! \brief retires an instance of type BackPropNeuronBank
     *        (this class) on the heap.
     *  \param instance the pointer to the instance of this class. It is
     *        expected that instance is in fact an instance of BackPropNeuronBank.
     */
    static void RetireInstance(NeuronBank * instance);
    virtual void Update(const std::vector<double> & inputs, double target);
    virtual void Deltas(std::vector<double> & deltas) const;
    virtual double MSE() const;
    virtual double TestMSE(const std::vector<double> & inputs,
        double target);
    virtual const Neuron * Tron(int index) const;
    virtual const size_t TronSize() const;
    virtual double Predict(const std::vector<double> & inputs) const;
    virtual bool Trained() const ;
    virtual int Window() const;
private:
    void free();
    BackPropNeuronBank(
        int NumInputs, int NumOutputs, Rand * r, double learn,
        double momentum, bool IsNominal,
        const std::vector<int> & levels);
    virtual ~BackPropNeuronBank();
    void ForwardPropogate(
        const std::vector<double> & inputs,
        std::vector<std::vector<double> > & OutputBuffer) const;
    int InputSize;
    int OutputSize;
    Rand * rand;
    double LearningRate;
    double MomentumRate;
    bool Nominal;
    std::vector<int> Levels;
    std::vector<std::vector<Neuron*> > MiddleTrons;
    std::vector<Neuron*> OutputTrons;
    double error;
    int ErrorCount;
    int window;
};
}

#endif // __ml_BackPropNeuronBank__
