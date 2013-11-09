#include <cassert>
#include "Perceptron.h"
#include "ArgumentException.h"
#include <sstream>
#include <cstdlib>

#include <exception>
#include <stdexcept>

using namespace std;
using namespace ml;

Perceptron::Perceptron(int num_inputs, Rand * r, double LearningRate,
                       double MomentumTerm, ml::Fire * f) :
    rand(r), learning_rate(LearningRate), momentum_term(MomentumTerm),
    momentum(num_inputs,0.0), bias_momentum(0.0), weights(num_inputs,1.0),
    bias(0.0), fire(f)
{
    for (int i = 0; i < weights.size(); i++)
    {
        weights[i] = rand->normal() * LearningRate;
    }
}

Perceptron::~Perceptron()
{
    free();
}

void Perceptron::free()
{
}

void Perceptron::Update(const std::vector<double> & inputs, double offset)
{
    if (inputs.size() != weights.size())
    {
        assert(0);
        stringstream ss;
        ss << "Update: Inputs must be equinumerous with weights." << " at "
           << __LINE__ << " in " << __FILE__;
        throw runtime_error(ss.str());
    }
    double momentum_term = .9;
    double delta = learning_rate * offset;
    double update;
    for (int i = 0; i < weights.size(); i++)
    {
        update = delta * inputs[i] + momentum_term*momentum[i];
        weights[i] += update;
        momentum[i] = update;
    }
    update = delta + momentum_term * bias_momentum;
    bias += update;
    bias_momentum = update;
}

double Perceptron::Fire(const std::vector<double> & inputs) const
{
    return (*fire)(Vote(inputs));
}

double Perceptron::Vote(const std::vector<double> & inputs) const
{
    if (inputs.size() != weights.size())
    {
        assert(0);
        stringstream ss;
        ss << "inputs not the same size as weights!" << " at "
           << __LINE__ << " in " << __FILE__;
        throw ArgumentException(ss.str());
    }

    double result = 0.0;
    for (int i = 0; i < weights.size(); i++)
    {
        result += weights[i]*inputs[i];
    }
    result += bias;
    return result;
}

double Perceptron::Weight(size_t index) const
{
    if (index < 0 ||
            index >= weights.size())
    {
        stringstream ss;
        ss << "index out of bounds" << " at "
           << __LINE__ << " in " << __FILE__;
        throw ArgumentException(ss.str());
    }
    return weights[index];
}

int Perceptron::InputSize() const
{
    return weights.size();
}

double Perceptron::LearningRate() const
{
    return learning_rate;
}
