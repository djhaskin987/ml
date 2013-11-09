#include "SigmoidFire.h"
#include <cmath>

using namespace std;
using namespace ml;

Fire* SigmoidFire::CreateInstance()
{
    return new SigmoidFire();
}

void SigmoidFire::RetireInstance(Fire * instance)
{
    delete (SigmoidFire*) instance;
}

double  SigmoidFire::operator () (double vote)
{
    return 1.0 / (1.0 + exp(-vote));
}

SigmoidFire::SigmoidFire()
{
}

SigmoidFire::~SigmoidFire()
{
    free();
}

void SigmoidFire::free()
{
}

