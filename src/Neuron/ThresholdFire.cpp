#include "ThresholdFire.h"

using namespace ml;

Fire* ThresholdFire::CreateInstance()
{
    return new ThresholdFire();
}

void ThresholdFire::RetireInstance(Fire * instance)
{
    delete (ThresholdFire*) instance;
}

double  ThresholdFire::operator () (double vote)
{
    return vote > 0.0 ? 1.0 : 0.0;
}

ThresholdFire::ThresholdFire()
{
}

ThresholdFire::~ThresholdFire()
{
    free();
}

void ThresholdFire::free()
{
}

