#ifndef __ml_UpdateRule__
#define __ml_UpdateRule__

#include <functional>

namespace ml
{
class UpdateRule :
    public std::binary_function<double,double,double>
{
public:
    virtual double operator () (double input, double offset) = 0;
    virtual ~UpdateRule()
    {
    }
};
}

#endif // __ml_UpdateRule__
