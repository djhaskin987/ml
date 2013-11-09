#ifndef __ml_Fire__
#define __ml_Fire__

namespace ml
{
class Fire
{
public:
    virtual double operator () (double vote) = 0;
    virtual ~Fire()
    {
    }
};
}

#endif // __ml_Fire__
