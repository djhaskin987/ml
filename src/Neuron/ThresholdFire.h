#ifndef __ml_ThresholdFire__
#define __ml_ThresholdFire__

#include "Fire.h"

namespace ml
{
class ThresholdFire : public Fire
{
public:
    /*! \brief returns an instance of type ThresholdFire
     *        (this class) on the heap.
     */
    static Fire * CreateInstance();
    /*! \brief retires an instance of type ThresholdFire
     *        (this class) on the heap.
     *  \param instance the pointer to the instance of this class. It is
     *        expected that instance is in fact an instance of ThresholdFire.
     */
    static void RetireInstance(Fire * instance);
    virtual double operator () (double vote);
private:
    void free();
    ThresholdFire();
    virtual ~ThresholdFire();
};
}

#endif // __ml_ThresholdFire__
