#ifndef __ml_SigmoidFire__
#define __ml_SigmoidFire__

#include "Fire.h"

namespace ml
{
class SigmoidFire : public Fire
{
public:
    /*! \brief returns an instance of type SigmoidFire
     *        (this class) on the heap.
     */
    static Fire * CreateInstance();
    /*! \brief retires an instance of type SigmoidFire
     *        (this class) on the heap.
     *  \param instance the pointer to the instance of this class. It is
     *        expected that instance is in fact an instance of SigmoidFire.
     */
    static void RetireInstance(Fire * instance);
    virtual double operator () (double vote);
private:
    void free();
    SigmoidFire();
    virtual ~SigmoidFire();
};
}

#endif // __ml_SigmoidFire__
