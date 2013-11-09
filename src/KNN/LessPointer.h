#ifndef __ml_LessPointer__
#define __ml_LessPointer__

#include <functional>

namespace ml
{
    template <class Type>
    class LessPointer : std::binary_function<Type*,Type*,bool>
    {
        return_type operator () (const first_argument_type a, 
            const second_argument_type b)
        {
            return (*a) < (*b);
        }
    };

}

#endif // __ml_LessPointer__
