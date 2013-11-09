#ifndef __ml_StateException__
#define __ml_StateException__

#include <stdexcept>

namespace ml
{
class StateException : public std::runtime_error
{
public:
    explicit StateException(const std::string & what_arg)
        : std::runtime_error(what_arg)
    {
    }
};
}

#endif // __ml_StateException__
