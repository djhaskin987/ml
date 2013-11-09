#ifndef __ml_NotImplementedException__
#define __ml_NotImplementedException__

#include <stdexcept>

namespace ml
{
class NotImplementedException : public std::runtime_error
{
public:
    explicit NotImplementedException(const std::string & what_arg)
        : std::runtime_error(what_arg)
    {
    }
};
}

#endif // __ml_NotImplementedException__
