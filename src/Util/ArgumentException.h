#ifndef __ml_ArgumentException__
#define __ml_ArgumentException__

#include <stdexcept>

namespace ml
{
class ArgumentException : public std::runtime_error
{
public:
    explicit ArgumentException(const std::string & what_arg)
        : std::runtime_error(what_arg)
    {
    }
};
}

#endif // __ml_ArgumentException__
