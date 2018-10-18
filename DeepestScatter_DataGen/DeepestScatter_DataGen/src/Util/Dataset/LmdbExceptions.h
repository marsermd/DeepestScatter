#pragma once
#include <exception>
#include <string>
#include <utility>

namespace DeepestScatter
{
    class LmdbExceptions
    {
    public:
        struct GenericException: std::exception
        {
            GenericException(std::string cause): cause(std::move(cause)){}
            const std::string cause;
        };

        struct MapFull: public std::exception
        {
            MapFull(): std::exception("LMDB environment map limit reached") {}
        };

        static void checkError(int code);
    };
}
