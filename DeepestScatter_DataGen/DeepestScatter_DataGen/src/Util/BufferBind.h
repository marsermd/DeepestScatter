#pragma once
#include <cassert>
#include <gsl/span>

#include <iostream>

#include <optixu/optixpp_namespace.h>

namespace DeepestScatter
{
    template<typename T>
    class BufferBind
    {
    public:
        explicit BufferBind(
            optix::Buffer buffer, 
            unsigned int level = 0, unsigned int map_flags = RT_BUFFER_MAP_READ_WRITE);
        
        ~BufferBind();

        gsl::span<T>& getData();

        const T& operator [](size_t pos) const;
        T& operator [](size_t pos);

    private:
        optix::Buffer buffer;
        unsigned int level;

        gsl::span<T> optixOwned;
    };

    
    template<typename T>
    BufferBind<T>::BufferBind(optix::Buffer buffer, unsigned int level, unsigned int map_flags):
        buffer(buffer),
        level(level)
    {
        assert(sizeof(T) == buffer->getElementSize());
        auto rawArray = static_cast<T*>(buffer->map(level, map_flags));

        RTsize totalSize = 1;
        RTsize sizes[3];
        buffer->getMipLevelSize(level, sizes[0], sizes[1], sizes[2]);
        for (unsigned int i = 0; i < buffer->getDimensionality(); i++)
        {
            totalSize *= sizes[i];
        }

        optixOwned = gsl::make_span(rawArray, totalSize);
    }

    template <typename T>
    BufferBind<T>::~BufferBind()
    {
        buffer->unmap(level);
    }


    template <typename T>
    gsl::span<T>& BufferBind<T>::getData()
    {
        return optixOwned;
    }

    template <typename T>
    inline const T& BufferBind<T>::operator[](size_t pos) const
    {
        return optixOwned[pos];
    }

    template <typename T>
    inline T& BufferBind<T>::operator[](size_t pos)
    {
        return optixOwned[pos];
    }
}
