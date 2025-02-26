#ifndef BASE_H
#define BASE_H

namespace base{
    enum class DeviceAllocType{
        UNKNOWN,
        CPU,
        GPU
    };

    class NoCopyable {
    protected:
        NoCopyable() = default;
        ~NoCopyable() = default;
        NoCopyable(const NoCopyable&) = delete;
        NoCopyable& operator=(const NoCopyable&) = delete;
    };
}

#endif