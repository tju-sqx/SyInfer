#ifndef BASE_H
#define BASE_H

#include <cstdint>

namespace base{
    enum class DeviceType{
        UNKNOWN = 0,
        CPU = 1,
        GPU = 2
    };

    class NoCopyable {
    protected:
        NoCopyable() = default;
        ~NoCopyable() = default;
        NoCopyable(const NoCopyable&) = delete;
        NoCopyable& operator=(const NoCopyable&) = delete;
    };

    enum class DateType: uint8_t{
        DATA_UNKONWN = 0,
        DATA_FP32 = 1,
        DATA_INT8 = 2,
        DATA_INT16 = 3
    };

    int32_t data_type_size(DateType data_type);
    
}

#endif