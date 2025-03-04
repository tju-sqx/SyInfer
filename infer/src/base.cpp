
#include "base.h"

namespace base
{
    int32_t data_type_size(DateType data_type){
        switch (data_type) {
            case DateType::DATA_FP32:
                return 4;
            case DateType::DATA_INT8:
                return 1;
            case DateType::DATA_INT16:
                return 2;
            default:
                return 0;
        }
    }
    
} // namespace base

