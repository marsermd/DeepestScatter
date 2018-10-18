#include "LmdbExceptions.h"
#include <lmdb.h>

namespace DeepestScatter
{
    void LmdbExceptions::checkError(int code)
    {
        switch (code)
        {
        case 0:
            return;
        case MDB_MAP_FULL:
            throw MapFull();
        default:
            throw GenericException(mdb_strerror(code));
        }
    }
}
