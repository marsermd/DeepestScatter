#include "Dataset.h"
#include <iostream>
#include "LmdbExceptions.h"
#include <cassert>

namespace DeepestScatter
{
    Dataset::Dataset(const Settings& settings)
    {
        std::cout << "creating environment" << std::endl;
        LmdbExceptions::checkError(mdb_env_create(&mdbEnv));

        std::cout << "setting db count" << std::endl;
        LmdbExceptions::checkError(mdb_env_set_maxdbs(mdbEnv, 64u));

        std::cout << "opening db" << std::endl;
        LmdbExceptions::checkError(
            mdb_env_open(mdbEnv, settings.path.c_str(), MDB_NOTLS | MDB_NOSUBDIR, 0)
        );

        std::cout << "success" << std::endl;
    }

    Dataset::~Dataset()
    {
        for (const auto& nameAndTable: openedTables)
        {
            closeTable(nameAndTable.first);
        }

        std::cout << "closing environment" << std::endl;
        mdb_env_close(mdbEnv);
    }

    void Dataset::increaseSizeIfNeededWhile(const std::function<void(void)>& action)
    {
        for (int retry = 0; retry < 2; retry++)
        {
            try
            {
                action();
            }
            catch (const LmdbExceptions::MapFull& mapFull)
            {
                increaseSize();
                continue;
            }
            break;
        }
    }

    void Dataset::increaseSize()
    {
        MDB_envinfo info;
        LmdbExceptions::checkError(
            mdb_env_info(mdbEnv, &info)
        );

        mdb_size_t size = info.me_mapsize;
        size *= 2;

        mdb_env_set_mapsize(mdbEnv, size);
    }

    MDB_dbi Dataset::getTable(const TableName& name)
    {
        const auto tableIt = openedTables.find(name);
        if (tableIt == openedTables.end())
        {
            return openTable(name);
        }
        return tableIt->second;
    }

    MDB_dbi Dataset::openTable(const TableName& name)
    {
        assert(openedTables.find(name) == openedTables.end());

        return Transaction::withTransaction<MDB_dbi>(mdbEnv, nullptr, 0, [&](Transaction& transaction)
        {
            MDB_dbi dbi;
            mdb_dbi_open(transaction, name.c_str(), MDB_INTEGERKEY | MDB_CREATE, &dbi);
            openedTables[name] = dbi;

            return dbi;
        });
        
    }

    void Dataset::closeTable(const TableName& name)
    {
        assert(openedTables.find(name) != openedTables.end());

        Transaction::withTransaction<void>(mdbEnv, nullptr, 0, [&](Transaction& transaction)
        {

            mdb_dbi_close(mdbEnv, openedTables[name]);
            openedTables.erase(name);
        });
    }
}
