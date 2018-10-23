#pragma once

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <gsl/span>

#include <lmdb.h>
#include <functional>
#include "Transaction.h"
#include "LmdbExceptions.h"

namespace DeepestScatter
{
    class Dataset
    {
    public:
        struct Settings;
        class Example;

        explicit Dataset(std::shared_ptr<Settings> settings);
        ~Dataset();

        template<class T>
        void append(const T& example);

        template<class T>
        void batchAppend(gsl::span<T> examples);

        struct Settings
        {
            Settings(std::string path) :
                path(std::move(path))
            {}

            std::string path;
        };

    private:
        using TableName = std::string;

        template<class T>
        void tryAppend(const T& example, int entryId);

        template<class T>
        void tryBatchAppend(gsl::span<T> examples, int nextExampleId);

        void increaseSizeIfNeededWhile(const std::function<void(void)>& action);
        void increaseSize();

        MDB_dbi getTable(const TableName& name);
        MDB_dbi openTable(const TableName& name);
        void closeTable(const TableName& name);

        std::unordered_map<TableName, MDB_dbi> openedTables;
        std::unordered_map<TableName, int32_t> nextIds;

        MDB_env* mdbEnv = nullptr;
    };

    template <class T>
    void Dataset::append(const T& example)
    {
        const TableName tableName = T::descriptor()->full_name();
        // Returns zero if not initialized;
        int entryId = nextIds[tableName];

        increaseSizeIfNeededWhile([&]()
        {
            return tryAppend(example, entryId);
        });

        nextIds[tableName] = entryId + 1;
    }

    template <class T>
    void Dataset::batchAppend(gsl::span<T> examples)
    {
        const TableName tableName = T::descriptor()->full_name();
        // Returns zero if not initialized;
        int startId = nextIds[tableName];

        increaseSizeIfNeededWhile([&]()
        {
            return tryBatchAppend(examples, startId);
        });

        nextIds[tableName] = startId + examples.size();
    }


    template <class T>
    void Dataset::tryAppend(const T& example, int entryId)
    {
        const TableName tableName = T::descriptor()->full_name();
        MDB_dbi dbi = getTable(tableName);

        Transaction::withTransaction<void>(mdbEnv, nullptr, 0, [&](Transaction& transaction)
        {

            MDB_val mdbKey
            {
                sizeof(int32_t),
                &entryId
            };

            const size_t length = example.ByteSizeLong();
            std::vector<uint8_t> serialized(length);
            example.SerializeWithCachedSizesToArray(&serialized[0]);

            MDB_val mdbVal
            {
                length * sizeof(uint8_t),
                static_cast<void*>(&serialized[0])
            };

            LmdbExceptions::checkError(mdb_put(transaction, dbi, &mdbKey, &mdbVal, 0));
        });
    }

    template <class T>
    void Dataset::tryBatchAppend(gsl::span<T> examples, int nextExampleId)
    {
        const TableName tableName = T::descriptor()->full_name();
        MDB_dbi dbi = getTable(tableName);

        Transaction::withTransaction<void>(mdbEnv, nullptr, 0, [&](Transaction& transaction)
        {
            for (const auto& example : examples)
            {
                MDB_val mdbKey
                {
                    sizeof(int32_t),
                    &nextExampleId
                };

                const size_t length = example.ByteSizeLong();
                std::vector<uint8_t> serialized(length);
                example.SerializeWithCachedSizesToArray(&serialized[0]);

                MDB_val mdbVal
                {
                    length * sizeof(uint8_t),
                    static_cast<void*>(&serialized[0])
                };

                LmdbExceptions::checkError(mdb_put(transaction, dbi, &mdbKey, &mdbVal, 0));
                nextExampleId++;
            }
        });
    }
}
