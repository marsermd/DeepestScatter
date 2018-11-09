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
        size_t getRecordsCount();

        template<class T>
        T getRecord(int32_t recordId);

        template<class T>
        void dropTable();

        template<class T>
        void append(const T& example);

        template<class T>
        void batchAppend(const gsl::span<T>& examples, int32_t startId);

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
        void tryBatchAppend(const gsl::span<T>& examples, int nextExampleId);

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
    size_t Dataset::getRecordsCount()
    {
        const TableName tableName = T::descriptor()->full_name();

        MDB_dbi dbi = getTable(tableName);

        return Transaction::withTransaction<size_t>(mdbEnv, nullptr, 0, [&](Transaction& transaction)
        {
            MDB_stat stats;
            LmdbExceptions::checkError(mdb_stat(transaction, dbi, &stats));
            return stats.ms_entries;
        });
    }

    template <class T>
    T Dataset::getRecord(int32_t recordId)
    {
        const TableName tableName = T::descriptor()->full_name();
        MDB_dbi dbi = getTable(tableName);

        return Transaction::withTransaction<T>(mdbEnv, nullptr, 0, [&](Transaction& transaction)
        {
            MDB_val mdbKey
            {
                sizeof(int32_t),
                &recordId
            };

            MDB_val mdbVal;
            LmdbExceptions::checkError(mdb_get(transaction, dbi, &mdbKey, &mdbVal));

            T record{};
            record.ParseFromArray(mdbVal.mv_data, mdbVal.mv_size);

            return record;
        });
    }

    template <class T>
    void Dataset::dropTable()
    {
        const TableName tableName = T::descriptor()->full_name();
        MDB_dbi dbi = getTable(tableName);

        while (true)
        {
            std::string tableConfirmation;

            std::cout << "YOU ARE GOING TO DELETE " << getRecordsCount<T>() << " RECORDS!!!" << std::endl;
            std::cout << "Type table name: \"" << T::descriptor()->name() << "\" to drop it" << std::endl;
            std::cin >> tableConfirmation;

            if (tableConfirmation == T::descriptor()->name())
            {
                break;
            }

            std::cout << "Mismatched table name. Try again.";
        }

        Transaction::withTransaction<void>(mdbEnv, nullptr, 0, [&](Transaction& transaction)
        {
            mdb_drop(transaction, dbi, 0);
            nextIds[tableName] = 0;
        });
    }

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
    void Dataset::batchAppend(const gsl::span<T>& examples, int32_t startId)
    {
        const TableName tableName = T::descriptor()->full_name();

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
    void Dataset::tryBatchAppend(const gsl::span<T>& examples, int nextExampleId)
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
