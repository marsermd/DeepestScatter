#pragma once
#include <functional>
#include <iostream>

#include <lmdb.h>

namespace DeepestScatter
{
    class Transaction
    {
    public:
        template<typename T>
        static T withTransaction(MDB_env* env, MDB_txn* parent, unsigned flags, const std::function<T(Transaction&)>& action);

        operator MDB_txn*() const;

    private:
        Transaction(MDB_env* env, MDB_txn* parent, unsigned flags);
        ~Transaction();

        void commit();
        void abort();

        MDB_txn* transaction = nullptr;
    };

    template<typename T>
    T Transaction::withTransaction(MDB_env* env, MDB_txn* parent, unsigned flags,
        const std::function<T(Transaction&)>& action)
    {
        Transaction transaction(env, parent, flags);
        try
        {
            T t = action(transaction);
            transaction.commit();
            return t;
        }
        catch (std::exception& e)
        {
            if (transaction.transaction != nullptr)
            {
                std::cout << "Dataset transaction aborted because of " << e.what() << std::endl;
                transaction.abort();
            }
            throw;
        }
    }

    template<>
    inline void Transaction::withTransaction(MDB_env* env, MDB_txn* parent, unsigned flags,
        const std::function<void(Transaction&)>& action)
    {
        Transaction transaction(env, parent, flags);
        try
        {
            action(transaction);
            transaction.commit();
        }
        catch (std::exception& e)
        {
            std::cout << "Dataset transaction aborted because of " << e.what() << std::endl;
            transaction.abort();
            throw;
        }
    }
}