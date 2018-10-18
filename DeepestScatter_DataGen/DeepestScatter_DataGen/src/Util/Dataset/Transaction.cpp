#include "Transaction.h"
#include "LmdbExceptions.h"

namespace DeepestScatter
{
    Transaction::Transaction(MDB_env* env, MDB_txn* parent, unsigned flags)
    {
        LmdbExceptions::checkError(mdb_txn_begin(env, parent, flags, &transaction));
    }

    Transaction::~Transaction()
    {
        if (transaction != nullptr)
        {
            std::cerr << "Transaction was not closed! Aborting." << std::endl;
        }
    }

    Transaction::operator MDB_txn*() const
    {
        return transaction;
    }

    void Transaction::commit()
    {
        if (transaction == nullptr)
        {
            return;
        }

        const auto tempTransaction = transaction;
        transaction = nullptr;
        LmdbExceptions::checkError(mdb_txn_commit(tempTransaction));
    }

    void Transaction::abort()
    {
        if (transaction == nullptr)
        {
            return;
        }

        const auto tempTransaction = transaction;
        transaction = nullptr; 
        mdb_txn_abort(tempTransaction);
    }

}
