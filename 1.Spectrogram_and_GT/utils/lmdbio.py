import lmdb

def read_lmdb_data(lmdb_loc):
    env = lmdb.open(lmdb_loc, max_readers=1, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
    return len(keys) / 2