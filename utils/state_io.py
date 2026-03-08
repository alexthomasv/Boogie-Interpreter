"""Redis state serialization: get/put for State, COI, and DataFlow objects."""

import pickle
import hashlib
import zlib
from functools import lru_cache

__all__ = [
    '_cached_sha256',
    'get_state', 'get_state_only', 'get_state_raw',
    'put_state', 'create_df_key', 'put_df', 'get_df',
    'find_unpicklable',
]


@lru_cache(maxsize=10000)
def _cached_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_state(serialized_state_key: bytes, state_cache):
    sha256_hex = _cached_sha256(serialized_state_key)
    pipe = state_cache.redis_runtime.pipeline()
    pipe.get(f"state_key_{sha256_hex}")
    pipe.get(f"coi_key_{sha256_hex}")
    serialized_state, serialized_coi = pipe.execute()
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        if serialized_coi:
            state.coi = pickle.loads(zlib.decompress(serialized_coi))
        if state.coi:
            state.coi.deserialize(state_cache)
        return state
    else:
        return None


def get_state_only(serialized_state_key: bytes, state_cache):
    sha256_hex = _cached_sha256(serialized_state_key)
    serialized_state = state_cache.redis_runtime.get(f"state_key_{sha256_hex}")
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        return state
    else:
        return None


def get_state_raw(redis_key: str, state_cache):
    serialized_state = state_cache.redis_runtime.get(redis_key)
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        if state.coi:
            state.coi.deserialize(state_cache)
        return state
    else:
        return None


def put_state(serialized_state_key: bytes, state, state_cache):
    serialized_state = pickle.dumps(state)
    try:
        serialized_coi = pickle.dumps(state.coi)
    except Exception as e:
        print(f"Error serializing state: {e}")
        find_unpicklable(state.coi)
        assert False, f"Error serializing state: {e}"
    sha256_hex = _cached_sha256(serialized_state_key)
    compressed_state = zlib.compress(serialized_state)
    compressed_coi = zlib.compress(serialized_coi)
    pipe = state_cache.redis_runtime.pipeline()
    pipe.set(f"state_key_{sha256_hex}", compressed_state)
    pipe.set(f"coi_key_{sha256_hex}", compressed_coi)
    pipe.execute()


def create_df_key(target_serialized, serialized_key):
    sha256_hex_target = _cached_sha256(target_serialized)
    sha256_hex_key = _cached_sha256(serialized_key)
    return f"df_key_{sha256_hex_target}_{sha256_hex_key}"


def put_df(df, df_serialized_key, state_cache):
    serialized_df = pickle.dumps(df)
    compressed_df = zlib.compress(serialized_df)
    state_cache.redis_runtime.set(df_serialized_key, compressed_df)


def get_df(df_serialized_key, state_cache):
    compressed_df = state_cache.redis_runtime.get(df_serialized_key)
    if compressed_df:
        df = pickle.loads(zlib.decompress(compressed_df))
        df.deserialize(state_cache)
        return df
    else:
        return None


def find_unpicklable(obj, path="root"):
    from interpreter.utils.utils_cvc5 import HollowCvc5Term
    """
    Recursively searches for objects of type HollowCvc5Term.
    """
    if isinstance(obj, HollowCvc5Term):
        print(f"FOUND CULPRIT at {path}: {type(obj)} -> {obj}")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            find_unpicklable(v, f"{path}['{k}']")
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i, v in enumerate(obj):
            find_unpicklable(v, f"{path}[{i}]")
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            find_unpicklable(v, f"{path}.{k}")
