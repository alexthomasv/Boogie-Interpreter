"""Redis state serialization: get/put for State, COI, and DataFlow objects."""

import pickle
import zlib
from functools import lru_cache


def _key_mints():
    """The wire-frozen proof-state key mints, resolved lazily.

    Minted ONCE in ``src.state.state_store`` so the spelling cannot fork
    between this hot path and the driver/StateCache deleters/maskers.
    Lazy for the same reason ``cached_proof_obligation_id`` lazily imports
    persistence: ``src.state`` package init reaches back into
    ``interpreter.utils`` (cvc5 serde), so a module-level import here is a
    cycle. The module is cached after the first call — per-call cost is a
    dict lookup, noise against the Redis round-trip these keys gate.
    """
    from src.state.state_store import (
        coi_key_name, state_key_from_id, state_serialized_key_name)
    return state_key_from_id, coi_key_name, state_serialized_key_name

__all__ = [
    'cached_proof_obligation_id',
    'get_state', 'get_state_batch', 'get_state_only', 'get_state_raw',
    'put_state', 'create_df_key', 'put_df', 'get_df',
    'find_unpicklable',
]


@lru_cache(maxsize=10000)
def cached_proof_obligation_id(data: bytes) -> str:
    """proof_obligation_id for already-serialized obligation bytes (cached).

    This is THE state-key digest — a term-canonical hash over (pc, predicate).
    No raw-sha256 fallback: it would diverge from the canonical key for the
    same obligation (RISK B). Malformed bytes raise.
    """
    from src.state.persistence import proof_obligation_id

    return proof_obligation_id(data)


def get_state(serialized_state_key: bytes, state_cache):
    state_key_from_id, coi_key_name, _ = _key_mints()
    proof_obligation_id = cached_proof_obligation_id(serialized_state_key)
    pipe = state_cache.redis_runtime.pipeline()
    pipe.get(state_key_from_id(proof_obligation_id))
    pipe.get(coi_key_name(proof_obligation_id))
    serialized_state, serialized_coi = pipe.execute()
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        if serialized_coi:
            state.iterator = pickle.loads(zlib.decompress(serialized_coi))
        if state.iterator:
            state.iterator.deserialize(state_cache)
        return state
    else:
        return None


def get_state_batch(serialized_keys, state_cache):
    """Batch-fetch states for many keys in a single Redis round-trip.

    Returns ``{serialized_key: state | None}``. ``state.iterator`` is deserialized
    when present, matching :func:`get_state`. Use this instead of calling
    :func:`get_state` in a loop when you already have the full key set —
    collapses ``2*N`` round-trips to one pipeline.
    """
    if not serialized_keys:
        return {}
    state_key_from_id, coi_key_name, _ = _key_mints()
    keys = list(serialized_keys)
    proof_obligation_ids = [cached_proof_obligation_id(k) for k in keys]
    pipe = state_cache.redis_runtime.pipeline()
    for h in proof_obligation_ids:
        pipe.get(state_key_from_id(h))
        pipe.get(coi_key_name(h))
    raw = pipe.execute()
    out = {}
    for i, key in enumerate(keys):
        ser_state = raw[2 * i]
        ser_coi = raw[2 * i + 1]
        if not ser_state:
            out[key] = None
            continue
        state = pickle.loads(zlib.decompress(ser_state))
        if ser_coi:
            state.iterator = pickle.loads(zlib.decompress(ser_coi))
        if state.iterator:
            state.iterator.deserialize(state_cache)
        out[key] = state
    return out


def get_state_only(serialized_state_key: bytes, state_cache):
    state_key_from_id, _, _ = _key_mints()
    proof_obligation_id = cached_proof_obligation_id(serialized_state_key)
    serialized_state = state_cache.redis_runtime.get(state_key_from_id(proof_obligation_id))
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        return state
    else:
        return None


def get_state_raw(redis_key: str, state_cache):
    serialized_state = state_cache.redis_runtime.get(redis_key)
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        if state.iterator:
            state.iterator.deserialize(state_cache)
        return state
    else:
        return None


def put_state(serialized_state_key: bytes, state, state_cache):
    serialized_state = pickle.dumps(state)
    try:
        serialized_coi = pickle.dumps(state.iterator)
    except Exception as e:
        import traceback
        print(f"Error serializing COI: {e}")
        traceback.print_exc()
        find_unpicklable(state.iterator)
        raise
    state_key_from_id, coi_key_name, state_serialized_key_name = _key_mints()
    proof_obligation_id = cached_proof_obligation_id(serialized_state_key)
    compressed_state = zlib.compress(serialized_state)
    compressed_coi = zlib.compress(serialized_coi)
    pipe = state_cache.redis_runtime.pipeline()
    pipe.set(state_key_from_id(proof_obligation_id), compressed_state)
    pipe.set(state_serialized_key_name(proof_obligation_id), serialized_state_key)
    pipe.set(coi_key_name(proof_obligation_id), compressed_coi)
    pipe.execute()


def create_df_key(target_serialized, serialized_key):
    proof_obligation_id_target = cached_proof_obligation_id(target_serialized)
    proof_obligation_id_key = cached_proof_obligation_id(serialized_key)
    return f"df_key_{proof_obligation_id_target}_{proof_obligation_id_key}"


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


def find_unpicklable(obj, path="root", _visited=None, _depth=0):
    """Recursively search for unpicklable objects (e.g. live cvc5 Terms)."""
    if _depth > 30:
        return
    if _visited is None:
        _visited = set()
    obj_id = id(obj)
    if obj_id in _visited:
        return
    _visited.add(obj_id)

    from interpreter.utils.utils_cvc5 import HollowCvc5Term
    if isinstance(obj, HollowCvc5Term):
        print(f"FOUND CULPRIT at {path}: {type(obj)} -> {obj}")
        return

    # Check if obj itself is unpicklable
    try:
        pickle.dumps(obj)
        return  # picklable, no need to recurse
    except Exception:
        pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            find_unpicklable(v, f"{path}['{k}']", _visited, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_unpicklable(v, f"{path}[{i}]", _visited, _depth + 1)
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            find_unpicklable(v, f"{path}.{k}", _visited, _depth + 1)
