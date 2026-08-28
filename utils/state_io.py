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
        coi_key_name,
        enqueue_state_serialized_key_index_write,
        state_key_from_id,
    )
    return (
        state_key_from_id,
        coi_key_name,
        enqueue_state_serialized_key_index_write,
    )

__all__ = [
    'cached_proof_obligation_id',
    'get_state', 'get_state_batch', 'get_state_only',
    'get_state_payload_bytes', 'get_state_payload_batch', 'put_state',
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
    state_key_from_id, coi_key_name, *_ = _key_mints()
    proof_obligation_id = cached_proof_obligation_id(serialized_state_key)
    pipe = state_cache.redis_runtime.pipeline()
    pipe.get(state_key_from_id(proof_obligation_id))
    pipe.get(coi_key_name(proof_obligation_id))
    serialized_state, serialized_coi = pipe.execute()
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        iterator = None
        if serialized_coi:
            iterator = pickle.loads(zlib.decompress(serialized_coi))
        binder = getattr(state, "bind_iterator_checkpoint", None)
        if callable(binder):
            binder(iterator)
        else:
            state.iterator = iterator
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
    state_key_from_id, coi_key_name, *_ = _key_mints()
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
        iterator = None
        if ser_coi:
            iterator = pickle.loads(zlib.decompress(ser_coi))
        binder = getattr(state, "bind_iterator_checkpoint", None)
        if callable(binder):
            binder(iterator)
        else:
            state.iterator = iterator
        if state.iterator:
            state.iterator.deserialize(state_cache)
        out[key] = state
    return out


def get_state_only(serialized_state_key: bytes, state_cache):
    state_key_from_id, *_ = _key_mints()
    proof_obligation_id = cached_proof_obligation_id(serialized_state_key)
    serialized_state = state_cache.redis_runtime.get(state_key_from_id(proof_obligation_id))
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        return state
    else:
        return None


def get_state_payload_bytes(serialized_state_key: bytes, state_cache):
    """Return the exact persisted ``(State, iterator)`` payload bytes.

    This is the raw counterpart of :func:`get_state`: callers that must attest
    the worker-written persistence occurrence hash these bytes before any
    driver-side mutation.  No decompression, migration, or fallback occurs.
    ``(None, None)`` means the obligation State is absent; a present State with
    a missing iterator is returned as ``(state_bytes, None)`` so completeness
    remains explicit.
    """

    if not isinstance(serialized_state_key, bytes):
        raise TypeError("serialized_state_key must be bytes")
    state_key_from_id, coi_key_name, *_ = _key_mints()
    identity = cached_proof_obligation_id(serialized_state_key)
    pipe = state_cache.redis_runtime.pipeline()
    pipe.get(state_key_from_id(identity))
    pipe.get(coi_key_name(identity))
    state_payload, iterator_payload = pipe.execute()
    if state_payload is None:
        return None, None
    if not isinstance(state_payload, bytes):
        raise TypeError("persisted State payload must be bytes")
    if iterator_payload is not None and not isinstance(iterator_payload, bytes):
        raise TypeError("persisted iterator payload must be bytes")
    return state_payload, iterator_payload


def get_state_payload_batch(serialized_keys, state_cache):
    """Batch-read exact persisted payload pairs without decoding them."""

    keys = tuple(serialized_keys or ())
    if not keys:
        return {}
    if any(not isinstance(key, bytes) for key in keys):
        raise TypeError("serialized State payload keys must be bytes")
    if len(keys) != len(set(keys)):
        raise ValueError("serialized State payload keys must be distinct")
    state_key_from_id, coi_key_name, *_ = _key_mints()
    identities = [cached_proof_obligation_id(key) for key in keys]
    pipe = state_cache.redis_runtime.pipeline()
    for identity in identities:
        pipe.get(state_key_from_id(identity))
        pipe.get(coi_key_name(identity))
    raw = pipe.execute()
    if len(raw) != 2 * len(keys):
        raise RuntimeError("persisted State payload batch has wrong result count")
    result = {}
    for index, key in enumerate(keys):
        state_payload = raw[2 * index]
        iterator_payload = raw[2 * index + 1]
        if state_payload is None:
            if iterator_payload is not None:
                raise RuntimeError(
                    "persisted iterator exists without its State payload"
                )
            result[key] = (None, None)
            continue
        if not isinstance(state_payload, bytes):
            raise TypeError("persisted State payload must be bytes")
        if iterator_payload is not None and not isinstance(
            iterator_payload, bytes
        ):
            raise TypeError("persisted iterator payload must be bytes")
        result[key] = (state_payload, iterator_payload)
    return result


def put_state(serialized_state_key: bytes, state, state_cache):
    proof_obligation_id = cached_proof_obligation_id(serialized_state_key)
    from src.state.state_store import (
        enqueue_canonical_state_checkpoint_write,
        prepare_canonical_state_checkpoint_write,
    )

    try:
        prepared_state = prepare_canonical_state_checkpoint_write(
            proof_obligation_id,
            serialized_state_key,
            state,
        )
    except Exception as e:
        import traceback
        print(f"Error serializing State checkpoint: {e}")
        traceback.print_exc()
        _find_unpicklable(state.iterator)
        raise
    pipe = state_cache.redis_runtime.pipeline()
    enqueue_canonical_state_checkpoint_write(pipe, prepared_state)
    pipe.execute()


def _find_unpicklable(obj, path="root", _visited=None, _depth=0):
    """Recursively search for unpicklable objects (e.g. live cvc5 Terms)."""
    if _depth > 30:
        return
    if _visited is None:
        _visited = set()
    obj_id = id(obj)
    if obj_id in _visited:
        return
    _visited.add(obj_id)

    from interpreter.utils.cvc5_serde import SerializedCvc5TermV2
    if isinstance(obj, SerializedCvc5TermV2):
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
            _find_unpicklable(v, f"{path}['{k}']", _visited, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _find_unpicklable(v, f"{path}[{i}]", _visited, _depth + 1)
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            _find_unpicklable(v, f"{path}.{k}", _visited, _depth + 1)
