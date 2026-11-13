# for dfbench..
import hashlib
import importlib.metadata
import os
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from differometor.simulate import run_setups
from differometor.utils import calculate_sensitivities


_UIFO_CACHE = {}


def get_uifo_artifact(size, random_seed):
    # Imported here, not at the top: uifo_generation.py does not import yet (see the
    # note at the end of this file), and a top-level import would break every
    # `from dfbench.problems import UIFOProblem`.
    from .uifo_generation import generate_uifo_artifact

    key = (size, random_seed)

    if key not in _UIFO_CACHE:
        _UIFO_CACHE[key] = generate_uifo_artifact(
            size=size,
            random_seed=random_seed,
        )

    # Return independent objects so callers cannot mutate cached setups.
    return deepcopy(_UIFO_CACHE[key])


# ---------------------------------------------------------------------------
# Voyager reference cache (target sensitivities in UIFOProblem.__init__)
# ---------------------------------------------------------------------------
# UIFOProblem.__init__ simulates the fixed Voyager detector (run_setups) only to get
# one small array: the target sensitivities its loss compares against. Voyager and
# the frequencies are the same for every seed, so that array is computed once and
# then reused:
#   * in memory : later UIFOProblems in the same process;
#   * on disk   : later processes/jobs, if the environment variable
#                 DFBENCH_CACHE_DIR names a folder (unset = memory only).
# key = everything the result depends on: the contents of the reference setups,
#       the frequencies, the sensitivity function, signal_floor, the homodyne flag,
#       the differometor version and the JAX backend (CPU and GPU round differently).
_TARGET_CACHE = {}
CACHE_DIR_ENV = "DFBENCH_CACHE_DIR"


def _stable(obj):
    """A description of `obj` that is identical in every process (unlike repr(),
    which prints memory addresses for functions), so it can be part of a disk key."""
    if isinstance(obj, dict):
        items = ((_stable(k), _stable(v)) for k, v in obj.items())
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(_stable(x) for x in obj))
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):          # numpy / jax array
        a = np.asarray(obj)
        return ("array", a.shape, str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest())
    if callable(obj):                                            # e.g. jnp.sqrt, a lambda
        code = getattr(obj, "__code__", None)
        where = (code.co_filename.rsplit("/", 1)[-1], code.co_firstlineno) if code else None
        captured = tuple(_stable(c.cell_contents) for c in (getattr(obj, "__closure__", None) or ()))
        return ("fn", getattr(obj, "__module__", None),
                getattr(obj, "__qualname__", type(obj).__name__), where, captured)
    return repr(obj)                                             # numbers, strings, None


def _setup_fingerprint(setup):
    """Everything a Setup holds (setups.py:208-211): optimizable parameters, nodes, edges."""
    return _stable((setup.parameters, setup._nodes, setup._edges))


def _target_key(setups, frequencies, sensitivity_function, signal_floor, homodyne):
    try:
        differometor_version = importlib.metadata.version("differometor")
    except importlib.metadata.PackageNotFoundError:
        differometor_version = "unknown"
    parts = (
        [_setup_fingerprint(s) for s in setups],
        _stable(frequencies),
        _stable(sensitivity_function),
        repr(float(signal_floor)),
        bool(homodyne),
        differometor_version,
        jax.default_backend(),
    )
    return hashlib.sha1(repr(parts).encode()).hexdigest()


def _disk_path(key):
    cache_dir = os.environ.get(CACHE_DIR_ENV)
    if not cache_dir:
        return None
    return Path(cache_dir) / f"voyager_target_{key[:20]}.npy"


def cached_target_sensitivities(setups, frequencies, sensitivity_function,
                                signal_floor, homodyne=True):
    """Target sensitivities of the reference `setups`, computed at most once.

    Returns (target, source) with source "memory", "disk" or "computed".
    Computing = run_setups(setups, frequencies) + calculate_sensitivities(...),
    exactly what UIFOProblem.__init__ did before.
    """
    key = _target_key(setups, frequencies, sensitivity_function, signal_floor, homodyne)

    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key], "memory"

    path = _disk_path(key)
    if path is not None and path.exists():
        try:
            target = jnp.asarray(np.load(path))
            _TARGET_CACHE[key] = target
            return target, "disk"
        except Exception:             # unreadable/half-written file: recompute below
            pass

    results = run_setups(setups, frequencies)
    target = calculate_sensitivities(results, sensitivity_function, frequencies,
                                     homodyne, signal_floor)
    _TARGET_CACHE[key] = target

    if path is not None:
        # Write to a temporary file, then rename: several Slurm jobs may write the
        # same file at once, and a reader must never see a half-written one.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, np.asarray(target))
        os.replace(tmp, path)
    return target, "computed"
