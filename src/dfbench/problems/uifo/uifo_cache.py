# for dfbench..
import hashlib
from copy import deepcopy

import numpy as np
from differometor.build import LINKING_FUNCTIONS
from differometor.simulate import run_setups


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
# run_setups cache (Voyager reference simulation in UIFOProblem.__init__)
# ---------------------------------------------------------------------------
# key   = the two arguments of run_setups(setups, frequencies): a fingerprint of the
#         setups' contents + the frequency values.
# value = (simulation results, linking formulas the call appended).
# In memory only: the first UIFOProblem of a process computes it, all later ones reuse it.
_RUN_SETUPS_CACHE = {}


def _setup_fingerprint(setup):
    """Everything a Setup holds (setups.py:208-211): optimizable parameters, nodes, edges.
    repr() of the formulas inside (e.g. jnp.sqrt) is stable within one process, which is
    all an in-memory cache needs."""
    nodes = sorted(setup._nodes.items(), key=lambda kv: repr(kv[0]))
    edges = sorted(setup._edges.items(), key=lambda kv: repr(kv[0]))
    return repr((setup.parameters, nodes, edges))


def _run_setups_key(setups, frequencies):
    f = np.asarray(frequencies)
    h = hashlib.sha1()
    for setup in setups:
        h.update(_setup_fingerprint(setup).encode())
    h.update(repr((f.shape, str(f.dtype))).encode())
    h.update(f.tobytes())
    return h.hexdigest()


def cached_run_setups(setups, frequencies):
    """run_setups(setups, frequencies) computed once per process.

    Returns (results, hit). Besides the results, run_setups has a side effect we
    have to replay: every build it does appends linking formulas to differometor's
    global LINKING_FUNCTIONS list (build.py:401, :1418), and the topology built
    afterwards is compiled with that whole list (simulate.py:278); topologies
    without linked parameters even point at entry 0 of it (build.py:1414-1418).
    On a hit we append the same formula objects again, so the list - and so the
    compiled program and its persistent-cache key - is exactly as without the cache.
    """
    key = _run_setups_key(setups, frequencies)
    hit = key in _RUN_SETUPS_CACHE
    if not hit:
        n_before = len(LINKING_FUNCTIONS)
        results = run_setups(setups, frequencies)
        _RUN_SETUPS_CACHE[key] = (results, list(LINKING_FUNCTIONS[n_before:]))
    results, appended = _RUN_SETUPS_CACHE[key]
    if hit:
        LINKING_FUNCTIONS.extend(appended)
    # No deepcopy: the results are JAX arrays (immutable) plus metadata that
    # calculate_sensitivities only reads.
    return results, hit


