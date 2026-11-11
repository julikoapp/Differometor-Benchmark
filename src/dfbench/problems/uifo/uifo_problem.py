"""UIFO (Uniform Interferometer Field Optimization) problems with power constraints."""

import time
from contextlib import contextmanager

import jax
import numpy as np
from jaxtyping import Array, Float
from differometor.build import LINKING_FUNCTIONS
from differometor.setups import uifo, constrain_inter_grid_cell_spaces
from differometor.simulate import run_setups, simulate, run_build_step
from differometor.utils import (
    sensitivity_qamplfreq_noise,
    calculate_sensitivities,
    calculate_powers,
)


def _print_time(label, t0, out=None):
    """TIMING (temporary): print seconds since t0. Waits for JAX results in `out` first,
    since JAX returns before the computation has finished."""
    if out is not None:
        jax.block_until_ready(out)
    print(f"[uifo timing] {label}: {time.time() - t0:.3f} s", flush=True)

from ..base_problem import DEFAULT_SIGNAL_FLOOR, OpticalSetupProblem, register_problem
from .uifo_cache import cached_run_setups


# ---------------------------------------------------------------------------
# Per-problem linking functions
# ---------------------------------------------------------------------------
# differometor keeps the formulas of linked settings (e.g. amplitude = sqrt(power))
# in ONE global list, LINKING_FUNCTIONS (differometor/build.py:79):
#   * building a setup APPENDS its formulas and stores only their positions in the
#     setup's arrays (build.py:401-402; no formulas -> identity, position 0, :1414-1418);
#   * simulate() READS the whole list every time it is recorded/traced (simulate.py:278).
# So each UIFOProblem keeps its own copy of the list and puts it back exactly when
# simulate() runs. The global list is then only a temporary workbench, and emptying
# it before each new problem is safe for every UIFOProblem.
# Limit: other problem types (e.g. dfbench's VoyagerProblem) don't keep a copy; one
# built before a UIFOProblem in the same process could break if traced afterwards.
_LINKING_FUNCTIONS_AT_IMPORT = len(LINKING_FUNCTIONS)


def _reset_linking_functions():
    """Cut the global list back to how it was at import (normally empty).

    In place (del / slice assignment), never rebinding the name: simulate() holds a
    reference to this very list object, a new list would not be seen by it.
    """
    del LINKING_FUNCTIONS[_LINKING_FUNCTIONS_AT_IMPORT:]


@contextmanager
def _use_linking_functions(functions):
    """Inside the `with` block the global list holds exactly `functions`; afterwards
    it holds again whatever it held before. Used around every simulate() call."""
    saved = list(LINKING_FUNCTIONS)
    LINKING_FUNCTIONS[:] = functions
    try:
        yield
    finally:                         # restore even if simulate() raises
        LINKING_FUNCTIONS[:] = saved


# ---------------------------------------------------------------------------
# Topology string encoding
# ---------------------------------------------------------------------------
# Interior cells: 2 component types × 4 orientations = 8 options -> A-H
#   A-D = beamsplitter     (left, right, top, bottom)
#   E-H = directional_beamsplitter (left, right, top, bottom)
#
# Boundary cells: 4 component types -> L, S, D, H
#   L = laser, S = squeezer, D = detector, H = balanced_homodyne
#
# Format: "<interior_chars>-<boundary_chars>"
#   Interior chars: row-major order of the size×size interior grid
#   Boundary chars: row-major order of all boundary positions (edges only,
#                   no corners), scanning top row -> left/right cols -> bottom row
#
# Example for size=3:
#   Interior positions (row-major): 11,12,13,21,22,23,31,32,33
#   Boundary positions (row-major): 01,02,03,10,14,20,24,30,34,41,42,43
#   String: "AFCECCEA-SLLSSHLLASS" (21 chars)

_CENTER_TYPES = ["beamsplitter", "directional_beamsplitter"]
_CENTER_ORIENTATIONS = ["left", "right", "top", "bottom"]
_BOUNDARY_TYPES = ["laser", "squeezer", "detector", "balanced_homodyne"]

# Build encoding maps
_CENTER_TO_CHAR: dict[tuple[str, str], str] = {}
_CHAR_TO_CENTER: dict[str, tuple[str, str]] = {}
_char_idx = 0
for _comp in _CENTER_TYPES:
    for _orient in _CENTER_ORIENTATIONS:
        _ch = chr(ord("A") + _char_idx)
        _CENTER_TO_CHAR[(_comp, _orient)] = _ch
        _CHAR_TO_CENTER[_ch] = (_comp, _orient)
        _char_idx += 1

_BOUNDARY_TO_CHAR: dict[str, str] = {}
_CHAR_TO_BOUNDARY: dict[str, str] = {}
for _ch_label, _btype in zip("LSDH", _BOUNDARY_TYPES):
    _BOUNDARY_TO_CHAR[_btype] = _ch_label
    _CHAR_TO_BOUNDARY[_ch_label] = _btype


def _interior_positions(size: int) -> list[str]:
    """Return interior grid position keys in row-major order."""
    return [f"{r}{c}" for r in range(1, size + 1) for c in range(1, size + 1)]


def _boundary_positions(size: int) -> list[str]:
    """Return boundary position keys in row-major scan order (no corners)."""
    grid = size + 2  # total grid dimension including boundaries
    positions = []
    # Top edge row (row=0), skip corners
    for c in range(1, grid - 1):
        positions.append(f"0{c}")
    # Left and right edge columns (rows 1..grid-2)
    for r in range(1, grid - 1):
        positions.append(f"{r}0")
        positions.append(f"{r}{grid - 1}")
    # Bottom edge row (row=grid-1), skip corners
    for r_last in [grid - 1]:
        for c in range(1, grid - 1):
            positions.append(f"{r_last}{c}")
    return positions


def topology_to_string(
    centers: dict[str, tuple[str, str]],
    boundaries: dict[str, str],
    size: int,
) -> str:
    """Encode a UIFO topology as a compact string.

    Args:
        centers: Interior cell mapping, e.g. ``{"11": ("beamsplitter", "left"), ...}``.
        boundaries: Boundary cell mapping, e.g. ``{"01": "squeezer", ...}``.
        size: Grid size (e.g. 3 for 3×3 interior).

    Returns:
        A string like ``"AFCECCEA-SLLSSHLLASS"``.
    """
    interior_chars = []
    for pos in _interior_positions(size):
        comp, orient = centers[pos]
        interior_chars.append(_CENTER_TO_CHAR[(comp, orient)])

    boundary_chars = []
    for pos in _boundary_positions(size):
        if pos in boundaries:
            boundary_chars.append(_BOUNDARY_TO_CHAR[boundaries[pos]])
        # Skip positions not present (shouldn't happen for valid topologies)

    return "".join(interior_chars) + "-" + "".join(boundary_chars)


def topology_from_string(
    topology: str,
    size: int,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Decode a topology string into centers and boundaries dicts.

    Args:
        topology: Encoded string, e.g. ``"AFCECCEA-SLLSSHLLASS"``.
        size: Grid size.

    Returns:
        ``(centers, boundaries)`` tuple.
    """
    parts = topology.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Topology string must contain exactly one '-' separator, got: {topology!r}"
        )
    interior_part, boundary_part = parts

    interior_pos = _interior_positions(size)
    if len(interior_part) != len(interior_pos):
        raise ValueError(
            f"Expected {len(interior_pos)} interior chars for size={size}, "
            f"got {len(interior_part)}"
        )

    centers = {}
    for pos, ch in zip(interior_pos, interior_part):
        if ch not in _CHAR_TO_CENTER:
            raise ValueError(
                f"Invalid interior character '{ch}' at position {pos}. "
                f"Valid characters: {sorted(_CHAR_TO_CENTER.keys())}"
            )
        centers[pos] = _CHAR_TO_CENTER[ch]

    boundary_pos = _boundary_positions(size)
    if len(boundary_part) != len(boundary_pos):
        raise ValueError(
            f"Expected {len(boundary_pos)} boundary chars for size={size}, "
            f"got {len(boundary_part)}"
        )

    boundaries = {}
    for pos, ch in zip(boundary_pos, boundary_part):
        if ch not in _CHAR_TO_BOUNDARY:
            raise ValueError(
                f"Invalid boundary character '{ch}' at position {pos}. "
                f"Valid characters: {sorted(_CHAR_TO_BOUNDARY.keys())}"
            )
        boundaries[pos] = _CHAR_TO_BOUNDARY[ch]

    return centers, boundaries


@register_problem
class UIFOProblem(OpticalSetupProblem):
    """UIFO (Quasi-Universal Interferometer) optimization problem.

    Creates interferometer configurations in a grid pattern. The topology is
    fixed at initialization and only continuous optical parameters are optimized.

    There are three ways to specify the topology (mutually exclusive):

    1. **topology_seed** | Generate a random topology deterministically from a seed.
       This is the simplest way to get started::

           UIFOProblem(size=3, topology_seed=42)

    2. **topology string** | A string encoding of the grid layout::

           UIFOProblem(size=3, topology="AECGCCHEG-SLLSSHLLLLS")

    3. **centers + boundaries dicts** | Explicit component placement::

           UIFOProblem(
               size=3,
               centers={"11": ("beamsplitter", "left"), ...},
               boundaries={"01": "squeezer", ...},
           )

    The topology string uses single-character codes:

    - **Interior cells** (beamsplitters): A-D = beamsplitter (left/right/top/bottom),
      E-H = directional_beamsplitter (left/right/top/bottom)
    - **Boundary cells**: L = laser, S = squeezer, D = detector, H = balanced_homodyne
    - Format: ``"<interior_chars>-<boundary_chars>"`` in row-major order.
    """

    _supports_power_penalty = True

    def __init__(
        self,
        size: int = 3,
        n_frequencies: int = 50,
        topology_seed: int | None = None,
        topology: str | None = None,
        centers: dict[str, tuple[str, str]] | None = None,
        boundaries: dict[str, str] | None = None,
        power_penalty_fn=None,
        bounds_overrides: dict[str, tuple[float, float]] | None = None,
        signal_floor: float = DEFAULT_SIGNAL_FLOOR,
    ):
        """Initialize the UIFO optimization problem.

        Args:
            size: Grid size (e.g., 3 for 3x3, 5 for 5x5). Defaults to 3.
            n_frequencies: Number of frequency points. Defaults to 50.
            topology_seed: Seed for random topology generation. Defaults to ``None``.
                Pass ``None`` (with no ``topology`` or ``centers``/``boundaries``)
                to generate a random topology.  Mutually exclusive
                with ``topology`` and ``centers``/``boundaries``.
            topology: Compact topology string (see class docstring for format).
                Mutually exclusive with ``topology_seed`` and ``centers``/``boundaries``.
            centers: Interior cell placement dict. Must be provided together with
                ``boundaries``. Mutually exclusive with ``topology_seed`` and ``topology``.
            boundaries: Boundary cell placement dict. Must be provided together with
                ``centers``. Mutually exclusive with ``topology_seed`` and ``topology``.
            power_penalty_fn: A callable ``fn(value, threshold) -> penalty`` applied
                per-element to compute power-constraint violations.  Built-in
                options are ``squashed_relu_penalty`` (default),
                ``relu_penalty``, and ``zero_penalty`` from
                ``dfbench.problems.base_problem``.
            bounds_overrides: Optional property-level bound overrides.
                Example: ``{"tuning": (0, 45)}``.
                Overrides must narrow default bounds.
            signal_floor: Optional lower bound for detector signal
                magnitudes before sensitivity normalization.
        """
        # --- Validate topology specification (exactly one path) ---
        has_seed = topology_seed is not None
        has_string = topology is not None
        has_dicts = (
            centers is not None or boundaries is not None
        )  # Handled by ValueError below

        n_specified = sum([has_seed, has_string, has_dicts])
        if n_specified == 0:
            # No topology specified; generate a truly random one
            topology_seed = int(np.random.randint(0, 2**31))
            has_seed = True
        elif n_specified > 1:
            raise ValueError(
                "Specify exactly one of: topology_seed, topology (string), "
                "or centers+boundaries. Got multiple."
            )

        if has_dicts and (centers is None or boundaries is None):
            raise ValueError(
                "Both 'centers' and 'boundaries' must be provided together."
            )

        # --- Resolve topology to centers + boundaries ---
        if has_seed:
            name = f"uifo_{size}x{size}_seed{topology_seed}"
        elif has_string:
            centers, boundaries = topology_from_string(topology, size)
            name = f"uifo_{size}x{size}_{topology}"
        else:
            name = f"uifo_{size}x{size}_custom"

        if has_seed:
            print(f"UIFOProblem topology seed: {topology_seed}")

        super().__init__(
            name=name,
            n_frequencies=n_frequencies,
            signal_floor=signal_floor,
        )
        signal_floor = self._signal_floor
        if power_penalty_fn is not None:
            self._power_penalty_fn = power_penalty_fn
        self._size = size
        self._topology_seed = topology_seed
        self._topology_string = None  # computed lazily or set below
        self._bounds_overrides = bounds_overrides

        # Start this problem's formulas at position 0: empty the global list before
        # anything is built (safe: earlier UIFOProblems carry their own copy).
        _reset_linking_functions()

        ### Calculate the target sensitivity using Voyager reference ###
        # -------------------------------------------------------------#

        # Import voyager for reference sensitivity
        from differometor.setups import voyager

        # use a predefined Voyager setup with three different modulations for reference
        _t = time.time()
        q_setup, _ = voyager(mode="space_modulation")
        _print_time("voyager (space_modulation)", _t)
        _t = time.time()
        ampl_setup, _ = voyager(mode="amplitude_modulation")
        _print_time("voyager (amplitude_modulation)", _t)
        _t = time.time()
        freq_setup, _ = voyager(mode="frequency_modulation")
        _print_time("voyager (frequency_modulation)", _t)
        reference_setups = [q_setup, ampl_setup, freq_setup]

        # choose a sensitivity function that calculates sensitivities taking into account the three noise sources
        self._sensitivity_function = sensitivity_qamplfreq_noise

        
        
        # todo: figure out is it the same for multiple seeds because we have the cache hit
        # print(f"reference_setups: {reference_setups}")
        # print(f"self._frequencies: {self._frequencies}")

        # simulate the reference setups; cached per (setups, frequencies), see uifo_cache.py
        _t = time.time()
        simulation_results, _hit = cached_run_setups(reference_setups, self._frequencies)
        _print_time(f"run_setups (Voyager reference, cache {'hit' if _hit else 'miss'})",
                    _t, simulation_results)

        # calculate the sensitivity values taking into account the three noise sources
        _t = time.time()
        self._target_sensitivities = calculate_sensitivities(
            simulation_results,
            self._sensitivity_function,
            self._frequencies,
            True,
            signal_floor,
        )
        _print_time("calculate_sensitivities (Voyager reference)", _t, self._target_sensitivities)

        ### Create UIFO setup ###
        # -----------------------#

        _t = time.time()
        if has_seed:
            # Generate topology from seed
            q_noise_setup, component_property_pairs, centers, boundaries = uifo(
                size=size,
                mode="space_modulation",
                random=True,
                verbose=True,
                random_seed=topology_seed,
            )
        else:
            # Use provided centers and boundaries
            q_noise_setup, component_property_pairs, centers, boundaries = uifo(
                size=size,
                mode="space_modulation",
                random=True,
                verbose=True,
                centers=centers,
                boundaries=boundaries,
            )
        _print_time("uifo (q, space_modulation)", _t)

        _t = time.time()
        ampl_noise_setup, _ = uifo(
            size=size,
            mode="amplitude_modulation",
            centers=centers,
            boundaries=boundaries,
        )
        _print_time("uifo (amplitude_modulation)", _t)
        _t = time.time()
        freq_noise_setup, _ = uifo(
            size=size,
            mode="frequency_modulation",
            centers=centers,
            boundaries=boundaries,
        )
        _print_time("uifo (frequency_modulation)", _t)

        self._setup = [q_noise_setup, ampl_noise_setup, freq_noise_setup]
        self._centers = centers
        self._boundaries = boundaries

        # check if the random uifo uses a balanced homodyne detection scheme
        self._homodyne = False
        for node in q_noise_setup.nodes:
            if node[1]["component"] == "qhd":
                self._homodyne = True
                break

        ### Setup optimization parameters ###
        # ----------------------------------#

        # select properties to be optimized
        optimized_properties = [
            "reflectivity",
            "tuning",
            "db",
            "angle",
            "power",
            "mass",
            "length",
        ]

        # specify the ranges for the properties to be optimized
        property_bounds = {
            "db": [0, 10],
            "angle": [-360, 360],
            "power": [0, 200],
            "tuning": [-360, 360],
            "mass": [0.01, 200],
            "length": [0.1, 4000],
            "reflectivity": [0, 1],
        }
        property_bounds = self._apply_property_bounds_overrides(
            property_bounds,
            bounds_overrides,
        )

        # couple vertical and horizontal spaces at same positions, so that the grid structure of the uifo is always preserved
        _t = time.time()
        self._optimization_pairs = constrain_inter_grid_cell_spaces(
            component_property_pairs, optimized_properties
        )
        _print_time("constrain_inter_grid_cell_spaces", _t)

        # calculate the bounds for the properties to be optimized
        lower_bounds = []
        upper_bounds = []
        for optimization_pair in self._optimization_pairs:
            property_name = self._property_name_from_optimization_pair(
                optimization_pair
            )
            lower_bounds.append(property_bounds[property_name][0])
            upper_bounds.append(property_bounds[property_name][1])
        self._bounds = np.array([lower_bounds, upper_bounds])

        # build the three modulation setups and store as instance attributes
        _t = time.time()
        self._q_arrays, *self._q_metadata = run_build_step(
            self._setup[0],
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )
        _print_time("run_build_step (q)", _t, self._q_arrays)
        _t = time.time()
        self._ampl_arrays, *self._ampl_metadata = run_build_step(
            self._setup[1],
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )
        _print_time("run_build_step (ampl)", _t, self._ampl_arrays)
        _t = time.time()
        self._freq_arrays, *self._freq_metadata = run_build_step(
            self._setup[2],
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )
        _print_time("run_build_step (freq)", _t, self._freq_arrays)

        # All builds of this problem are done (Voyager reference + 3 topology setups):
        # the global list now holds exactly the formulas its arrays point to. Keep a copy.
        self._linking_functions = list(LINKING_FUNCTIONS)

        self._build_objective_function()

        # Compute and cache topology string
        self._topology_string = topology_to_string(
            self._centers, self._boundaries, size
        )

    def _eval_core(self, optimized_parameters):
        """Shared evaluation body for the UIFO objective.

        Runs the three modulation simulations, computes sensitivities and
        per-group powers, then the loss tuple. Used by both
        ``objective_function`` and ``objective_function_aux`` so the two
        stay in sync after ``_build_objective_function`` re-traces them.
        """
        # TIMING: this body runs inside jax.jit, so these timers measure TRACING only
        # (building the graph). They print once per compile, not on every evaluation,
        # and nothing can be waited on here (the values are placeholders, not numbers).
        # simulate() reads the global formula list here: give it this problem's copy.
        with _use_linking_functions(self._linking_functions):
            _t = time.time()
            q_results = simulate(
                **{**self._q_arrays, "optimized_parameters": optimized_parameters}
            )
            _print_time("simulate (q) [trace]", _t)
            _t = time.time()
            ampl_results = simulate(
                **{**self._ampl_arrays, "optimized_parameters": optimized_parameters}
            )
            _print_time("simulate (ampl) [trace]", _t)
            _t = time.time()
            freq_results = simulate(
                **{**self._freq_arrays, "optimized_parameters": optimized_parameters}
            )
            _print_time("simulate (freq) [trace]", _t)
        results = [
            (*q_results, *self._q_metadata),
            (*ampl_results, *self._ampl_metadata),
            (*freq_results, *self._freq_metadata),
        ]

        _t = time.time()
        sensitivities = calculate_sensitivities(
            results,
            self._sensitivity_function,
            self._frequencies,
            self._homodyne,
            self._signal_floor,
        )
        _print_time("calculate_sensitivities [trace]", _t)
        _t = time.time()
        powers = calculate_powers(q_results[0], *self._q_metadata)
        _print_time("calculate_powers [trace]", _t)
        sensitivity_loss, penalty, violations = self._calculate_loss(
            sensitivities, self._target_sensitivities, powers
        )
        return powers, sensitivity_loss, penalty, violations

    def _build_objective_function(self) -> None:
        """(Re)build the JIT-compiled objective and aux objective.

        Re-tracing picks up the current ``_power_penalty_fn`` so that
        ``set_penalty_fn`` takes effect on subsequent evaluations. Both
        the plain and the aux variant close over the same ``_eval_core``
        call so their results agree up to the returned aux dict.
        """

        @jax.jit
        def objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            _, sensitivity_loss, penalty, _ = self._eval_core(optimized_parameters)
            return sensitivity_loss + penalty

        @jax.jit
        def objective_function_aux(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> tuple[Float, dict]:
            powers, sensitivity_loss, penalty, violations = self._eval_core(
                optimized_parameters
            )
            aux = self._build_aux(powers, sensitivity_loss, penalty, violations)
            return sensitivity_loss + penalty, aux

        self.objective_function = objective_function
        self.objective_function_aux = objective_function_aux

    @property
    def optimization_pairs(self) -> list[tuple]:
        """List of (component, property) pairs to be optimized."""
        return self._optimization_pairs

    @property
    def bounds(self) -> Float[Array, "2 {self.n_params}"]:
        """Bounds for each parameter to be optimized. Shape: (2, n_params)."""
        return self._bounds

    @property
    def n_params(self) -> int:
        """Number of parameters to be optimized. Is equal to len(self.optimization_pairs)."""
        return len(self._optimization_pairs)

    @property
    def topology_seed(self) -> int | None:
        """The seed used to generate this problem's topology, or None if topology was specified directly."""
        return self._topology_seed

    @property
    def topology_string(self) -> str:
        """Compact string encoding of the topology."""
        return self._topology_string

    @property
    def centers(self) -> dict[str, tuple[str, str]]:
        """Interior cell placement dict."""
        return self._centers

    @property
    def boundaries(self) -> dict[str, str]:
        """Boundary cell placement dict."""
        return self._boundaries

    @property
    def structure_info(self) -> dict:
        """Metadata about the problem's discrete structure."""
        return {
            "size": self._size,
            "topology_seed": self._topology_seed,
            "topology_string": self._topology_string,
            "n_params": self.n_params,
            "homodyne": self._homodyne,
            "signal_floor": self._signal_floor,
            "power_penalty_fn": getattr(
                self._power_penalty_fn, "__name__", str(self._power_penalty_fn)
            ),
        }

    def to_spec(self) -> dict:
        """Return a serializable spec sufficient to rebuild this problem.

        Reconstruction uses the explicit ``topology_string`` (deterministic,
        independent of the RNG used to generate the topology) plus ``size``,
        so an equivalent problem can be rebuilt in any process without
        relying on the original seed.
        """
        spec = self._base_spec()
        spec["type"] = "UIFOProblem"
        spec["size"] = int(self._size)
        spec["topology"] = str(self._topology_string)
        if self._bounds_overrides:
            spec["bounds_overrides"] = {
                k: [float(v[0]), float(v[1])] for k, v in self._bounds_overrides.items()
            }
        return spec

    def calculate_sensitivity(
        self,
        optimized_parameters: Float[Array, "{self.n_params}"],
    ) -> Float[Array, "n_frequencies"]:
        """Calculate the sensitivity curve for given parameters.

        Args:
            optimized_parameters: Parameters to evaluate.

        Returns:
            Sensitivity values at each frequency point.
        """
        # simulate the three modulation setups (not via _eval_core, so it needs the
        # same swap of the formula list; called e.g. by core/storage/exporter.py:226)
        with _use_linking_functions(self._linking_functions):
            q_results = simulate(
                **{**self._q_arrays, "optimized_parameters": optimized_parameters}
            )
            ampl_results = simulate(
                **{**self._ampl_arrays, "optimized_parameters": optimized_parameters}
            )
            freq_results = simulate(
                **{**self._freq_arrays, "optimized_parameters": optimized_parameters}
            )
        results = [
            (*q_results, *self._q_metadata),
            (*ampl_results, *self._ampl_metadata),
            (*freq_results, *self._freq_metadata),
        ]

        # calculate the sensitivities taking into account the three noise sources
        sensitivities = calculate_sensitivities(
            results,
            self._sensitivity_function,
            self._frequencies,
            self._homodyne,
            self._signal_floor,
        )

        return sensitivities


# Backwards compatibility alias
RandomUIFOProblem = UIFOProblem
