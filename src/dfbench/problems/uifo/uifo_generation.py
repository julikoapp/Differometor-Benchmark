# NOT ACTIVE FILE, NOT USED ANYWHERE
# SERVED FOR DESIGN-RELATED PURPOSES
# SHOULD BE DELETED IN THE FUTURE IF NOT USED

from dataclasses import dataclass
# from differometer.uifo ...

@dataclass
class UIFOArtifact:
    size: int
    random_seed: int
    centers: dict
    boundaries: dict
    component_property_pairs: list
    q_noise_setup: object
    ampl_noise_setup: object
    freq_noise_setup: object


def generate_uifo_artifact(size, random_seed):
    q_noise_setup, component_property_pairs, centers, boundaries = uifo(
        size=size,
        mode="space_modulation",
        random=True,
        verbose=True,
        random_seed=random_seed,
    )

    ampl_noise_setup, _ = uifo(
        size=size,
        mode="amplitude_modulation",
        centers=centers,
        boundaries=boundaries,
    )

    freq_noise_setup, _ = uifo(
        size=size,
        mode="frequency_modulation",
        centers=centers,
        boundaries=boundaries,
    )

    return UIFOArtifact(
        size=size,
        random_seed=random_seed,
        centers=centers,
        boundaries=boundaries,
        component_property_pairs=component_property_pairs,
        q_noise_setup=q_noise_setup,
        ampl_noise_setup=ampl_noise_setup,
        freq_noise_setup=freq_noise_setup,
    )


# possible thing for caching too
# self._optimization_pairs = constrain_inter_grid_cell_spaces(
#     component_property_pairs, optimized_properties
# )