"""Ultrametric quantum kernels: exact p-adic feature maps for hierarchical data."""

from padic_kernel.encoding import Encoding, EncodingFactory, register_encoding
from padic_kernel.kernels import (
    depolarise,
    fidelity_gram,
    overlap_gram,
    register_dimension,
    sample_kernel,
)
from padic_kernel.metrics import (
    dimension_lower_bound,
    gromov_delta,
    kernel_target_alignment,
    level_constancy,
    profile_residual,
    qubit_lower_bound,
    regular_tree_dimension_bound,
    regular_tree_qubit_bound,
    resolution_depth,
    strong_triangle_violations,
    violation_rate,
)
from padic_kernel.profiles import (
    Profile,
    delta_profile,
    geometric_profile,
    linear_profile,
    uniform_profile,
)
from padic_kernel.reupload import ReuploadEncoding, haar_unitary
from padic_kernel.tree import (
    RootedTree,
    ancestor_matrix,
    build_padic_tree,
    digit_matrix,
    from_parent_map,
    lca_depth_matrix,
    pad_to_uniform_depth,
)
from padic_kernel.utils import env_info, make_output_dir, set_seed, write_json

__all__ = [
    "Encoding",
    "EncodingFactory",
    "Profile",
    "ReuploadEncoding",
    "RootedTree",
    "ancestor_matrix",
    "build_padic_tree",
    "delta_profile",
    "depolarise",
    "digit_matrix",
    "dimension_lower_bound",
    "env_info",
    "fidelity_gram",
    "from_parent_map",
    "geometric_profile",
    "gromov_delta",
    "haar_unitary",
    "kernel_target_alignment",
    "lca_depth_matrix",
    "level_constancy",
    "linear_profile",
    "make_output_dir",
    "overlap_gram",
    "pad_to_uniform_depth",
    "profile_residual",
    "qubit_lower_bound",
    "register_dimension",
    "register_encoding",
    "regular_tree_dimension_bound",
    "regular_tree_qubit_bound",
    "resolution_depth",
    "sample_kernel",
    "set_seed",
    "strong_triangle_violations",
    "uniform_profile",
    "violation_rate",
    "write_json",
]
