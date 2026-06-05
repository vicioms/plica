import numpy as np
from scipy.fft import dst, idst
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


# -----------------------------------------------------------------------------
# Geometry / sampling
# -----------------------------------------------------------------------------


def curve_length(points):
    """Return the polyline arclength of points with shape (n, dim)."""
    points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        raise ValueError(f"points must have shape (n, dim), got {points.shape}.")

    if len(points) < 2:
        return 0.0

    return np.linalg.norm(np.diff(points, axis=0), axis=1).sum()


def resample_arclength(points, num, *, tol=0.0, bc_type="not-a-knot"):
    """
    Resample a polyline by normalized arclength.

    Parameters
    ----------
    points : array, shape (n, dim)
        Input curve samples.

    num : int
        Number of output samples.

    tol : float, optional
        Consecutive points separated by a distance <= tol are removed.

    bc_type : str or 2-tuple, optional
        Boundary condition passed to scipy.interpolate.CubicSpline.

    Returns
    -------
    points_new : array, shape (num, dim)
        Resampled points.

    s_new : array, shape (num,)
        Uniform normalized arclength parameter in [0, 1].
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        raise ValueError(f"points must have shape (n, dim), got {points.shape}.")

    num = int(num)
    if num < 2:
        raise ValueError(f"num must be >= 2, got {num}.")

    if len(points) < 2:
        raise ValueError("Need at least two input points.")

    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], d > tol])
    points = points[keep]

    if len(points) < 2:
        raise ValueError("Curve has fewer than two distinct points after filtering.")

    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])

    if s[-1] <= 0:
        raise ValueError("Cannot resample a curve with zero arclength.")

    s = s / s[-1]
    s_new = np.linspace(0.0, 1.0, num)

    spline = CubicSpline(s, points, axis=0, bc_type=bc_type)
    return spline(s_new), s_new


# -----------------------------------------------------------------------------
# DST fit / reconstruction
# -----------------------------------------------------------------------------


def linear_interpolant(s, start, end):
    """Linear chord between start and end evaluated at s."""
    s = np.asarray(s, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return start + s[:, None] * (end - start)[None, :]


def fit_dst(points):
    """
    Fit a DST-I sine expansion to uniformly sampled open-curve data.

    The curve is represented as

        x(s) = start + s (end - start) + sum_k c_k sin(k pi s),

    with s uniformly sampled on [0, 1].

    Parameters
    ----------
    points : array, shape (num, dim)
        Uniform samples in normalized arclength.

    Returns
    -------
    coeffs : array, shape (num - 2, dim)
        Sine amplitudes c_k.

    start : array, shape (dim,)
        First point.

    end : array, shape (dim,)
        Last point.

    s : array, shape (num,)
        Uniform parameter.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        raise ValueError(f"points must have shape (num, dim), got {points.shape}.")

    if len(points) < 2:
        raise ValueError("Need at least two points.")

    s = np.linspace(0.0, 1.0, len(points))
    start = points[0].copy()
    end = points[-1].copy()

    residual = points - linear_interpolant(s, start, end)
    interior = residual[1:-1]
    num_modes = len(interior)

    if num_modes == 0:
        coeffs = np.zeros((0, points.shape[1]), dtype=float)
    else:
        coeffs = dst(interior, type=1, axis=0, norm="backward") / (num_modes + 1)

    return coeffs, start, end, s


def fit_dst_curve(points, num=None, *, tol=0.0, bc_type="not-a-knot"):
    """
    Resample a curve by arclength and fit a DST-I representation.

    Returns a dict with keys:
        points, s, coeffs, start, end, length
    """
    points = np.asarray(points, dtype=float)

    if num is None:
        num = len(points)

    points_s, s = resample_arclength(points, num, tol=tol, bc_type=bc_type)
    coeffs, start, end, s = fit_dst(points_s)

    return {
        "points": points_s,
        "s": s,
        "coeffs": coeffs,
        "start": start,
        "end": end,
        "length": curve_length(points_s),
    }


def sine_basis(s, num_modes):
    """Return Phi[i, k - 1] = sin(k pi s[i])."""
    s = np.asarray(s, dtype=float)
    mode = np.arange(1, int(num_modes) + 1)
    return np.sin(np.pi * s[:, None] * mode[None, :])


def apply_spectral_filter(coeffs, weights=None):
    """Apply per-mode spectral weights to DST coefficients."""
    coeffs = np.asarray(coeffs, dtype=float)

    if weights is None:
        return coeffs

    weights = np.asarray(weights, dtype=float)

    if weights.shape != (len(coeffs),):
        raise ValueError(f"weights must have shape {(len(coeffs),)}, got {weights.shape}.")

    return coeffs * weights[:, None]


def reconstruct_residual_dst(coeffs, s, *, weights=None):
    """Evaluate only the sine residual at parameter values s."""
    coeffs = apply_spectral_filter(coeffs, weights)
    Phi = sine_basis(s, len(coeffs))
    return Phi @ coeffs


def reconstruct_dst(coeffs, start, end, s, *, weights=None):
    """Evaluate the full DST curve at parameter values s."""
    return linear_interpolant(s, start, end) + reconstruct_residual_dst(
        coeffs,
        s,
        weights=weights,
    )


def reconstruct_dst_idst(coeffs, start, end, *, weights=None):
    """
    Same-grid reconstruction using scipy.fft.idst.

    This only reconstructs on the original grid with len(coeffs) + 2 points.
    """
    coeffs = apply_spectral_filter(coeffs, weights)
    num_modes = len(coeffs)
    num = num_modes + 2
    s = np.linspace(0.0, 1.0, num)

    residual = np.zeros((num, coeffs.shape[1]), dtype=coeffs.dtype)

    if num_modes > 0:
        residual[1:-1] = idst(
            coeffs * (num_modes + 1),
            type=1,
            axis=0,
            norm="backward",
        )

    return linear_interpolant(s, start, end) + residual


# -----------------------------------------------------------------------------
# Derivatives / curvature
# -----------------------------------------------------------------------------


def derivatives_dst(coeffs, start, end, s, *, weights=None):
    """
    First and second derivatives with respect to normalized parameter s.

    Returns
    -------
    d1 : array, shape (len(s), dim)
        First derivative.

    d2 : array, shape (len(s), dim)
        Second derivative.
    """
    coeffs = apply_spectral_filter(coeffs, weights)
    s = np.asarray(s, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    mode = np.arange(1, len(coeffs) + 1)
    k = np.pi * mode
    ks = np.outer(s, k)

    chord = end - start

    if len(coeffs) == 0:
        d1 = np.broadcast_to(chord, (len(s), len(chord))).copy()
        d2 = np.zeros_like(d1)
        return d1, d2

    d1 = chord[None, :] + np.cos(ks) @ (coeffs * k[:, None])
    d2 = -np.sin(ks) @ (coeffs * (k**2)[:, None])

    return d1, d2


def derivative_dst(coeffs, start, end, s, *, order=1, weights=None):
    """Return derivative of order 1 or 2."""
    if order not in (1, 2):
        raise ValueError("Only order=1 and order=2 are supported.")

    d1, d2 = derivatives_dst(coeffs, start, end, s, weights=weights)
    return d1 if order == 1 else d2


def curvature_dst(coeffs, start, end, s, *, weights=None, eps=1e-12):
    """Signed curvature for a 2D DST curve."""
    coeffs = np.asarray(coeffs, dtype=float)

    if coeffs.shape[1] != 2:
        raise ValueError(f"curvature_dst requires 2D coefficients, got dim={coeffs.shape[1]}.")

    d1, d2 = derivatives_dst(coeffs, start, end, s, weights=weights)
    dx, dy = d1[:, 0], d1[:, 1]
    ddx, ddy = d2[:, 0], d2[:, 1]

    speed2 = dx**2 + dy**2
    denom = np.maximum(speed2 ** 1.5, eps)

    return (dx * ddy - dy * ddx) / denom


# -----------------------------------------------------------------------------
# Spectral filters / spectrum
# -----------------------------------------------------------------------------


def cutoff_from_energy(coeffs, *, fraction=0.99, derivative_order=0):
    """
    Smallest cutoff preserving a given fraction of weighted spectral energy.

    derivative_order=0 gives position energy |c_k|^2.
    derivative_order=1 gives tangent energy (k pi)^2 |c_k|^2.
    derivative_order=2 gives bending-like energy (k pi)^4 |c_k|^2.
    """
    coeffs = np.asarray(coeffs, dtype=float)

    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")

    if len(coeffs) == 0:
        return 0

    mode = np.arange(1, len(coeffs) + 1)
    k = np.pi * mode

    energy = np.sum(coeffs**2, axis=1) * k ** (2 * int(derivative_order))
    total = energy.sum()

    if total <= 0:
        return len(coeffs)

    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, fraction) + 1)


def cutoff_from_min_scale(length, min_scale, num_modes, *, half_wavelength=True):
    """
    Convert a physical minimum scale to a mode cutoff.

    For sin(k pi s), the half-wavelength scale is length / k.
    The full wavelength scale is 2 * length / k.
    """
    if min_scale is None:
        return int(num_modes)

    length = float(length)
    min_scale = float(min_scale)

    if length <= 0:
        raise ValueError(f"length must be positive, got {length}.")

    if min_scale <= 0:
        raise ValueError(f"min_scale must be positive, got {min_scale}.")

    factor = 1.0 if half_wavelength else 2.0
    cutoff = int(np.floor(factor * length / min_scale))
    return max(0, min(cutoff, int(num_modes)))


def spectral_filter(
    coeffs,
    kind="none",
    *,
    cutoff=None,
    fraction=0.99,
    order=4,
    min_scale=None,
    length=None,
    half_wavelength=True,
):
    """
    Build per-mode spectral weights.

    Parameters
    ----------
    coeffs : array, shape (num_modes, dim)
        DST coefficients.

    kind : {'none', 'hard', 'energy', 'tangent_energy', 'bending_energy',
            'min_scale', 'tikhonov', 'exp'}
        Filter type.

    cutoff : int, optional
        Mode cutoff used by hard/tikhonov/exp filters.

    fraction : float, optional
        Energy fraction used by energy filters.

    order : float, optional
        Smooth-filter order for tikhonov/exp filters.

    min_scale : float, optional
        Physical minimum scale for kind='min_scale'.

    length : float, optional
        Physical curve length, required for kind='min_scale'.

    Returns
    -------
    weights : array, shape (num_modes,)
        Spectral weights.

    cutoff : int
        Effective cutoff index.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    num_modes = len(coeffs)
    kind = "none" if kind is None else str(kind)

    if kind in ("none", "all"):
        return np.ones(num_modes, dtype=float), num_modes

    if kind == "energy":
        cutoff = cutoff_from_energy(coeffs, fraction=fraction, derivative_order=0)
        kind = "hard"

    elif kind == "tangent_energy":
        cutoff = cutoff_from_energy(coeffs, fraction=fraction, derivative_order=1)
        kind = "hard"

    elif kind == "bending_energy":
        cutoff = cutoff_from_energy(coeffs, fraction=fraction, derivative_order=2)
        kind = "hard"

    elif kind == "min_scale":
        if length is None:
            raise ValueError("length is required when kind='min_scale'.")
        cutoff = cutoff_from_min_scale(
            length,
            min_scale,
            num_modes,
            half_wavelength=half_wavelength,
        )
        kind = "hard"

    if cutoff is None:
        cutoff = num_modes

    cutoff = int(cutoff)
    cutoff = max(0, min(cutoff, num_modes))

    mode = np.arange(1, num_modes + 1)
    kc = max(float(cutoff), 1.0)

    if kind == "hard":
        weights = np.zeros(num_modes, dtype=float)
        weights[:cutoff] = 1.0

    elif kind == "tikhonov":
        weights = 1.0 / (1.0 + (mode / kc) ** order)

    elif kind == "exp":
        weights = np.exp(-(mode / kc) ** order)

    else:
        raise ValueError(
            "kind must be one of 'none', 'hard', 'energy', 'tangent_energy', "
            "'bending_energy', 'min_scale', 'tikhonov', or 'exp'."
        )

    return weights, cutoff


def spectrum_dst(coeffs, *, weights=None):
    """Return position, tangent, and bending spectral energies."""
    coeffs = np.asarray(coeffs, dtype=float)
    coeffs_active = apply_spectral_filter(coeffs, weights)

    mode = np.arange(1, len(coeffs_active) + 1)
    k = np.pi * mode

    energy_per_dim = coeffs_active**2
    energy = energy_per_dim.sum(axis=1)
    tangent_energy = k**2 * energy
    bending_energy = k**4 * energy

    def normalized(x):
        total = x.sum()
        if total <= 0:
            return np.zeros_like(x), np.zeros_like(x)
        frac = x / total
        return frac, np.cumsum(frac)

    energy_fraction, cumulative_energy = normalized(energy)
    tangent_fraction, cumulative_tangent_energy = normalized(tangent_energy)
    bending_fraction, cumulative_bending_energy = normalized(bending_energy)

    out = {
        "mode": mode,
        "k": k,
        "coeffs": coeffs_active,
        "weights": None if weights is None else np.asarray(weights, dtype=float).copy(),
        "energy_per_dim": energy_per_dim,
        "energy": energy,
        "energy_fraction": energy_fraction,
        "cumulative_energy": cumulative_energy,
        "tangent_energy": tangent_energy,
        "tangent_fraction": tangent_fraction,
        "cumulative_tangent_energy": cumulative_tangent_energy,
        "bending_energy": bending_energy,
        "bending_fraction": bending_fraction,
        "cumulative_bending_energy": cumulative_bending_energy,
    }

    if coeffs_active.shape[1] >= 1:
        out["energy_x"] = energy_per_dim[:, 0]
    if coeffs_active.shape[1] >= 2:
        out["energy_y"] = energy_per_dim[:, 1]

    return out


# -----------------------------------------------------------------------------
# Rotation
# -----------------------------------------------------------------------------


def _get_rotation_matrix(theta):
    """2D counterclockwise rotation matrix."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _rotate_points(points, theta, *, center=None):
    """Rotate 2D row-vector points by theta around center."""
    points = np.asarray(points, dtype=float)

    if points.shape[-1] != 2:
        raise ValueError(f"_rotate_points requires 2D points, got shape {points.shape}.")
    
    R = _get_rotation_matrix(theta)

    if center is None:
        center = np.zeros(2, dtype=float)
    else:
        center = np.asarray(center, dtype=float)

    return (points - center) @ R.T + center


def rotate_coeffs(coeffs, theta):
    """Rotate 2D DST coefficients in coefficient space."""
    coeffs = np.asarray(coeffs, dtype=float)

    if coeffs.shape[1] != 2:
        raise ValueError(f"rotate_coeffs requires 2D coefficients, got dim={coeffs.shape[1]}.")

    return coeffs @ _get_rotation_matrix(theta).T


def optimal_rotation_dst(
    coeffs,
    start,
    end,
    s,
    *,
    weights=None,
    method="sampled",
    init="grid",
    num_grid=720,
    refine_width=np.deg2rad(5.0),
    eps=1e-12,
):
    """
    Find a rotation making a 2D curve as safely representable as y=f(x) as possible.

    The objective maximizes a margin based on x'(s).

    method='sampled'
        Directly maximizes min_s x'(s) / |chord_x|.

    method='bound'
        Uses the sufficient lower bound
            chord_x - sum_k |a_k| k pi.

    init='grid'
        Coarse grid search followed by local bounded refinement.

    init='chord'
        Refine near the angle that aligns the endpoint chord with +x.

    init='full'
        Single bounded optimization over [-pi, pi].
    """
    coeffs = apply_spectral_filter(coeffs, weights)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    s = np.asarray(s, dtype=float)

    if coeffs.shape[1] != 2:
        raise ValueError(f"optimal_rotation_dst requires 2D coefficients, got dim={coeffs.shape[1]}.")

    chord = end - start
    mode = np.arange(1, len(coeffs) + 1)
    k = np.pi * mode
    ks = np.outer(s, k)
    cos_ks = np.cos(ks)

    def margin(theta):
        R = _get_rotation_matrix(theta)
        coeffs_rot = coeffs @ R.T
        chord_rot = R @ chord

        a = coeffs_rot[:, 0]
        chord_x = chord_rot[0]

        if chord_x <= eps:
            return -np.inf

        if method == "bound":
            lower = chord_x - np.sum(np.abs(a) * k)
            return lower / abs(chord_x)

        if method == "sampled":
            dx = chord_x + cos_ks @ (a * k)
            return dx.min() / abs(chord_x)

        raise ValueError("method must be 'sampled' or 'bound'.")

    if init == "grid":
        theta_grid = np.linspace(-np.pi, np.pi, int(num_grid), endpoint=False)
        margins = np.array([margin(theta) for theta in theta_grid])
        theta0 = theta_grid[np.nanargmax(margins)]
        bounds = (theta0 - refine_width, theta0 + refine_width)

    elif init == "chord":
        theta0 = -np.arctan2(chord[1], chord[0])
        bounds = (theta0 - refine_width, theta0 + refine_width)

    elif init == "full":
        bounds = (-np.pi, np.pi)

    else:
        raise ValueError("init must be 'grid', 'chord', or 'full'.")

    res = minimize_scalar(lambda theta: -margin(theta), bounds=bounds, method="bounded")
    theta = float(res.x)

    # Map to [-pi, pi)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    return theta, margin(theta)


# -----------------------------------------------------------------------------
# Thin object wrapper
# -----------------------------------------------------------------------------


class DstCurve:
    """Thin wrapper around the standalone DST curve functions."""

    def __init__(self, points, num=None, *, tol=0.0, bc_type="not-a-knot"):
        fit = fit_dst_curve(points, num=num, tol=tol, bc_type=bc_type)

        self.points = fit["points"]
        self.s = fit["s"]
        self.coeffs = fit["coeffs"]
        self.start = fit["start"]
        self.end = fit["end"]
        self.length = fit["length"]

        self.weights, self.cutoff = spectral_filter(self.coeffs, kind="none")

    @classmethod
    def from_dst(cls, coeffs, start, end, s=None, *, weights=None, length=None):
        """Build a DstCurve directly from DST data."""
        coeffs = np.asarray(coeffs, dtype=float)
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        if s is None:
            s = np.linspace(0.0, 1.0, len(coeffs) + 2)
        else:
            s = np.asarray(s, dtype=float)

        obj = cls.__new__(cls)
        obj.s = s
        obj.coeffs = coeffs
        obj.start = start
        obj.end = end
        obj.points = reconstruct_dst(coeffs, start, end, s)
        obj.length = curve_length(obj.points) if length is None else float(length)

        if weights is None:
            obj.weights, obj.cutoff = spectral_filter(coeffs, kind="none")
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (len(coeffs),):
                raise ValueError(f"weights must have shape {(len(coeffs),)}, got {weights.shape}.")
            obj.weights = weights
            obj.cutoff = int(np.count_nonzero(weights))

        return obj

    @property
    def filtered_coeffs(self):
        return apply_spectral_filter(self.coeffs, self.weights)

    def filter(self, kind="none", **kwargs):
        """Set the active spectral filter and return self."""
        if "length" not in kwargs:
            kwargs["length"] = self.length
        self.weights, self.cutoff = spectral_filter(self.coeffs, kind=kind, **kwargs)
        return self

    def reconstruct(self, s=None, *, filtered=True):
        """Reconstruct the curve."""
        if s is None:
            s = self.s
        weights = self.weights if filtered else None
        return reconstruct_dst(self.coeffs, self.start, self.end, s, weights=weights)

    def residual(self, s=None, *, filtered=True):
        """Reconstruct only the sine residual."""
        if s is None:
            s = self.s
        weights = self.weights if filtered else None
        return reconstruct_residual_dst(self.coeffs, s, weights=weights)

    def derivative(self, order=1, s=None, *, filtered=True):
        """Return derivative of order 1 or 2."""
        if s is None:
            s = self.s
        weights = self.weights if filtered else None
        return derivative_dst(
            self.coeffs,
            self.start,
            self.end,
            s,
            order=order,
            weights=weights,
        )

    def derivatives(self, s=None, *, filtered=True):
        """Return first and second derivatives."""
        if s is None:
            s = self.s
        weights = self.weights if filtered else None
        return derivatives_dst(self.coeffs, self.start, self.end, s, weights=weights)

    def curvature(self, s=None, *, filtered=True, eps=1e-12):
        """Return signed curvature. Only valid for 2D curves."""
        if s is None:
            s = self.s
        weights = self.weights if filtered else None
        return curvature_dst(
            self.coeffs,
            self.start,
            self.end,
            s,
            weights=weights,
            eps=eps,
        )

    def spectrum(self, *, filtered=True):
        """Return spectral energy information."""
        weights = self.weights if filtered else None
        return spectrum_dst(self.coeffs, weights=weights)

    def optimal_rotation(self, *, filtered=True, **kwargs):
        """Return theta, margin for function-like rotation."""
        weights = self.weights if filtered else None
        return optimal_rotation_dst(
            self.coeffs,
            self.start,
            self.end,
            self.s,
            weights=weights,
            **kwargs,
        )

    def rotate(self, theta, *, filtered=True, refit=True, center=None):
        """
        Rotate the curve.

        refit=True rotates reconstructed points and refits/reparametrizes.
        refit=False rotates start/end/coefficients directly on the same s-grid.
        """
        if refit:
            points = self.reconstruct(filtered=filtered)
            points = _rotate_points(points, theta, center=center)
            return DstCurve(points, num=len(points))

        if center is not None:
            raise ValueError("center is only supported when refit=True.")

        weights = self.weights if filtered else None
        coeffs = apply_spectral_filter(self.coeffs, weights)
        coeffs = rotate_coeffs(coeffs, theta)
        start = _rotate_points(self.start[None, :], theta)[0]
        end = _rotate_points(self.end[None, :], theta)[0]

        return DstCurve.from_dst(coeffs, start, end, self.s)

    def shift(self, shift, *, filtered=True, refit=True):
        """Translate the curve."""
        shift = np.asarray(shift, dtype=float)

        if refit:
            return DstCurve(self.reconstruct(filtered=filtered) + shift, num=len(self.s))

        coeffs = self.filtered_coeffs if filtered else self.coeffs
        return DstCurve.from_dst(coeffs, self.start + shift, self.end + shift, self.s)
