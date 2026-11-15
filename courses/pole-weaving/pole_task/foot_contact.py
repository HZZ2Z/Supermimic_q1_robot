"""Shared foot-slip measurement definitions and evaluation limits."""

import math

CONTACT_FORCE_N = 10.0
SLIP_SPEED_M_S = 0.05
FOOT_CONTACT_VERSION = 1


def foot_slip_limits(config: dict | None = None) -> tuple[float, float]:
    config = config or {}
    mean_limit = float(config.get("foot_slip_mean_max_m_s", 0.1))
    fraction_limit = float(config.get("foot_slip_fraction_max", 0.2))
    if not math.isfinite(mean_limit) or mean_limit <= 0:
        raise ValueError("foot_slip_mean_max_m_s must be finite and positive")
    if not math.isfinite(fraction_limit) or not 0 < fraction_limit <= 1:
        raise ValueError("foot_slip_fraction_max must be in (0, 1]")
    return mean_limit, fraction_limit


def foot_slip_passes(row: dict, config: dict | None = None) -> bool:
    mean_limit, fraction_limit = foot_slip_limits(config)
    mean, fraction = row.get("foot_slip_mean_m_s"), row.get("foot_slip_fraction")
    contact_time = row.get("foot_contact_time_s", [])
    return bool(
        isinstance(contact_time, (list, tuple)) and len(contact_time) == 2
        and all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in contact_time)
        and isinstance(mean, (int, float)) and math.isfinite(mean) and 0 <= mean <= mean_limit
        and isinstance(fraction, (int, float)) and math.isfinite(fraction) and 0 <= fraction <= fraction_limit
    )
