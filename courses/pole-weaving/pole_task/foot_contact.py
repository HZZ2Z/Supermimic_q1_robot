"""Shared foot-slip measurement definitions and evaluation limits."""

import math

import numpy as np

CONTACT_FORCE_N = 10.0
SLIP_SPEED_M_S = 0.05
FOOT_CONTACT_VERSION = 1
FOOT_SLIP_PHASE_VERSION = 1


def foot_slip_phase_metrics(
    samples, dt: float, *, transition_window_s: float = 0.04,
    max_contact_gap_s: float = 0.02,
) -> dict:
    """Partition loaded foot-time; each sample is [loaded, v, v², fast].

    v and v² are independently force-weighted contact-point moments, already
    masked by load. Short gaps join support bouts but never contribute time.
    Recording boundaries are censored, not assumed touchdown/liftoff events.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(transition_window_s) or transition_window_s <= 0:
        raise ValueError("transition_window_s must be finite and positive")
    if not math.isfinite(max_contact_gap_s) or max_contact_gap_s < 0:
        raise ValueError("max_contact_gap_s must be finite and nonnegative")
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        samples = np.empty((0, 2, 4))
    if samples.ndim != 3 or samples.shape[1:] != (2, 4) or not np.isfinite(samples).all():
        raise ValueError("Expected finite contact samples with shape (steps, 2, 4)")
    loaded = samples[..., 0] > 0.5
    # -1 is airborne; 0/1/2/3 are sustained/touchdown/liftoff/boundary.
    phase = np.full(loaded.shape, -1, dtype=np.int8)
    window = max(1, math.ceil(transition_window_s / dt - 1e-9))

    def runs(mask):
        bounds = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
        return zip(bounds[::2], bounds[1::2])

    for foot in range(2):
        closed = loaded[:, foot].copy()
        for start, end in runs(~closed):
            if (start > 0 and end < len(closed)
                    and (end - start) * dt <= max_contact_gap_s + 1e-12):
                closed[start:end] = True
        for start, end in runs(closed):
            index = np.arange(start, end)[loaded[start:end, foot]]
            age, remaining = index - start, end - 1 - index
            near_start, near_end = age < window, remaining < window
            # Short bouts split at the nearest boundary; ties go to touchdown.
            touchdown = near_start & (~near_end | (age <= remaining))
            liftoff = near_end & ~touchdown
            phase[index, foot] = 0
            phase[index[touchdown], foot] = 3 if start == 0 else 1
            phase[index[liftoff], foot] = 3 if end == len(closed) else 2

    duration_all = float(samples[..., 0].sum() * dt)
    result = {
        "version": FOOT_SLIP_PHASE_VERSION,
        "transition_window_s": transition_window_s,
        "max_contact_gap_s": max_contact_gap_s,
        "contact_force_threshold_n": CONTACT_FORCE_N,
        "slip_speed_threshold_m_s": SLIP_SPEED_M_S,
        "velocity_timing": "pre_integration",
    }
    def summarize(mask):
        totals = (samples * mask[..., None]).sum(axis=0) * dt
        duration, distance, squared, fast = totals.sum(axis=0)
        return {
            "contact_time_s": totals[:, 0].tolist(),
            "loaded_time_fraction": float(duration / duration_all) if duration_all > 0 else None,
            "slip_mean_m_s": float(distance / duration) if duration > 0 else None,
            "slip_rms_m_s": float(np.sqrt(squared / duration)) if duration > 0 else None,
            "slip_fraction": float(fast / duration) if duration > 0 else None,
        }
    for number, name in enumerate(("sustained", "touchdown", "liftoff", "boundary")):
        result[name] = summarize(phase == number)

    sustained = phase == 0
    reloading = np.zeros_like(loaded)
    unknown_onset = np.zeros_like(loaded)
    events = [0, 0]
    reload_window_s = 0.01
    reload_steps = math.floor(reload_window_s / dt + 1e-9)
    for foot in range(2):
        for start, end in runs(loaded[:, foot]):
            edge = slice(start, min(end, start + reload_steps))
            if start == 0:
                # Initial load onset is unobserved, never invent a reloading event.
                unknown_onset[edge, foot] = sustained[edge, foot]
            else:
                reloading[edge, foot] = sustained[edge, foot]
                events[foot] += int(reloading[edge, foot].any())
    result["sustained_continuity"] = {
        "version": 1,
        "reload_window_s": reload_window_s,
        "reloading_events_by_foot": events,
        "continuous": summarize(sustained & ~reloading & ~unknown_onset),
        "reloading": summarize(reloading),
        "unknown_onset": summarize(unknown_onset),
    }
    return result


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
