"""Index time series: masks, area weights, regional means, annualisation.

Monthly indices per member (all area-weighted):
  gmsat         global-mean near-surface air temperature            K
  nino34        Nino3.4-box mean SST                                K (or degC for tos)
  trop_sst      tropical-band (20S-20N) mean ocean SST              same as nino34
  nino34_rel    nino34 - trop_sst                                   K
  sia_nh        Northern Hemisphere sea-ice area                    1e6 km2
  ism           monsoon-box land-only precipitation                 mm/day

Annual indices (see annualise / seasonal_mean). For SAI scenarios the
first Oct-Sep year is completed from the paired baseline member
(stitch_from_baseline) before annualising:
  gmsat, nino34, trop_sst, nino34_rel   Oct-Sep means (labelled by the Sep year)
  sia_sep                               September value
  ism_jja                               JJA mean

Every function takes the choices block from inputs.yaml, so the definitions
live in the config, not here.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from inputs import get_latlon, spatial_shape

MONTHLY_INDICES = ["gmsat", "nino34", "trop_sst", "nino34_rel", "sia_nh", "ism"]
ANNUAL_INDICES = ["gmsat", "nino34", "trop_sst", "nino34_rel", "sia_sep", "ism_jja"]
UNITS = {"gmsat": "K", "nino34": "K", "trop_sst": "K", "nino34_rel": "K",
         "sia_nh": "1e6 km2", "sia_sep": "1e6 km2", "ism": "mm/day", "ism_jja": "mm/day"}


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def spatial_dims(da: xr.DataArray) -> list[str]:
    return [d for d in da.dims if d != "time"]


def latlon(da: xr.DataArray, fx: xr.DataArray | None = None):
    """2D-broadcast lat/lon on da's spatial dims, from da's coords or else the fx field."""
    for src in (da, fx):
        if src is None:
            continue
        lat, lon = get_latlon(src.to_dataset(name="_") if isinstance(src, xr.DataArray) else src)
        if lat is not None and lon is not None:
            lat, lon = xr.broadcast(lat, lon)
            sd = spatial_dims(da)
            if lat.shape == tuple(da.sizes[d] for d in sd):
                return (xr.DataArray(lat.values, dims=sd), xr.DataArray(lon.values % 360, dims=sd))
    raise ValueError("no latitude/longitude found on data or fx field")


def box(lat, lon, lat_range, lon_range=None) -> xr.DataArray:
    m = (lat >= lat_range[0]) & (lat <= lat_range[1])
    if lon_range is not None:
        lo, hi = lon_range[0] % 360, lon_range[1] % 360
        m = m & ((lon >= lo) & (lon <= hi) if lo <= hi else (lon >= lo) | (lon <= hi))
    return m


def area_weights(da: xr.DataArray, area: xr.DataArray | None):
    """Cell areas on da's spatial dims: the fx area if shapes match, else cos(lat).
    Returns (weights, method)."""
    sd = spatial_dims(da)
    if area is not None and tuple(area.shape) == spatial_shape(da, "time"):
        return xr.DataArray(area.values, dims=sd), "fx area"
    lat, _ = latlon(da)
    return np.cos(np.deg2rad(lat)), "cos(lat)"


def as_fraction(field: xr.DataArray) -> tuple[xr.DataArray, float]:
    """Divide by 100 if the field is in percent. Decided from the values
    (max > 1.5 means percent), not the units attribute, which is
    known to be wrong for some files. Returns (field, divisor)."""
    divisor = 100.0 if float(field.max()) > 1.5 else 1.0
    return field / divisor, divisor


def ocean_mask(sftlf: xr.DataArray, sd, how: str) -> xr.DataArray:
    frac, _ = as_fraction(sftlf)
    m = (frac == 0) if how == "strict" else (frac < 0.5)
    return xr.DataArray(m.values, dims=sd)


def land_mask(sftlf: xr.DataArray, sd, how: str) -> xr.DataArray:
    frac, _ = as_fraction(sftlf)
    m = (frac >= 1.0) if how == "strict" else (frac > 0.5)
    return xr.DataArray(m.values, dims=sd)


def area_mean(da, w, mask=None) -> xr.DataArray:
    sd = spatial_dims(da)
    if mask is not None:
        w = w.where(mask, 0)
    return da.weighted(w.fillna(0)).mean(sd)


# ---------------------------------------------------------------------
# Monthly indices, one per role
# ---------------------------------------------------------------------

def gmsat_monthly(da, area):
    w, how = area_weights(da, area)
    return {"gmsat": area_mean(da, w)}, {"gmsat_weights": how}


def sst_monthly(da, area, sftlf, grid, choices):
    """nino34, trop_sst, nino34_rel. On the atmosphere grid the land-sea mask
    selects ocean cells; on the ocean grid all valid cells are ocean."""
    w, how = area_weights(da, area)
    lat, lon = latlon(da, area)
    sd = spatial_dims(da)
    if grid == "atmos":
        if sftlf is None:
            raise ValueError("SST on the atmosphere grid needs sftlf for the ocean mask")
        ocean = ocean_mask(sftlf, sd, choices["ocean_mask"])
    else:
        ocean = xr.ones_like(lat, dtype=bool)
    nb = choices["nino34_box"]
    nino = area_mean(da, w, box(lat, lon, nb["lat"], nb["lon"]) & ocean)
    trop = area_mean(da, w, box(lat, lon, choices["relative_nino34_band"]["lat"]) & ocean)
    meta = {"sst_weights": how,
            "nino34_land_cells": int((box(lat, lon, nb["lat"], nb["lon"]) & ~ocean).sum())}
    return {"nino34": nino, "trop_sst": trop, "nino34_rel": nino - trop}, meta


def seaice_monthly(da, area, choices):
    """Hemispheric sea-ice area = sum(concentration x cell area), in 1e6 km2.
    Needs true cell areas (fx); cos(lat) weights have no absolute scale."""
    sd = spatial_dims(da)
    if area is None or tuple(area.shape) != spatial_shape(da, "time"):
        raise ValueError("sea-ice area needs a matching areacella/areacello")
    w = xr.DataArray(area.values, dims=sd)
    lat, _ = latlon(da, area)
    hemi = (lat > 0) if choices["seaice"]["hemisphere"] == "north" else (lat < 0)
    frac, divisor = as_fraction(da.isel(time=slice(0, 24)))
    sia = ((da / divisor).fillna(0) * w.where(hemi, 0)).sum(sd) / 1e12
    return {"sia_nh": sia}, {"seaice_divisor": divisor}


def ism_monthly(da, area, sftlf, choices):
    w, how = area_weights(da, area)
    lat, lon = latlon(da, area)
    c = choices["ism"]
    if sftlf is None:
        raise ValueError("land-only monsoon index needs sftlf")
    mask = box(lat, lon, c["lat"], c["lon"]) & land_mask(sftlf, spatial_dims(da), c["land_mask"])
    return ({"ism": area_mean(da, w, mask) * 86400.0},
            {"ism_weights": how, "ism_cells": int(mask.sum())})


# ---------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------

def _days(s: xr.DataArray) -> xr.DataArray:
    return s.time.dt.days_in_month.astype(float)


def annualise(s: xr.DataArray, complete_only: bool = True) -> xr.DataArray:
    """Oct-Sep mean, weighted by days in month, labelled by the year of the
    September it ends in. Years with fewer than 12 months dropped if complete_only."""
    label = s.time.dt.year + (s.time.dt.month >= 10)
    s = s.assign_coords(year=("time", label.values))
    d = _days(s).where(s.notnull(), 0)
    num = (s.fillna(0) * d).groupby("year").sum()
    den = d.groupby("year").sum()
    n = s.notnull().groupby("year").sum()
    out = num / den
    return out.where(n == 12, drop=True) if complete_only else out


def seasonal_mean(s: xr.DataArray, months, complete_only: bool = True) -> xr.DataArray:
    """Days-weighted mean over the given calendar months of each year
    (single month -> that month's value)."""
    s = s.sel(time=s.time.dt.month.isin(list(months)))
    s = s.assign_coords(year=("time", s.time.dt.year.values))
    d = _days(s).where(s.notnull(), 0)
    out = (s.fillna(0) * d).groupby("year").sum() / d.groupby("year").sum()
    n = s.notnull().groupby("year").sum()
    return out.where(n == len(months), drop=True) if complete_only else out


def stitch_from_baseline(sai: xr.DataArray, base: xr.DataArray | None):
    """Prepend the paired baseline member's months that the SAI run lacks
    within its first Oct-Sep year (before branching, the SAI run IS that
    baseline member). Returns (series, n_months_stitched)."""
    sai = sai.reset_coords(drop=True)           # drop stray coords (e.g. UM forecast_period)
    if base is None or sai.size == 0:
        return sai, 0
    base = base.reset_coords(drop=True)
    y0, m0 = int(sai.time.dt.year[0]), int(sai.time.dt.month[0])
    first_oct = (y0 - 1) * 12 + 9 if m0 < 10 else y0 * 12 + 9      # ym of Oct starting that year
    start = y0 * 12 + m0 - 1
    ym = (base.time.dt.year * 12 + base.time.dt.month - 1).values
    pre = base.isel(time=np.where((ym >= first_oct) & (ym < start))[0])
    if pre.sizes["time"] == 0:
        return sai, 0
    return xr.concat([pre, sai], dim="time"), int(pre.sizes["time"])


def annual_indices(monthly: dict[str, xr.DataArray], choices) -> dict[str, xr.DataArray]:
    complete = choices.get("complete_years_only", True)
    monthly = {k: v.reset_coords(drop=True) for k, v in monthly.items()}
    out = {}
    for k in ("gmsat", "nino34", "trop_sst", "nino34_rel"):
        if k in monthly:
            out[k] = annualise(monthly[k], complete)
    if "sia_nh" in monthly:
        out["sia_sep"] = seasonal_mean(monthly["sia_nh"], [choices["seaice"]["month"]], complete)
    if "ism" in monthly:
        out["ism_jja"] = seasonal_mean(monthly["ism"], choices["ism"]["months"], complete)
    return out
