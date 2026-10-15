"""Index time series: masks, area weights, regional means, annualisation.

Monthly indices per member (all area-weighted):
  gmsat         global-mean near-surface air temperature            K
  nino34        Nino3.4-box mean SST                                K (or degC for tos)
  trop_sst      tropical-band (20S-20N) mean ocean SST              same as nino34
  nino34_rel    nino34 - trop_sst                                   K
  sia_nh        Northern Hemisphere sea-ice area                    1e6 km2
  ism           monsoon-box land-only precipitation                 mm/day

Step 2 saves only the monthly series. Annual values are derived downstream
with these helpers, so annualisation and lag choices stay open:
  annual_mean(s, start_month)      days-weighted 12-month means (1 = calendar year,
                                   10 = Oct-Sep), labelled by the year they end in
  seasonal_mean(s, months)         e.g. September sea ice, JJA monsoon
  stitch_from_baseline(sai, base, n_months)
                                   prepend the paired baseline member's months before
                                   the SAI branch point (needed for lagged ENSO terms
                                   or windows starting before January)
  annual_indices(monthly, choices, start_month)
                                   all six annual indices at once

Every function takes the choices block from inputs.yaml, so the definitions
live in the config, not here.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from inputs import get_latlon, grids_match, spatial_shape

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


def _lat_lon_1d(obj):
    """1D latitude and longitude (lon in 0-360) if obj has them, else (None, None)."""
    ds = obj.to_dataset(name="_") if isinstance(obj, xr.DataArray) else obj
    lat, lon = get_latlon(ds)
    if lat is None or lon is None or lat.ndim != 1 or lon.ndim != 1:
        return None, None
    return lat, lon % 360


def align_fx(fx: xr.DataArray, da: xr.DataArray, atol: float = 0.01) -> xr.DataArray:
    """fx field on da's spatial dims, matched by coordinate VALUES, not array position.

    Regular grids (1D lat/lon): each data row/column takes the fx cell with the same
    lat/lon, so fields stored in a different order (e.g. latitude flipped) line up.
    Raises if any data coordinate has no fx match within atol degrees.
    Curvilinear grids (2D lat/lon): shapes must match and, where both carry
    coordinates, the coordinates must agree; otherwise raises.
    Returned without coordinates, dims = da's spatial dims."""
    sd = spatial_dims(da)
    lat_d, lon_d = _lat_lon_1d(da)
    lat_f, lon_f = _lat_lon_1d(fx)
    if lat_d is not None and lat_f is not None:
        ilat = np.abs(lat_f.values[None, :] - lat_d.values[:, None]).argmin(1)
        dlon = np.abs(lon_f.values[None, :] - lon_d.values[:, None])
        dlon = np.minimum(dlon, 360 - dlon)
        ilon = dlon.argmin(1)
        err = max(np.abs(lat_f.values[ilat] - lat_d.values).max(), dlon[np.arange(len(ilon)), ilon].max())
        if err > atol:
            raise ValueError(f"fx field '{fx.name}' does not match the data grid "
                             f"(max coordinate mismatch {err:.3f} deg)")
        out = fx.isel({lat_f.dims[0]: ilat, lon_f.dims[0]: ilon})
        out = out.rename({lat_f.dims[0]: lat_d.dims[0], lon_f.dims[0]: lon_d.dims[0]})
        return xr.DataArray(out.transpose(*sd).values, dims=sd)
    if tuple(fx.shape) != spatial_shape(da, "time"):
        raise ValueError(f"fx field '{fx.name}' shape {fx.shape} does not match data "
                         f"{spatial_shape(da, 'time')}")
    lat_a, lon_a = get_latlon(da.to_dataset(name="_"))
    lat_b, lon_b = get_latlon(fx.to_dataset(name="_"))
    if grids_match(lat_a, lon_a, lat_b, lon_b, atol=atol) is False:
        raise ValueError(f"fx field '{fx.name}' coordinates differ from the data grid")
    return xr.DataArray(fx.values, dims=sd)


def box(lat, lon, lat_range, lon_range=None) -> xr.DataArray:
    m = (lat >= lat_range[0]) & (lat <= lat_range[1])
    if lon_range is not None:
        lo, hi = lon_range[0] % 360, lon_range[1] % 360
        m = m & ((lon >= lo) & (lon <= hi) if lo <= hi else (lon >= lo) | (lon <= hi))
    return m


def area_weights(da: xr.DataArray, area: xr.DataArray | None):
    """Cell areas on da's spatial dims: the fx area if shapes match, else cos(lat).
    Returns (weights, method)."""
    if area is not None and tuple(area.shape) == spatial_shape(da, "time"):
        return align_fx(area, da), "fx area"
    lat, _ = latlon(da)
    return np.cos(np.deg2rad(lat)), "cos(lat)"


def as_fraction(field: xr.DataArray) -> tuple[xr.DataArray, float]:
    """Divide by 100 if the field is in percent. Decided from the values
    (max > 1.5 means percent), not the units attribute, which is
    known to be wrong for some files. Returns (field, divisor)."""
    divisor = 100.0 if float(field.max()) > 1.5 else 1.0
    return field / divisor, divisor


def ocean_mask(sftlf: xr.DataArray, da: xr.DataArray, how: str) -> xr.DataArray:
    frac, _ = as_fraction(align_fx(sftlf, da))
    return (frac == 0) if how == "strict" else (frac < 0.5)


def land_mask(sftlf: xr.DataArray, da: xr.DataArray, how: str) -> xr.DataArray:
    frac, _ = as_fraction(align_fx(sftlf, da))
    return (frac >= 1.0) if how == "strict" else (frac > 0.5)


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
        ocean = ocean_mask(sftlf, da, choices["ocean_mask"])
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
    w = align_fx(area, da)
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
    mask = box(lat, lon, c["lat"], c["lon"]) & land_mask(sftlf, da, c["land_mask"])
    return ({"ism": area_mean(da, w, mask) * 86400.0},
            {"ism_weights": how, "ism_cells": int(mask.sum())})


# ---------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------

def _days(s: xr.DataArray) -> xr.DataArray:
    return s.time.dt.days_in_month.astype(float)


def annual_mean(s: xr.DataArray, start_month: int = 1, complete_only: bool = True) -> xr.DataArray:
    """12-month means starting in start_month (1 = calendar year, 10 = Oct-Sep),
    weighted by days in month and labelled by the year of the last month.
    Years with fewer than 12 months dropped if complete_only."""
    label = s.time.dt.year + (s.time.dt.month >= start_month) * (start_month > 1)
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


def stitch_from_baseline(sai: xr.DataArray, base: xr.DataArray | None, n_months: int):
    """Prepend the paired baseline member's last n_months before the SAI series
    starts (before branching, the SAI run IS that baseline member). Use for lagged
    regressors or annual windows starting before January.
    Returns (series, n_months_actually_prepended)."""
    sai = sai.reset_coords(drop=True)           # drop stray coords (e.g. UM forecast_period)
    if base is None or sai.size == 0 or n_months <= 0:
        return sai, 0
    base = base.reset_coords(drop=True)
    start = int(sai.time.dt.year[0]) * 12 + int(sai.time.dt.month[0]) - 1
    ym = (base.time.dt.year * 12 + base.time.dt.month - 1).values
    pre = base.isel(time=np.where((ym >= start - n_months) & (ym < start))[0])
    if pre.sizes["time"] == 0:
        return sai, 0
    return xr.concat([pre, sai], dim="time"), int(pre.sizes["time"])


def annual_indices(monthly: dict[str, xr.DataArray], choices,
                   start_month: int = 1) -> dict[str, xr.DataArray]:
    """Annual means of the temperature/ENSO indices (start_month as in annual_mean),
    plus September sea ice and the JJA monsoon index."""
    complete = choices.get("complete_years_only", True)
    monthly = {k: v.reset_coords(drop=True) for k, v in monthly.items()}
    out = {}
    for k in ("gmsat", "nino34", "trop_sst", "nino34_rel"):
        if k in monthly:
            out[k] = annual_mean(monthly[k], start_month, complete)
    if "sia_nh" in monthly:
        out["sia_sep"] = seasonal_mean(monthly["sia_nh"], [choices["seaice"]["month"]], complete)
    if "ism" in monthly:
        out["ism_jja"] = seasonal_mean(monthly["ism"], choices["ism"]["months"], complete)
    return out
