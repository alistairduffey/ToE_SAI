"""Input configuration: loading the config, resolving files, opening members.

Lives alongside inputs.yaml and the notebooks (import with `from inputs import ...`).
Local paths may start with ~ (expanded to the user's home directory).

Every notebook gets its data through here, so the config file is the
single source of truth for what the pipeline reads and how it is
standardised on load (variable names, units, time stamps, duplicates).

Main entry points
-----------------
load_config(path)                -> dict
expected_files(cfg)              -> one row per (model, scenario, role, member)
check_existence(df)              -> adds status, files, n_files, size_bytes
load_member(row)                 -> standardised xr.Dataset for one member
open_fx(cfg, model, name)        -> fx field (areacella, areacello, sftlf)
"""
from __future__ import annotations

import glob as _glob
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Keys that may be set at model level and overridden per scenario
MERGE_KEYS = ("member_labels", "file_vars", "tables", "unit_scale", "time_fix",
              "anon", "format", "path", "path_overrides", "roles_used")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

def load_config(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_configured(cfg: dict, model: str) -> bool:
    mcfg = cfg["models"].get(model) or {}
    return mcfg.get("status") != "not_configured" and "scenarios" in mcfg


def model_roles(cfg: dict, model: str) -> dict[str, str]:
    """Role -> variable for a model, after role_overrides."""
    roles = dict(cfg["roles"])
    roles.update(cfg["models"][model].get("role_overrides") or {})
    return roles


def variable_grid(cfg: dict, model: str, var: str) -> str | None:
    grids = dict(cfg.get("variable_grids") or {})
    grids.update(cfg["models"][model].get("variable_grids") or {})
    return grids.get(var)


def scenario_required(cfg: dict, model: str, scenario: str) -> bool:
    """False if the scenario is restricted to other models (e.g. historical)."""
    only = (cfg["scenarios"][scenario] or {}).get("models")
    return only is None or model in only


def scenario_spec(cfg: dict, model: str, scenario: str) -> dict | None:
    """Model-level settings overlaid with scenario-level ones (dicts merged)."""
    mcfg = cfg["models"][model]
    scfg = (mcfg.get("scenarios") or {}).get(scenario)
    if scfg is None:
        return None
    spec = {k: mcfg[k] for k in MERGE_KEYS if k in mcfg}
    for k, v in scfg.items():
        if isinstance(v, dict) and isinstance(spec.get(k), dict):
            spec[k] = {**spec[k], **v}
        else:
            spec[k] = v
    return spec


# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------

def is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


@lru_cache(maxsize=2)
def s3(anon: bool = False):
    import s3fs
    return s3fs.S3FileSystem(anon=anon)  # non-anon: credentials from environment


def _has_magic(p: str) -> bool:
    return any(c in p for c in "*?[")


def resolve(pattern: str, anon: bool = False) -> list[tuple[str, float]]:
    """(path, size_bytes) for every file or store matching pattern, sorted.
    Size is NaN for directories (e.g. zarr stores)."""
    pattern = pattern.rstrip("/")
    if not is_s3(pattern):
        pattern = os.path.expanduser(pattern)
    if is_s3(pattern):
        fs = s3(anon)
        try:
            if _has_magic(pattern):
                hits = fs.glob(pattern, detail=True)
            else:
                hits = {pattern: fs.info(pattern)}
        except (FileNotFoundError, PermissionError):
            return []
        out = []
        for p, info in hits.items():
            size = info.get("size") if info.get("type") == "file" else None
            out.append(("s3://" + p.removeprefix("s3://"), float(size) if size else np.nan))
        return sorted(out)
    hits = sorted(_glob.glob(pattern)) if _has_magic(pattern) else (
        [pattern] if Path(pattern).exists() else [])
    return [(p, float(Path(p).stat().st_size) if Path(p).is_file() else np.nan) for p in hits]


def path_exists(path: str, anon: bool = False) -> bool:
    return bool(resolve(path, anon))


def _time_kwargs() -> dict:
    try:
        from xarray.coders import CFDatetimeCoder
        return {"decode_times": CFDatetimeCoder(use_cftime=True)}
    except ImportError:  # older xarray
        return {"use_cftime": True}


def open_nc(path: str, anon: bool = False, **kwargs) -> xr.Dataset:
    """Open one netCDF file lazily from S3 or local disk, with cftime dates."""
    kw = {**_time_kwargs(), **kwargs}
    if not is_s3(path):
        return xr.open_dataset(os.path.expanduser(path), **kw)
    try:
        return xr.open_dataset(s3(anon).open(path, "rb"), engine="h5netcdf", **kw)
    except (OSError, ValueError):
        # netCDF3 classic files are not HDF5
        return xr.open_dataset(s3(anon).open(path, "rb"), engine="scipy", **kw)


def open_zarr(path: str, anon: bool = False, chunks=None) -> xr.Dataset:
    store = s3(anon).get_mapper(path) if is_s3(path) else os.path.expanduser(path)
    kw = {} if chunks is None else {"chunks": chunks}
    return xr.open_zarr(store, consolidated=None, **kw, **_time_kwargs())


# ---------------------------------------------------------------------
# Expected files
# ---------------------------------------------------------------------

def build_path(cfg: dict, model: str, spec: dict, var: str, member: str) -> str:
    template = (spec.get("path_overrides") or {}).get(var, spec["path"])
    return template.format(
        root=cfg["storage"]["root"], model=model, var=var, member=member,
        member_label=(spec.get("member_labels") or {}).get(member, member),
        file_var=(spec.get("file_vars") or {}).get(var, var),
        table=(spec.get("tables") or {}).get(var, ""),
    )


def expected_files(cfg: dict, models=None, scenarios=None) -> pd.DataFrame:
    """One row per (model, scenario, role, member) the run expects.

    status is one of:
      expected        pattern built, existence not yet checked
      not_configured  model has no layout in the config yet
      not_available   model has no entry for this scenario
      not_required    scenario restricted to other models, or role not
                      in the scenario's roles_used
    """
    run = cfg.get("run") or {}
    models = models or run.get("models") or list(cfg["models"])
    scenarios = scenarios or run.get("scenarios") or list(cfg["scenarios"])
    rows = []

    for model in models:
        if not is_configured(cfg, model):
            rows.append(dict(model=model, status="not_configured"))
            continue
        mcfg = cfg["models"][model]
        for scen in scenarios:
            spec = scenario_spec(cfg, model, scen)
            used = (spec or {}).get("roles_used")
            for role, var in model_roles(cfg, model).items():
                for member in mcfg["members"]:
                    row = dict(model=model, scenario=scen, role=role, var=var,
                               member=member, pattern=None, status="expected")
                    if not scenario_required(cfg, model, scen) or (used and role not in used):
                        row["status"] = "not_required"
                    elif spec is None:
                        row["status"] = "not_available"
                    else:
                        row.update(
                            pattern=build_path(cfg, model, spec, var, member),
                            file_var=(spec.get("file_vars") or {}).get(var, var),
                            member_label=(spec.get("member_labels") or {}).get(member, member),
                            anon=bool(spec.get("anon", False)),
                            format=spec.get("format", "netcdf"),
                            time_fix=spec.get("time_fix", "none"),
                            unit_scale=float((spec.get("unit_scale") or {}).get(var, 1.0)),
                            grid_expected=variable_grid(cfg, model, var),
                        )
                    rows.append(row)

    cols = ["model", "scenario", "role", "var", "file_var", "member", "member_label",
            "pattern", "anon", "format", "time_fix", "unit_scale", "grid_expected", "status"]
    return pd.DataFrame(rows).reindex(columns=cols)


def check_existence(df: pd.DataFrame, workers: int = 16) -> pd.DataFrame:
    """Resolve each pattern; set status present/missing, add files, n_files, size_bytes.
    For zarr, only the last match (latest version) is used."""
    from concurrent.futures import ThreadPoolExecutor
    df = df.copy()
    todo = df.index[df["status"] == "expected"]
    with ThreadPoolExecutor(workers) as ex:
        hits = list(ex.map(lambda i: resolve(df.at[i, "pattern"], df.at[i, "anon"]), todo))
    df["files"] = None
    df["n_files"] = 0
    df["size_bytes"] = np.nan
    for i, h in zip(todo, hits):
        if df.at[i, "format"] == "zarr":
            h = h[-1:]
        df.at[i, "files"] = [p for p, _ in h]
        df.at[i, "n_files"] = len(h)
        df.at[i, "size_bytes"] = np.nansum([s for _, s in h]) if h else np.nan
        df.at[i, "status"] = "present" if h else "missing"
    return df


# ---------------------------------------------------------------------
# Dataset introspection
# ---------------------------------------------------------------------

_BOUNDS = re.compile(r"(_bnds|_bnd|_bounds|_vertices|^time_)", re.I)


def time_dim(ds: xr.Dataset) -> str | None:
    if "time" in ds.dims:
        return "time"
    for name, v in ds.coords.items():
        if v.attrs.get("axis") == "T" or v.attrs.get("standard_name") == "time":
            return v.dims[0]
    return None


def data_var(ds: xr.Dataset, candidates=()) -> str:
    """Name of the main data variable: a candidate if present, else the
    single non-bounds variable with a time dimension."""
    for c in candidates:
        if c and c in ds.data_vars:
            return c
    tdim = time_dim(ds)
    found = [v for v in ds.data_vars
             if not _BOUNDS.search(v) and (tdim is None or tdim in ds[v].dims)]
    if len(found) == 1:
        return found[0]
    raise KeyError(f"cannot identify data variable among {list(ds.data_vars)}")


def get_latlon(ds: xr.Dataset) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """Latitude and longitude, found by standard_name or common names."""
    out = {}
    for key, names in (("latitude", ("lat", "latitude", "nav_lat", "TLAT")),
                       ("longitude", ("lon", "longitude", "nav_lon", "TLONG"))):
        hit = None
        for name, v in ds.variables.items():
            if v.attrs.get("standard_name") == key and not _BOUNDS.search(name):
                hit = ds[name]
                break
        if hit is None:
            hit = next((ds[n] for n in names if n in ds.variables), None)
        out[key] = hit
    return out["latitude"], out["longitude"]


def spatial_shape(da: xr.DataArray, tdim: str | None) -> tuple[int, ...]:
    return tuple(n for d, n in zip(da.dims, da.shape) if d != tdim)


def grids_match(lat_a, lon_a, lat_b, lon_b, atol=1e-3) -> bool | None:
    """True/False if both grids have coordinates; None if either lacks them."""
    if any(x is None for x in (lat_a, lon_a, lat_b, lon_b)):
        return None
    try:
        la, lb = np.broadcast_arrays(*[np.asarray(x) for x in (lat_a, lat_b)])
        oa, ob = np.broadcast_arrays(*[np.asarray(x) % 360 for x in (lon_a, lon_b)])
    except ValueError:
        return False
    dlon = np.abs(oa - ob)
    dlon = np.minimum(dlon, 360 - dlon)
    return bool(np.nanmax(np.abs(la - lb)) < atol and np.nanmax(dlon) < atol)


# ---------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------

def time_bounds(ds: xr.Dataset, tdim: str) -> np.ndarray | None:
    """(n, 2) array of decoded time bounds, or None if absent."""
    name = ds[tdim].attrs.get("bounds") or ds[tdim].encoding.get("bounds")
    if not name or name not in ds:
        name = next((n for n in ("time_bnds", "time_bounds", "time_bnd") if n in ds), None)
    if name is None:
        return None
    b = ds[name].values
    if np.issubdtype(np.asarray(b).dtype, np.number):  # undecoded: use time encoding
        import cftime
        enc = ds[tdim].encoding
        b = cftime.num2date(b, enc.get("units"), enc.get("calendar", "standard"))
    return np.asarray(b)


def time_position(ds: xr.Dataset) -> float | None:
    """Where the first time stamp sits in its averaging interval:
    0 = start, 0.5 = centre, 1 = end. None if there are no bounds."""
    tdim = time_dim(ds)
    b = time_bounds(ds, tdim)
    if b is None:
        return None
    t0, lo, hi = ds[tdim].values[0], b[0, 0], b[0, 1]
    return float((t0 - lo) / (hi - lo))


def center_time(ds: xr.Dataset) -> xr.Dataset:
    tdim = time_dim(ds)
    b = time_bounds(ds, tdim)
    if b is None:
        raise ValueError("time_fix=center_of_bounds but the file has no time bounds")
    mid = np.array([lo + (hi - lo) / 2 for lo, hi in b])
    return ds.assign_coords({tdim: mid})


def month_index(time: xr.DataArray) -> np.ndarray:
    """year*12 + month-1, for aligning monthly series across calendars."""
    return (time.dt.year * 12 + time.dt.month - 1).values


# ---------------------------------------------------------------------
# Loading members
# ---------------------------------------------------------------------

def open_parts(row) -> list[xr.Dataset]:
    """Raw datasets for one member, one per file (or one zarr store), opened
    lazily without dask. Chunking is applied later to the selected variable only
    (see standardise), because files can contain variables dask cannot auto-chunk,
    such as text or decoded date variables (object dtype)."""
    if row["format"] == "zarr":
        return [open_zarr(row["files"][-1], row["anon"])]
    return [open_nc(p, row["anon"]) for p in row["files"]]


def chunk_like_storage(da: xr.DataArray, chunks="auto") -> xr.DataArray:
    """Chunk one variable with dask. With "auto", chunk sizes (~128 MB) are
    whole multiples of the variable's on-disk chunks, if it has any."""
    if chunks is None:
        return da
    if chunks != "auto":
        return da.chunk(chunks)
    import dask.array as dsa
    stored = da.encoding.get("preferred_chunks") or (
        dict(zip(da.dims, da.encoding["chunksizes"])) if da.encoding.get("chunksizes") else None)
    if stored:
        prev = tuple(stored.get(d, n) for d, n in zip(da.dims, da.shape))
        norm = dsa.core.normalize_chunks("auto", shape=da.shape, dtype=da.dtype, previous_chunks=prev)
    else:
        norm = dsa.core.normalize_chunks("auto", shape=da.shape, dtype=da.dtype)
    return da.chunk(dict(zip(da.dims, norm)))


def standardise(parts: list[xr.Dataset], row, chunks=None) -> xr.Dataset:
    """Combine a member's parts into one dataset with:
    time fix applied, duplicates removed (first kept), time sorted,
    the data variable renamed to row['var'], units scaled, dim named 'time'.
    chunks: None (no dask), "auto" (aligned with storage), or a dict."""
    var, file_var = row["var"], row.get("file_var") or row["var"]
    fixed = []
    for ds in parts:
        if row.get("time_fix") == "center_of_bounds":
            ds = center_time(ds)
        tdim = time_dim(ds)
        if tdim != "time":
            ds = ds.rename({tdim: "time"})
        v = data_var(ds, [file_var, var])
        part = ds[[v]].rename({v: var})
        part[var] = chunk_like_storage(part[var], chunks)
        fixed.append(part)
    ds = fixed[0] if len(fixed) == 1 else xr.concat(
        fixed, dim="time", data_vars="minimal", coords="minimal", compat="override")
    _, keep = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=np.sort(keep)).sortby("time")
    scale = float(row.get("unit_scale") or 1.0)
    if scale != 1.0:
        attrs = dict(ds[var].attrs)
        ds[var] = ds[var] * scale
        attrs["units_original"] = attrs.get("units")
        attrs["units"] = {"pr": "kg m-2 s-1"}.get(var, f"{attrs.get('units')} x{scale:g}")
        ds[var].attrs = attrs
    return ds


def load_member(row, chunks=None) -> xr.Dataset:
    """Standardised dataset for one row of check_existence() output.
    chunks="auto" streams the data variable through dask in storage-aligned chunks."""
    return standardise(open_parts(row), row, chunks=chunks)


# ---------------------------------------------------------------------
# Fixed fields
# ---------------------------------------------------------------------

def fx_path(cfg: dict, model: str, name: str) -> str | None:
    return ((cfg["models"][model].get("fx")) or {}).get(name)


def open_fx(cfg: dict, model: str, name: str) -> xr.DataArray | None:
    """Open an fx field (areacella, areacello, sftlf) as a DataArray."""
    path = fx_path(cfg, model, name)
    if path is None:
        return None
    ds = open_nc(path)
    return ds[data_var(ds, [name])]
