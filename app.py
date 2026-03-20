"""
HH Data Analysis Tool – v2
Streamlit app for visualising Half Hourly electricity demand vs ASC and solar production.
Handles multiple file formats, multiple sheets, and multiple sites/meters.
"""

import os
import re
import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

PF = 0.8
HH_PERIODS = 17520  # 48 × 365
SOLAR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "P3 Solar Production.xlsx")
MPAN_RE = re.compile(r'\b(\d{21}|\d{13})\b')


# ══════════════════════════════════════════════════════════════════════════════
#  MPAN helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_mpan(s) -> "str | None":
    m = MPAN_RE.search(str(s))
    return m.group(1) if m else None


# ══════════════════════════════════════════════════════════════════════════════
#  Column normalisation
# ══════════════════════════════════════════════════════════════════════════════

def _str_col(col) -> str:
    if isinstance(col, datetime.time):
        return col.strftime("%H:%M")
    if isinstance(col, datetime.timedelta):
        total_min = int(col.total_seconds() // 60)
        h, m = divmod(total_min, 60)
        return f"{h:02d}:{m:02d}"
    return str(col).strip()


def _is_time_col(s: str) -> bool:
    """Match HH:MM or HHH:MM (e.g. 24:00 for end-of-day midnight)."""
    return bool(re.match(r'^\d{1,3}:\d{2}$', s))


def _is_hh_indexed(s: str) -> bool:
    """Match HH1..HH48 but NOT HH1.1 quality-flag variants."""
    return bool(re.match(r'^HH\d{1,2}$', s, re.I))


def _is_kwh_indexed(s: str) -> bool:
    return bool(re.match(r'^kwh_?\d+$', s, re.I))


def _is_numeric_indexed(s: str) -> bool:
    try:
        v = int(s)
        return 1 <= v <= 96
    except (ValueError, TypeError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Date-column detection
# ══════════════════════════════════════════════════════════════════════════════

def _find_date_col(df: pd.DataFrame) -> "str | None":
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        sample = df[col].dropna().head(10)
        if len(sample) == 0:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= 3:
                return col
        except Exception:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MPAN extraction from DataFrame
# ══════════════════════════════════════════════════════════════════════════════

_MPAN_COL_RE = re.compile(r'mpan|identifier|meternumber|meterref|supplyno|supply\b', re.I)


def _extract_mpan_from_df(df: pd.DataFrame) -> "tuple[pd.DataFrame, str | None]":
    """Detect and remove MPAN column; return (cleaned_df, mpan_string|None)."""
    for col in df.columns:
        if _MPAN_COL_RE.search(str(col)):
            vals = df[col].dropna().astype(str)
            for v in vals:
                mpan = find_mpan(v)
                if mpan:
                    return df.drop(columns=[col]), mpan
    # Scan column names themselves
    for col in df.columns:
        mpan = find_mpan(str(col))
        if mpan:
            return df, mpan
    # Scan first 5 rows of every column
    for col in df.columns:
        for v in df[col].dropna().head(5).astype(str):
            mpan = find_mpan(v)
            if mpan:
                return df, mpan
    return df, None


# ══════════════════════════════════════════════════════════════════════════════
#  Sheet-skip heuristic
# ══════════════════════════════════════════════════════════════════════════════

def _is_empty_or_metadata_sheet(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 5:
        return True
    numeric_count = df.select_dtypes(include=[np.number]).notna().sum().sum()
    # Pass on numeric count OR many rows (covers messy-header sheets where
    # the first data row holds strings, making pandas infer object dtypes)
    return numeric_count < 336 and len(df) < 28


# ══════════════════════════════════════════════════════════════════════════════
#  Grouping-column detection (multi-site in one sheet)
# ══════════════════════════════════════════════════════════════════════════════

_GROUP_COL_RE = re.compile(r'^(sites?|location|meter|mpan)\s*$', re.I)
_LABEL_COL_RE = re.compile(r'^(sites?|location|name|sitename)\s*$', re.I)


def _find_group_col(df: pd.DataFrame) -> "str | None":
    for col in df.columns:
        if _GROUP_COL_RE.match(str(col).strip()):
            n_unique = df[col].nunique()
            if 1 < n_unique <= 500:
                return col
    return None


def _find_label_col(df: pd.DataFrame, group_col: str) -> "str | None":
    for col in df.columns:
        if col != group_col and _LABEL_COL_RE.match(str(col).strip()):
            return col
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Value conversion helper
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(val) -> float:
    if val is None:
        return np.nan
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    s = str(val).strip().upper()
    if s in ("", "N/A", "NA", "NAN", "-", "NULL"):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  Wide / Long parsers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_wide(df: pd.DataFrame, date_col: str, hh_cols: list) -> pd.Series:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Drop metadata rows (rows where HH columns contain non-numeric strings)
    def _is_data_row(row) -> bool:
        for c in hh_cols[:4]:
            v = row[c]
            if pd.notna(v):
                try:
                    float(v)
                    return True
                except (ValueError, TypeError):
                    return False
        return True  # all NaN → keep (gap-fill later)

    df = df[df.apply(_is_data_row, axis=1)].reset_index(drop=True)

    records: dict = {}
    for _, row in df.iterrows():
        base = row[date_col].normalize()
        for i, col in enumerate(hh_cols):
            ts = base + pd.Timedelta(minutes=i * 30)
            records[ts] = _to_float(row[col])

    s = pd.Series(records, name="kWh")
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def _parse_long(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period_col: "str | None" = None,
) -> pd.Series:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["_v"] = pd.to_numeric(df[value_col], errors="coerce")

    if period_col:
        def _p_to_min(p) -> int:
            try:
                return (int(str(p).replace("P", "").strip()) - 1) * 30
            except (ValueError, TypeError):
                return -1

        df["_min"] = df[period_col].apply(_p_to_min)
        df = df[df["_min"] >= 0]
        df["_ts"] = df[date_col].dt.normalize() + pd.to_timedelta(df["_min"], unit="m")
        s = df.set_index("_ts")["_v"]
    else:
        s = df.set_index(date_col)["_v"]

    s = s.sort_index()
    s.name = "kWh"
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Best value-column picker (long format)
# ══════════════════════════════════════════════════════════════════════════════

_ENERGY_RE = re.compile(r'kwh|hhc|units?|energy|reading|demand|value|consumption', re.I)
_SKIP_RE = re.compile(r'total|flag|status|quality|aei|period|count|avg|mean|max|min', re.I)


def _pick_value_col(df: pd.DataFrame, candidates: list) -> str:
    scored = []
    for c in candidates:
        score = 0
        if _ENERGY_RE.search(str(c)):
            score += 10
        if _SKIP_RE.search(str(c)):
            score -= 5
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) > 10 and 0.1 < vals.mean() < 10_000:
            score += 5
        scored.append((score, c))
    scored.sort(reverse=True)
    return scored[0][1]


# ══════════════════════════════════════════════════════════════════════════════
#  Main single-DataFrame parser
# ══════════════════════════════════════════════════════════════════════════════

_SKIP_COL_RE = re.compile(r'\b(total|sum|loaded|load|day|calb)\b', re.I)


def _parse_single_df(df: pd.DataFrame) -> "tuple[pd.Series, str | None]":
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [_str_col(c) for c in df.columns]
    df, mpan = _extract_mpan_from_df(df)

    date_col = _find_date_col(df)
    if date_col is None:
        raise ValueError("Could not find a date/timestamp column.")

    other = [c for c in df.columns if c != date_col]

    # Candidate column sets
    time_cols = [c for c in other if _is_time_col(c)]
    hh_indexed = [c for c in other if _is_hh_indexed(c)]
    kwh_indexed = [c for c in other if _is_kwh_indexed(c)]

    # Period column (P1-P48 in long format)
    period_col = None
    for c in other:
        sample = df[c].dropna().head(20).astype(str)
        if sample.str.match(r"^P\d{1,2}$").sum() >= 10:
            period_col = c
            break

    # ── Wide: HH:MM column names (preserving file order) ─────────────────────
    if len(time_cols) >= 24:
        ordered = [c for c in df.columns if c in set(time_cols)]
        return _parse_wide(df, date_col, ordered), mpan

    # ── Wide: HH1-HH48 style ─────────────────────────────────────────────────
    if len(hh_indexed) >= 24:
        ordered = sorted(hh_indexed, key=lambda x: int(re.search(r"\d+", x).group()))
        return _parse_wide(df, date_col, ordered), mpan

    # ── Wide: KWh_1-KWh_48 style ─────────────────────────────────────────────
    if len(kwh_indexed) >= 24:
        ordered = sorted(kwh_indexed, key=lambda x: int(re.search(r"\d+", x).group()))
        return _parse_wide(df, date_col, ordered), mpan

    # ── Wide: numeric column names 1-48 ──────────────────────────────────────
    num_named = [c for c in other if _is_numeric_indexed(c)]
    if len(num_named) >= 24:
        ordered = sorted(num_named, key=lambda x: int(x))[:48]
        return _parse_wide(df, date_col, ordered), mpan

    # ── Wide: any ≥24 numeric columns (fallback) ─────────────────────────────
    num_cols = [
        c for c in other
        if pd.api.types.is_numeric_dtype(df[c]) and not _SKIP_COL_RE.search(c)
    ]
    if len(num_cols) >= 24:
        return _parse_wide(df, date_col, num_cols[:48]), mpan

    # ── Long: period-based (P1-P48) ──────────────────────────────────────────
    if period_col:
        candidates = [c for c in other if c != period_col]
        num_cands = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        val_col = _pick_value_col(df, num_cands or candidates)
        return _parse_long(df, date_col, val_col, period_col), mpan

    # ── Long: timestamp + single value ───────────────────────────────────────
    num_cols = [c for c in other if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        val_col = _pick_value_col(df, num_cols)
        return _parse_long(df, date_col, val_col), mpan

    raise ValueError(f"Unrecognised HH data format. Columns: {list(df.columns)[:10]}")


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level: parse all sites from an uploaded file
# ══════════════════════════════════════════════════════════════════════════════

def parse_all_sites(uploaded_file) -> "tuple[list, list]":
    """
    Returns (results, warnings) where results is a list of
    (series_kWh_30min, mpan_or_None, display_label).
    """
    fname = uploaded_file.name
    results = []
    parse_warnings = []

    # ── CSV ──────────────────────────────────────────────────────────────────
    if fname.lower().endswith(".csv"):
        df = None
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Could not decode CSV file.")
        series, mpan = _parse_single_df(df)
        label = mpan or os.path.splitext(fname)[0]
        return [(series, mpan, label)], []

    # ── Excel ────────────────────────────────────────────────────────────────
    xl = pd.ExcelFile(uploaded_file)

    for sheet_name in xl.sheet_names:
        try:
            df = xl.parse(sheet_name, header=0)
        except Exception as e:
            parse_warnings.append(f"Could not read sheet '{sheet_name}': {e}")
            continue

        if _is_empty_or_metadata_sheet(df):
            continue

        # Normalise column names to detect grouping columns
        df_norm = df.copy()
        df_norm.columns = [_str_col(c) for c in df_norm.columns]
        group_col = _find_group_col(df_norm)

        if group_col:
            # Multi-site sheet: one chart per unique meter/site
            label_col = _find_label_col(df_norm, group_col)
            for group_val, group_df in df_norm.groupby(group_col, sort=False):
                try:
                    site_name = (
                        str(group_df[label_col].iloc[0]).strip()
                        if label_col else None
                    )
                    drop_cols = [c for c in [group_col, label_col] if c]
                    sub_df = group_df.drop(columns=drop_cols).reset_index(drop=True)
                    series, mpan = _parse_single_df(sub_df)
                    if mpan is None:
                        mpan = find_mpan(str(group_val))
                    if site_name and mpan:
                        label = f"{site_name} ({mpan})"
                    elif site_name:
                        label = site_name
                    else:
                        label = mpan or str(group_val)
                    results.append((series, mpan, label))
                except Exception as e:
                    parse_warnings.append(
                        f"Skipped '{group_val}' in sheet '{sheet_name}': {e}"
                    )
        else:
            try:
                series, mpan = _parse_single_df(df)
                if mpan is None:
                    mpan = find_mpan(sheet_name) or find_mpan(fname)
                label = mpan or sheet_name
                results.append((series, mpan, label))
            except Exception as e:
                parse_warnings.append(f"Skipped sheet '{sheet_name}': {e}")

    if not results:
        raise ValueError("No valid HH data found. Could not parse any sheet.")

    return results, parse_warnings


# ══════════════════════════════════════════════════════════════════════════════
#  Gap filling
# ══════════════════════════════════════════════════════════════════════════════

def fill_gaps(s: pd.Series) -> pd.Series:
    arr = s.to_numpy(dtype=float, copy=True)
    n = len(arr)
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return s

    runs: list = []
    in_run = False
    for i in range(n):
        if nan_mask[i]:
            if not in_run:
                run_start = i
                in_run = True
        else:
            if in_run:
                runs.append((run_start, i - 1))
                in_run = False
    if in_run:
        runs.append((run_start, n - 1))

    for start, end in runs:
        length = end - start + 1
        if length / 48 <= 3:
            for idx in range(start, end + 1):
                src = idx - 48
                while src >= 0:
                    if not np.isnan(arr[src]):
                        arr[idx] = arr[src]
                        break
                    src -= 48
                else:
                    arr[idx] = 0.0
        else:
            arr[start:end + 1] = 0.0

    return pd.Series(arr, index=s.index, name=s.name)


# ══════════════════════════════════════════════════════════════════════════════
#  12-month extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_12_months(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    last_valid = s.last_valid_index()
    if last_valid is None:
        raise ValueError("No valid data found in the uploaded file.")
    end_dt = pd.Timestamp(last_valid).floor("30min")
    start_dt = end_dt - pd.Timedelta(days=365) + pd.Timedelta(minutes=30)
    full_idx = pd.date_range(start=start_dt, periods=HH_PERIODS, freq="30min")
    return s.reindex(full_idx)


# ══════════════════════════════════════════════════════════════════════════════
#  Solar
# ══════════════════════════════════════════════════════════════════════════════

def load_solar(hhd_index: pd.DatetimeIndex) -> pd.Series:
    solar = pd.read_excel(SOLAR_FILE, sheet_name=0)
    solar.columns = [_str_col(c) for c in solar.columns]
    ts_col, val_col = solar.columns[0], solar.columns[1]
    solar[ts_col] = pd.to_datetime(solar[ts_col], errors="coerce")
    solar = solar.dropna(subset=[ts_col]).copy()
    solar["val"] = pd.to_numeric(solar[val_col], errors="coerce").fillna(0)
    solar = solar.set_index(ts_col)["val"].sort_index()
    solar = solar.groupby(solar.index).mean()

    if len(solar) > 1:
        med_diff = solar.index.to_series().diff().median()
        if med_diff > pd.Timedelta("25min"):
            new_idx = pd.date_range(start=solar.index[0], end=solar.index[-1], freq="30min")
            solar = (
                solar.reindex(solar.index.union(new_idx))
                .interpolate(method="time")
                .reindex(new_idx)
            )

    solar_df = solar.to_frame("val")
    solar_df["key"] = solar_df.index.strftime("%m-%d %H:%M")
    lookup = solar_df.groupby("key")["val"].mean()

    hhd_keys = pd.Series(hhd_index).dt.strftime("%m-%d %H:%M")
    hhd_keys_adj = hhd_keys.copy()
    feb29 = hhd_keys.str.startswith("02-29")
    hhd_keys_adj[feb29] = hhd_keys_adj[feb29].str.replace("02-29", "02-28", regex=False)

    aligned = hhd_keys_adj.map(lookup).fillna(0).values
    return pd.Series(aligned, index=hhd_index, name="solar")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart
# ══════════════════════════════════════════════════════════════════════════════

def make_chart(
    demand_kva: pd.Series,
    solar: pd.Series,
    asc_kva: float,
    client_name: str,
    site_label: "str | None",
    mpan: "str | None",
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=demand_kva.index, y=demand_kva.values,
        name="Demand (kVA)",
        marker_color="#374345",
        width=1_800_000,  # 30 min in milliseconds — fills each HH slot
    ))

    x_ends = [demand_kva.index.min(), demand_kva.index.max()]
    fig.add_trace(go.Scatter(
        x=x_ends, y=[asc_kva, asc_kva],
        mode="lines", name=f"ASC – {asc_kva:,.0f} kVA",
        line=dict(color="#E63946", width=1.8, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=solar.index, y=solar.values,
        mode="lines", name="Solar Production (kVA)",
        line=dict(color="#fdb913", width=0.6),
        fill="tozeroy", fillcolor="rgba(253,185,19,0.15)",
    ))

    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.4)

    title_parts = [f"<b>{client_name}</b>"]
    if site_label:
        title_parts.append(site_label)
    title_parts.append("HH Data Analysis")
    title_text = " – ".join(title_parts)

    annotations = []
    if mpan:
        annotations.append(dict(
            text=f"MPAN: {mpan}", xref="paper", yref="paper",
            x=0.0, y=1.10, showarrow=False,
            font=dict(size=11, color="black"),
            xanchor="left", yanchor="bottom",
        ))

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, color="black"),
            x=0.0, xanchor="left",
            y=0.96 if not mpan else 0.93,
        ),
        annotations=annotations,
        xaxis=dict(
            title="Date", tickformat="%b %Y", dtick="M1",
            showgrid=True, gridcolor="#E8E8E8",
            color="black", linecolor="#AAAAAA", mirror=True,
        ),
        yaxis=dict(
            title="kVA", showgrid=True, gridcolor="#E8E8E8",
            zeroline=False, color="black", linecolor="#AAAAAA", mirror=True,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            font=dict(color="black", size=11),
            bgcolor="white", bordercolor="#CCCCCC", borderwidth=1,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial, sans-serif"),
        height=580,
        margin=dict(l=65, r=30, t=130, b=65),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

_3TI_SVG = """<svg id="Layer_2" data-name="Layer 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 199.67 131.05" height="44" width="auto">
  <defs><style>.cls-1{fill:#fff;}.cls-2{fill:#f9a800;}</style></defs>
  <g id="Layer_1-2" data-name="Layer 1"><g>
    <path class="cls-1" d="M48.58,128.78c-24.59,0-39.57-10.53-45.72-32.16l24.92-3.08c3.14,12.57,11.63,15.29,18.38,15.29,4.68,0,8.57-1.27,11.58-3.76,3.13-2.6,4.72-6.12,4.72-10.47,0-2.93-.87-5.56-2.58-7.81-1.63-2.14-3.44-3.68-5.37-4.59-1.93-.9-4.96-1.33-9.26-1.33h-11.51v-19.12h9.35c6.99,0,11.48-1.37,13.73-4.17,2.09-2.62,3.16-5.41,3.16-8.31,0-3.65-1.37-6.73-4.07-9.16-2.63-2.37-6.08-3.57-10.24-3.57-5.78,0-13.05,2.26-15.92,12.65l-22.85-3.6c2.56-8.89,7.36-15.91,14.31-20.88,7.62-5.45,16.8-8.21,27.28-8.21,11.83,0,21.42,2.77,28.5,8.25,7,5.41,10.41,12.42,10.41,21.45,0,5.29-1.46,9.95-4.33,13.85-2.91,3.95-6.51,6.53-11,7.89l-7.15,2.17,7.14,2.18c12.66,3.86,18.82,11.87,18.82,24.48,0,10.17-3.75,17.84-11.48,23.45-7.84,5.69-18.22,8.58-30.84,8.58Z"/>
    <path class="cls-1" d="M169.29,127.06V58.43c10.38,8.52,19.83,18.15,28.14,28.69v39.94h-28.14Z"/>
    <path class="cls-1" d="M137.12,128.12c-9.42,0-16.36-1.98-20.63-5.88-4.21-3.84-6.35-10.23-6.35-18.96l.08-9.73v-28.58h-12.45v-16.67h12.41l.41-22.37c9.13,3.06,18.06,6.86,26.63,11.32v11.05h16.63v16.67h-16.63v32.99c0,3.31.38,7.9,3.63,9.53,1.89.95,3.89,1.43,5.93,1.43s4.45-.27,7.07-.81v18.52c-5.66,1-11.27,1.5-16.73,1.5Z"/>
    <path class="cls-2" d="M199.67,46.88C165.51,17.66,121.16,0,72.68,0,48.64,0,25.62,4.35,4.35,12.29h.01c15.12-3.74,30.92-5.74,47.2-5.74,59.22,0,112.27,26.34,148.11,67.94v-27.59Z"/>
  </g></g>
</svg>"""

st.set_page_config(page_title="HH Data Analysis", layout="wide")

_header_html = (
    '<div style="background:#222926;padding:12px 24px;border-radius:8px;display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;">'
    '<div style="color:#fff;font-family:Arial,sans-serif;font-size:1.2rem;font-weight:600;letter-spacing:0.02em;">'
    'HH Data Analysis&nbsp;<span style="color:#fdb913;">|</span>&nbsp;Half Hourly Demand Profiler'
    '</div>'
    '<div style="height:44px;display:flex;align-items:center;">'
    + _3TI_SVG +
    '</div></div>'
)
st.markdown(_header_html, unsafe_allow_html=True)

st.write(
    "Upload Half Hourly electricity demand data, enter the Agreed Supply Capacity "
    "and site name, then click **Generate Chart**."
)
st.divider()

with st.form("inputs", border=False):
    col_file, col_asc, col_name = st.columns([3, 1.6, 2])

    with col_file:
        uploaded_file = st.file_uploader(
            "Half Hourly Data (HHD)",
            type=["xlsx", "xls", "csv"],
            help="Accepted formats: .xlsx, .xls, .csv",
        )

    with col_asc:
        asc_value = st.number_input(
            "Agreed Supply Capacity (ASC)", min_value=1, value=1000, step=1
        )
        asc_unit = st.selectbox(
            "Unit", ["kVA", "kW"],
            help="kW will be converted to kVA assuming PF = 0.8",
        )

    with col_name:
        client_name = st.text_input(
            "Client / Site Name", placeholder="e.g. Acme Factory – Site A"
        )

    submitted = st.form_submit_button("Generate Chart", type="primary")

# ── Processing ─────────────────────────────────────────────────────────────────
if submitted:
    errors = []
    if not uploaded_file:
        errors.append("Please upload a Half Hourly Data file.")
    if not client_name.strip():
        errors.append("Please enter a Client / Site Name.")
    for e in errors:
        st.error(e)

    if not errors:
        with st.spinner("Processing data…"):
            try:
                asc_kva = float(asc_value) if asc_unit == "kVA" else float(asc_value) / PF

                sites, parse_warnings = parse_all_sites(uploaded_file)

                for w in parse_warnings:
                    st.warning(w)

                # Process every site
                site_results = []
                for raw_series, mpan, label in sites:
                    try:
                        windowed = extract_12_months(raw_series)
                        filled = fill_gaps(windowed)
                        demand_kva = filled / 0.5 / PF
                        solar = load_solar(demand_kva.index)
                        fig = make_chart(
                            demand_kva, solar, asc_kva,
                            client_name.strip(),
                            label if len(sites) > 1 else None,
                            mpan,
                        )
                        nan_pct = (windowed.isna().sum() / len(windowed)) * 100
                        site_results.append((fig, demand_kva, windowed, mpan, label, nan_pct))
                    except Exception as e:
                        st.warning(f"Could not generate chart for '{label}': {e}")

                if not site_results:
                    st.error("No charts could be generated.")
                else:
                    n = len(site_results)
                    st.success(f"Found **{n}** site{'s' if n > 1 else ''} / meter{'s' if n > 1 else ''}.")

                    def _render(fig, demand_kva, windowed, mpan, label, nan_pct):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Data points", f"{len(demand_kva):,}")
                        c2.metric(
                            "Period",
                            f"{demand_kva.index.min().strftime('%b %Y')} – "
                            f"{demand_kva.index.max().strftime('%b %Y')}",
                        )
                        c3.metric("Peak demand", f"{demand_kva.max():,.0f} kVA")
                        c4.metric("ASC", f"{asc_kva:,.0f} kVA")
                        if mpan:
                            st.info(f"MPAN: **{mpan}**")
                        if nan_pct > 0:
                            st.warning(
                                f"{nan_pct:.1f}% of slots were missing and have been "
                                "estimated or set to zero (runs > 3 days)."
                            )
                        st.plotly_chart(fig, use_container_width=True)

                    if n == 1:
                        _render(*site_results[0])
                    else:
                        tab_labels = [r[4][:40] for r in site_results]
                        tabs = st.tabs(tab_labels)
                        for tab, result in zip(tabs, site_results):
                            with tab:
                                _render(*result)

            except Exception as exc:
                st.error(f"Failed to process the file: {exc}")
                with st.expander("Full error details"):
                    st.exception(exc)
