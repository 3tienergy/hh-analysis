"""
HH Data Analysis Tool
Streamlit app for visualising Half Hourly electricity demand vs ASC and solar production.
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

# ── Constants ──────────────────────────────────────────────────────────────────
PF = 0.8          # Assumed power factor
HH_PERIODS = 17520  # 48 periods × 365 days
SOLAR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "P3 Solar Production.xlsx")


# ══════════════════════════════════════════════════════════════════════════════
#  Parsing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _str_col(col) -> str:
    """Normalise column name to a plain string."""
    if isinstance(col, (datetime.time,)):
        return col.strftime("%H:%M")
    return str(col).strip()


def _is_time_col(col_str: str) -> bool:
    """Return True if the normalised column name looks like HH:MM."""
    return bool(re.match(r"^\d{1,2}:\d{2}$", col_str.strip()))


def _find_date_col(df: pd.DataFrame) -> str:
    """Return the name of the first column that can be parsed as dates."""
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col].dropna().head(10), errors="coerce")
            if parsed.notna().sum() >= 3:
                return col
        except Exception:
            pass
    return df.columns[0]


def _extract_mpan(df: pd.DataFrame):
    """Find MPAN column, extract its value, drop the column.  Returns (df, mpan_str|None)."""
    for col in df.columns:
        if "mpan" in col.lower():
            vals = df[col].dropna().astype(str)
            mpan = vals.iloc[0] if len(vals) else None
            return df.drop(columns=[col]), mpan
    return df, None


def _to_float_or_nan(val) -> float:
    """Convert a cell value to float, returning NaN for blanks / 'N/A' / etc."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().upper()
    if s in ("", "N/A", "NA", "NAN", "-", "NULL"):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ── Wide format (one row per day, 48 half-hour columns) ───────────────────────

def _parse_wide(df: pd.DataFrame, date_col: str, value_cols: list) -> pd.Series:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).reset_index(drop=True)

    records: dict = {}
    for _, row in df.iterrows():
        base = row[date_col].normalize()
        for i, col in enumerate(value_cols):
            ts = base + pd.Timedelta(minutes=i * 30)
            records[ts] = _to_float_or_nan(row[col])

    s = pd.Series(records, name="kWh")
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


# ── Long format (timestamp + kWh column) ──────────────────────────────────────

def _parse_long(df: pd.DataFrame, date_col: str, num_cols: list) -> pd.Series:
    if not num_cols:
        raise ValueError("No numeric data column found in HHD file.")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["_v"] = pd.to_numeric(df[num_cols[0]], errors="coerce")
    s = df.set_index(date_col)["_v"].sort_index()
    s.name = "kWh"
    return s


# ── Public parse function ─────────────────────────────────────────────────────

def parse_hhd(uploaded_file) -> tuple[pd.Series, str | None]:
    """
    Parse an HHD file (xlsx / csv).
    Returns (series_kWh_30min, mpan_string_or_None).
    """
    fname = uploaded_file.name.lower()
    if fname.endswith(".csv"):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    else:
        df = pd.read_excel(uploaded_file, sheet_name=0)

    # Drop fully empty rows / columns
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [_str_col(c) for c in df.columns]

    # Extract MPAN if present
    df, mpan = _extract_mpan(df)

    # Identify time columns
    time_cols = [c for c in df.columns if _is_time_col(c)]

    # Find date column
    date_col = _find_date_col(df)

    if len(time_cols) >= 24:
        # Wide format: date + ≥24 named HH:MM columns
        # Order them correctly: sort by the time they represent, keeping ' 0:00' (midnight)
        # at the end (period 48).  We use their positional order from the file instead.
        ordered = [c for c in df.columns if c in time_cols]
        return _parse_wide(df, date_col, ordered), mpan

    # Fallback: count numeric columns
    non_date_cols = [c for c in df.columns if c != date_col]
    num_cols = [c for c in non_date_cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(num_cols) >= 24:
        # Wide format with non-time column headers
        return _parse_wide(df, date_col, num_cols), mpan

    # Long format
    return _parse_long(df, date_col, num_cols), mpan


# ══════════════════════════════════════════════════════════════════════════════
#  Gap filling
# ══════════════════════════════════════════════════════════════════════════════

def fill_gaps(s: pd.Series) -> pd.Series:
    """
    Fill NaN runs in a 30-min series:
      • Run ≤ 3 days (≤ 144 periods): replace each missing slot with the value
        from the same slot one day earlier (walking back as needed).
      • Run > 3 days: set to 0.
    """
    arr = s.to_numpy(dtype=float, copy=True)
    n = len(arr)
    nan_mask = np.isnan(arr)

    if not nan_mask.any():
        return s

    # Identify consecutive NaN runs as (start_idx, end_idx) inclusive
    runs: list[tuple[int, int]] = []
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
                # Walk back in 48-step (1-day) increments until we find a value
                src = idx - 48
                filled = False
                while src >= 0:
                    if not np.isnan(arr[src]):
                        arr[idx] = arr[src]
                        filled = True
                        break
                    src -= 48
                if not filled:
                    arr[idx] = 0.0
        else:
            arr[start : end + 1] = 0.0

    return pd.Series(arr, index=s.index, name=s.name)


# ══════════════════════════════════════════════════════════════════════════════
#  12-month extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_12_months(s: pd.Series) -> pd.Series:
    """
    Return the most recent 12 months (17 520 × 30-min slots) of data,
    reindexed to a complete 30-min DatetimeIndex so gaps become NaN.
    """
    s = s.sort_index()
    # Use last valid timestamp as the end point
    last_valid = s.last_valid_index()
    if last_valid is None:
        raise ValueError("No valid data found in the uploaded file.")

    end_dt = pd.Timestamp(last_valid).floor("30min")
    start_dt = end_dt - pd.Timedelta(days=365) + pd.Timedelta(minutes=30)

    full_idx = pd.date_range(start=start_dt, periods=HH_PERIODS, freq="30min")
    return s.reindex(full_idx)


# ══════════════════════════════════════════════════════════════════════════════
#  Solar data
# ══════════════════════════════════════════════════════════════════════════════

def load_solar(hhd_index: pd.DatetimeIndex) -> pd.Series:
    """
    Load the bundled P3 Solar Production file and align it to the HHD index.
    Solar values are already negative (export convention); they are returned as-is
    so they plot naturally on the negative y-axis.
    """
    solar = pd.read_excel(SOLAR_FILE, sheet_name=0)
    solar.columns = [_str_col(c) for c in solar.columns]
    ts_col, val_col = solar.columns[0], solar.columns[1]

    solar[ts_col] = pd.to_datetime(solar[ts_col], errors="coerce")
    solar = solar.dropna(subset=[ts_col]).copy()
    solar["val"] = pd.to_numeric(solar[val_col], errors="coerce").fillna(0)
    solar = solar.set_index(ts_col)["val"].sort_index()

    # Deduplicate timestamps (take mean) before resampling
    solar = solar.groupby(solar.index).mean()

    # Upsample to 30-min if the source is hourly (or coarser)
    if len(solar) > 1:
        med_diff = solar.index.to_series().diff().median()
        if med_diff > pd.Timedelta("25min"):
            # Resample to 30-min: first create the new index, then interpolate
            new_idx = pd.date_range(
                start=solar.index[0], end=solar.index[-1], freq="30min"
            )
            solar = solar.reindex(solar.index.union(new_idx)).interpolate(
                method="time"
            ).reindex(new_idx)

    # Build month-day-HHmm lookup (seasonal / generic pattern)
    key_fn = lambda idx: idx.strftime("%m-%d %H:%M")  # noqa: E731
    solar_df = solar.to_frame("val")
    solar_df["key"] = key_fn(solar_df.index)
    lookup = solar_df.groupby("key")["val"].mean()

    # Map each HHD timestamp to the lookup; handle leap-year Feb-29 fallback
    hhd_keys = pd.Series(hhd_index).dt.strftime("%m-%d %H:%M")
    # For Feb-29 entries not in the lookup, fall back to Feb-28
    feb29_mask = hhd_keys.str.startswith("02-29")
    hhd_keys_adj = hhd_keys.copy()
    hhd_keys_adj[feb29_mask] = hhd_keys_adj[feb29_mask].str.replace(
        "02-29", "02-28", regex=False
    )

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
    mpan: str | None,
) -> go.Figure:
    fig = go.Figure()

    # Demand profile
    fig.add_trace(
        go.Scatter(
            x=demand_kva.index,
            y=demand_kva.values,
            mode="lines",
            name="Demand (kVA)",
            line=dict(color="#374345", width=0.6),
        )
    )

    # ASC horizontal line
    x_ends = [demand_kva.index.min(), demand_kva.index.max()]
    fig.add_trace(
        go.Scatter(
            x=x_ends,
            y=[asc_kva, asc_kva],
            mode="lines",
            name=f"ASC – {asc_kva:,.0f} kVA",
            line=dict(color="#E63946", width=1.8, dash="dash"),
        )
    )

    # Solar production (negative y-axis; values are already negative)
    fig.add_trace(
        go.Scatter(
            x=solar.index,
            y=solar.values,
            mode="lines",
            name="Solar Production (kVA)",
            line=dict(color="#fdb913", width=0.6),
            fill="tozeroy",
            fillcolor="rgba(253,185,19,0.15)",
        )
    )

    # Zero reference line
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.4)

    # Title & MPAN annotation
    title_text = f"<b>{client_name}</b>  –  HH Data Analysis"

    annotations = []
    if mpan:
        annotations.append(
            dict(
                text=f"MPAN: {mpan}",
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.10,
                showarrow=False,
                font=dict(size=11, color="black"),
                xanchor="left",
                yanchor="bottom",
            )
        )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=20, color="black"),
            x=0.0,
            xanchor="left",
            y=0.96 if not mpan else 0.93,
        ),
        annotations=annotations,
        xaxis=dict(
            title="Date",
            tickformat="%b %Y",
            dtick="M1",
            showgrid=True,
            gridcolor="#E8E8E8",
            color="black",
            linecolor="#AAAAAA",
            mirror=True,
        ),
        yaxis=dict(
            title="kVA",
            showgrid=True,
            gridcolor="#E8E8E8",
            zeroline=False,
            color="black",
            linecolor="#AAAAAA",
            mirror=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(color="black", size=11),
            bgcolor="white",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", family="Arial, sans-serif"),
        height=620,
        margin=dict(l=65, r=30, t=130, b=65),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="HH Data Analysis", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        h1 { font-size: 1.6rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("HH Data Analysis")
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
            "Agreed Supply Capacity (ASC)",
            min_value=1,
            value=1000,
            step=1,
        )
        asc_unit = st.selectbox(
            "Unit",
            ["kVA", "kW"],
            help="kW will be converted to kVA assuming PF = 0.8",
        )

    with col_name:
        client_name = st.text_input(
            "Client / Site Name",
            placeholder="e.g. Acme Factory – Site A",
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
                # 1. ASC → kVA
                asc_kva = float(asc_value) if asc_unit == "kVA" else float(asc_value) / PF

                # 2. Parse HHD
                raw_series, mpan = parse_hhd(uploaded_file)

                # 3. Extract most recent 12 months & reindex to complete grid
                windowed = extract_12_months(raw_series)

                # 4. Fill gaps
                filled = fill_gaps(windowed)

                # 5. kWh → kVA  (÷ 0.5 hours ÷ PF)
                demand_kva = filled / 0.5 / PF

                # 6. Solar (aligned to same DatetimeIndex)
                solar = load_solar(demand_kva.index)

                # 7. Chart
                fig = make_chart(
                    demand_kva,
                    solar,
                    asc_kva,
                    client_name.strip(),
                    mpan,
                )

                # ── Summary metrics ────────────────────────────────────────
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
                    st.info(f"MPAN detected in file: **{mpan}**")

                nan_pct = (windowed.isna().sum() / len(windowed)) * 100
                if nan_pct > 0:
                    st.warning(
                        f"{nan_pct:.1f}% of slots were missing and have been "
                        "estimated or set to zero (runs > 3 days)."
                    )

                # ── Chart ──────────────────────────────────────────────────
                st.plotly_chart(fig, use_container_width=True)

            except Exception as exc:
                st.error(f"Failed to process the file: {exc}")
                with st.expander("Full error details"):
                    st.exception(exc)
