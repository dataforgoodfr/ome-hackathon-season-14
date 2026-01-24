from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st


SPLIT_COLUMN = "split"

DEFAULT_CATEGORICAL_COLUMNS = [
    "category",
    "llm_category",
    "predicted_category",
    "channel_name",
    "channel_title",
]

DEFAULT_TEXT_COLUMNS = [
    "report_text",
    "text",
]

DEFAULT_NUMERIC_COLUMNS = [
    "duration_seconds",
]

DEFAULT_THEMES_COLUMNS = [
    "themes",
    "theme",
    "topics",
    "topic",
]


def _split_multi_value_cell(value: object, *, sep: str) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, (list, tuple, set)):
        parts = [str(x) for x in value]
    else:
        parts = str(value).split(sep)

    tokens: list[str] = []
    for p in parts:
        t = str(p).strip()
        if not t:
            continue
        if t.lower() in {"nan", "none", "<na>"}:
            continue
        tokens.append(t)
    return tokens


def _maybe_render_themes(
    df: pd.DataFrame,
    *,
    themes_col: str,
    top_n: int,
    sep: str,
    show_by_split: bool,
    show_cooccurrence: bool,
) -> None:
    if themes_col not in df.columns:
        return

    work_cols = [themes_col] + ([SPLIT_COLUMN] if SPLIT_COLUMN in df.columns else [])
    work = df[work_cols].copy()
    work[themes_col] = work[themes_col].apply(_split_multi_value_cell, sep=sep)
    exploded = work.explode(themes_col).dropna(subset=[themes_col])
    exploded[themes_col] = exploded[themes_col].astype("string").str.strip()
    exploded = exploded[exploded[themes_col] != ""]

    if exploded.empty:
        st.info("No themes found after splitting values.")
        return

    vc = exploded[themes_col].value_counts(dropna=False).head(int(top_n))
    counts = pd.DataFrame({"theme": vc.index.astype(str), "count": vc.values})
    fig = px.bar(
        counts.sort_values("count", ascending=True),
        x="count",
        y="theme",
        orientation="h",
        title=f"Top {len(counts)} themes",
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_by_split and SPLIT_COLUMN in df.columns:
        top_set = set(counts["theme"].astype(str).tolist())
        subset = exploded[exploded[themes_col].astype(str).isin(top_set)].copy()
        subset[SPLIT_COLUMN] = subset[SPLIT_COLUMN].astype("string").fillna("<NA>")
        by_split = (
            subset.groupby([themes_col, SPLIT_COLUMN], dropna=False)
            .size()
            .reset_index(name="count")
            .rename(columns={themes_col: "theme"})
        )
        fig2 = px.bar(
            by_split,
            y="theme",
            x="count",
            color=SPLIT_COLUMN,
            orientation="h",
            barmode="stack",
            title=f"Themes by {SPLIT_COLUMN} (top themes)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    if show_cooccurrence:
        try:
            import numpy as np  # type: ignore
        except Exception:
            st.info("Co-occurrence heatmap requires numpy.")
            return

        top_set = set(counts["theme"].astype(str).tolist())
        labels = list(counts["theme"].astype(str).tolist())
        idx = {t: i for i, t in enumerate(labels)}
        mat = np.zeros((len(labels), len(labels)), dtype=int)

        # Count co-occurrences within each row's theme set.
        row_tokens = df[themes_col].apply(_split_multi_value_cell, sep=sep)
        for toks in row_tokens:
            uniq = sorted({t for t in toks if t in top_set})
            for i, a in enumerate(uniq):
                ai = idx[a]
                mat[ai, ai] += 1
                for b in uniq[i + 1 :]:
                    bi = idx[b]
                    mat[ai, bi] += 1
                    mat[bi, ai] += 1

        fig3 = px.imshow(
            mat,
            x=labels,
            y=labels,
            title="Theme co-occurrence (top themes)",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig3, use_container_width=True)


def _normalize_keywords(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join([str(x) for x in value])
    # Common patterns: comma/semicolon separated keywords, or a JSON-ish string.
    s = str(value)
    return (
        s.replace("[", " ")
        .replace("]", " ")
        .replace("\"", " ")
        .replace("'", " ")
        .replace(",", " ")
        .replace(";", " ")
    )


def _default_umap_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = ["duration", "duration_seconds", "keywords", "category"]
    return [c for c in candidates if c in df.columns]


@st.cache_data(show_spinner=False)
def _build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    keywords_col: str | None,
    max_keywords_features: int,
    seed: int,
) -> tuple["object", pd.DataFrame]:
    """Return (X, meta_df).

    X is a (possibly sparse) feature matrix built from selected columns.
    meta_df contains the subset rows used, preserving original columns.
    """

    try:
        from scipy import sparse  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "UMAP feature engineering requires numpy/scipy/scikit-learn (already in project deps)."
        ) from exc

    if not feature_cols:
        raise RuntimeError("Select at least one feature column.")

    # Preserve original row identities; sampling leaves non-contiguous indexes.
    work = df[feature_cols].copy()
    row_index = work.index
    work = work.reset_index(drop=True)

    blocks: list["object"] = []

    # Numeric features
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
    if numeric_cols:
        num = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
        # fill with median for stability
        for c in numeric_cols:
            med = float(num[c].median()) if not num[c].dropna().empty else 0.0
            num[c] = num[c].fillna(med)
        scaler = StandardScaler()
        num_arr = scaler.fit_transform(num.to_numpy(dtype=float))
        blocks.append(sparse.csr_matrix(num_arr))

    # Categorical features (one-hot)
    categorical_cols = [
        c
        for c in feature_cols
        if c not in numeric_cols and c != keywords_col
    ]
    if categorical_cols:
        cat = work[categorical_cols].astype("string").fillna("<NA>")
        cat_dummies = pd.get_dummies(cat, dummy_na=False)
        if cat_dummies.shape[1] > 0:
            blocks.append(sparse.csr_matrix(cat_dummies.to_numpy(dtype=float)))

    # Keywords/text feature
    if keywords_col is not None and keywords_col in feature_cols:
        kw = work[keywords_col].map(_normalize_keywords).astype(str).fillna("")
        vectorizer = TfidfVectorizer(
            max_features=max_keywords_features,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        kw_mat = vectorizer.fit_transform(kw.tolist())
        blocks.append(kw_mat)

    if not blocks:
        raise RuntimeError(
            "No usable features were produced. Try selecting a numeric column or a categorical column."
        )

    X = sparse.hstack(blocks).tocsr()

    # Minimal metadata frame for downstream joins
    meta = df.loc[row_index].reset_index(drop=True)
    return X, meta


def _fingerprint_matrix(m: "object") -> str:
    """Return a stable fingerprint for caching.

    Streamlit can't hash scipy sparse matrices by default; this fingerprint is used
    as the cache key while the matrix itself is passed as an ignored (underscored)
    argument.
    """

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        # Should never happen in this project.
        return "no-numpy"

    h = hashlib.blake2b(digest_size=16)
    if hasattr(m, "tocsr") and hasattr(m, "indptr") and hasattr(m, "indices") and hasattr(m, "data"):
        # scipy.sparse matrix (CSR-like)
        shape = getattr(m, "shape", None)
        h.update(repr(shape).encode("utf-8"))
        for arr_name in ("indptr", "indices", "data"):
            arr = getattr(m, arr_name)
            arr_np = np.asarray(arr)
            h.update(str(arr_np.dtype).encode("utf-8"))
            h.update(arr_np.tobytes(order="C"))
        return h.hexdigest()

    # numpy array-like
    arr = np.asarray(m)
    h.update(repr(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _umap_2d(
    embeddings: "object",
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> "object":
    embeddings_fp = _fingerprint_matrix(embeddings)
    return _umap_2d_cached(embeddings, embeddings_fp, n_neighbors, min_dist, seed)


@st.cache_data(show_spinner=False)
def _umap_2d_cached(
    _embeddings: "object",
    embeddings_fp: str,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> "object":
    try:
        import umap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Semantic UMAP requires umap-learn. Install it with: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    _ = embeddings_fp  # included in cache key
    return reducer.fit_transform(_embeddings)


def _kmeans_labels(embeddings: "object", k: int, seed: int) -> "object":
    embeddings_fp = _fingerprint_matrix(embeddings)
    return _kmeans_labels_cached(embeddings, embeddings_fp, k, seed)


@st.cache_data(show_spinner=False)
def _kmeans_labels_cached(_embeddings: "object", embeddings_fp: str, k: int, seed: int) -> "object":
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Clustering requires scikit-learn. Install project deps first."
        ) from exc

    model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    _ = embeddings_fp  # included in cache key
    return model.fit_predict(_embeddings)


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _infer_split_name(path: Path) -> str:
    stem = path.stem.lower()
    if "train" in stem:
        return "train"
    if "test" in stem:
        return "test"
    return path.stem


@dataclass(frozen=True)
class LoadSpec:
    data_dir: str
    pattern: str
    sample: int
    seed: int


@st.cache_data(show_spinner=False)
def _list_parquet_files(data_dir: str, pattern: str) -> list[str]:
    p = Path(data_dir)
    files = sorted(p.glob(pattern))
    return [str(f) for f in files]


@st.cache_data(show_spinner=True)
def _load_one_parquet(path_str: str, sample: int, seed: int) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_parquet(path)
    if sample > 0 and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
    df[SPLIT_COLUMN] = _infer_split_name(path)
    return df


@st.cache_data(show_spinner=True)
def _load_dataset(spec: LoadSpec) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    files = _list_parquet_files(spec.data_dir, spec.pattern)
    per_file: dict[str, pd.DataFrame] = {}
    dfs: list[pd.DataFrame] = []

    for f in files:
        df = _load_one_parquet(f, sample=spec.sample, seed=spec.seed)
        per_file[Path(f).name] = df
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()
    return merged, per_file


def _value_counts(df: pd.DataFrame, column: str, top_n: int = 25) -> pd.DataFrame:
    s = df[column].astype("string").fillna("<NA>")
    vc = s.value_counts(dropna=False).head(top_n)
    return pd.DataFrame({column: vc.index.astype(str), "count": vc.values})


def _text_length_stats(df: pd.DataFrame, column: str) -> dict[str, int]:
    lengths = df[column].astype("string").str.len().dropna()
    if len(lengths) == 0:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}

    return {
        "min": int(lengths.min()),
        "p50": int(lengths.quantile(0.5)),
        "p95": int(lengths.quantile(0.95)),
        "max": int(lengths.max()),
    }


def _maybe_render_split_overview(df: pd.DataFrame) -> None:
    if SPLIT_COLUMN not in df.columns:
        return

    split_counts = (
        df[SPLIT_COLUMN].astype("string").fillna("<NA>").value_counts(dropna=False)
    )
    split_df = pd.DataFrame({SPLIT_COLUMN: split_counts.index, "count": split_counts.values})

    fig = px.bar(
        split_df,
        x="count",
        y=SPLIT_COLUMN,
        orientation="h",
        title=f"Rows per {SPLIT_COLUMN}",
    )
    st.plotly_chart(fig, use_container_width=True)


def _maybe_render_categorical(df: pd.DataFrame, cat_col: str, top_n: int) -> None:
    counts = _value_counts(df, cat_col, top_n=top_n)
    fig = px.bar(
        counts.sort_values("count", ascending=True),
        x="count",
        y=cat_col,
        orientation="h",
        title=f"Top {len(counts)} values of {cat_col}",
    )
    st.plotly_chart(fig, use_container_width=True)

    if SPLIT_COLUMN in df.columns:
        subset = df[[cat_col, SPLIT_COLUMN]].copy()
        subset[cat_col] = subset[cat_col].astype("string").fillna("<NA>")
        subset[SPLIT_COLUMN] = subset[SPLIT_COLUMN].astype("string").fillna("<NA>")
        top_values = set(counts[cat_col].astype("string").tolist())
        subset = subset[subset[cat_col].isin(top_values)]

        if len(subset) > 0:
            by_split = (
                subset.groupby([cat_col, SPLIT_COLUMN], dropna=False)
                .size()
                .reset_index(name="count")
            )
            fig2 = px.bar(
                by_split,
                y=cat_col,
                x="count",
                color=SPLIT_COLUMN,
                orientation="h",
                barmode="stack",
                title=f"{cat_col} by {SPLIT_COLUMN} (top values)",
            )
            st.plotly_chart(fig2, use_container_width=True)


def _maybe_render_text(df: pd.DataFrame, text_col: str, bins: int) -> None:
    lengths = df[text_col].astype("string").str.len().dropna()
    if len(lengths) == 0:
        st.info("No non-null text values to plot.")
        return

    fig = px.histogram(
        x=lengths,
        nbins=bins,
        labels={"x": "chars"},
        title=f"Text length distribution ({text_col})",
    )
    st.plotly_chart(fig, use_container_width=True)


def _maybe_render_numeric(df: pd.DataFrame, num_col: str, bins: int) -> None:
    series = pd.to_numeric(df[num_col], errors="coerce").dropna()
    if len(series) == 0:
        st.info("No numeric values to plot.")
        return

    fig = px.histogram(
        x=series,
        nbins=bins,
        labels={"x": num_col},
        title=f"Numeric distribution ({num_col})",
    )
    st.plotly_chart(fig, use_container_width=True)


def _maybe_render_custom_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str | None,
    max_points: int,
    seed: int,
) -> None:
    cols = [x_col, y_col] + ([color_col] if color_col else [])
    cols = [c for c in cols if c in df.columns]
    # Plotly Express (via narwhals) requires unique column names.
    # Users may intentionally set x==y or color==x; keep behavior but avoid
    # selecting the same column twice.
    cols = list(dict.fromkeys(cols))
    if x_col not in cols or y_col not in cols:
        st.info("Pick valid x/y columns to render a scatter plot.")
        return

    plot_df = df[cols].copy()
    if not plot_df.columns.is_unique:
        # Extremely defensive: if the upstream dataframe contains duplicate column
        # labels, pandas selection returns all of them. Plotly/narwhals requires
        # unique column names, so keep the first occurrence.
        plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()].copy()
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        st.info("No rows available after dropping null x/y values.")
        return

    if max_points > 0 and len(plot_df) > int(max_points):
        plot_df = plot_df.sample(n=int(max_points), random_state=int(seed))

    try:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=(color_col if color_col else None),
            title=(
                f"Scatter: {x_col} vs {y_col}"
                + (f" (colored by {color_col})" if color_col else "")
            ),
        )
    except Exception as exc:
        st.error(f"Could not render scatter plot: {exc}")
        return

    fig.update_traces(marker={"size": 6, "opacity": 0.75})
    st.plotly_chart(fig, use_container_width=True)


def _maybe_render_timeline(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    color_col: str | None,
    max_points: int,
    seed: int,
) -> None:
    if time_col not in df.columns:
        st.info(f"No '{time_col}' column found.")
        return

    # segment_start is often a datetime-like string (e.g. "2025-06-19 07:06:11.061 +0200").
    # Prefer datetime parsing; fall back to numeric if datetime parsing fails.
    raw_time = df[time_col]
    time_dt = pd.to_datetime(raw_time, errors="coerce", utc=True)
    is_datetime = bool(time_dt.notna().any())
    if is_datetime:
        time_s: pd.Series = time_dt.dt.tz_convert(None)
    else:
        time_s = pd.to_numeric(raw_time, errors="coerce")

    work = df.copy()
    work[time_col] = time_s
    work = work.dropna(subset=[time_col])
    if work.empty:
        st.info("No rows available after dropping null timeline values.")
        return

    # Convert to time-of-day (hours in [0, 24)).
    tod_col = "time_of_day_hours"
    if is_datetime:
        dt = pd.to_datetime(work[time_col], errors="coerce")
        work[tod_col] = (
            dt.dt.hour
            + dt.dt.minute / 60.0
            + dt.dt.second / 3600.0
            + dt.dt.microsecond / 3_600_000_000.0
        )
    else:
        # Heuristic: if values are small (<=48) treat as already-hours,
        # otherwise assume seconds and convert to hours-of-day.
        s_num = pd.to_numeric(work[time_col], errors="coerce")
        if s_num.notna().any() and float(s_num.max()) <= 48.0:
            work[tod_col] = s_num
        else:
            work[tod_col] = (s_num % 86_400) / 3_600.0

    work = work.dropna(subset=[tod_col])
    if work.empty:
        st.info("No rows available after computing time-of-day values.")
        return

    if y_col not in work.columns:
        st.info("Pick a valid y-axis column.")
        return

    cols = [tod_col, time_col, y_col] + ([color_col] if color_col else [])
    cols = [c for c in cols if c in work.columns]
    cols = list(dict.fromkeys(cols))
    plot_df = work[cols].copy()

    # Coerce y to numeric when it mostly looks numeric.
    y_series = plot_df[y_col]
    y_non_null = y_series.notna().sum()
    if pd.api.types.is_bool_dtype(y_series):
        plot_df[y_col] = y_series.astype(float)
    else:
        y_num = pd.to_numeric(y_series, errors="coerce")
        if y_non_null > 0 and (y_num.notna().sum() / float(y_non_null)) >= 0.9:
            plot_df[y_col] = y_num

    # Reduce extremely high-cardinality categorical colors.
    color_col_eff: str | None = None
    if color_col and color_col in plot_df.columns:
        cser = plot_df[color_col]
        if not pd.api.types.is_numeric_dtype(cser) and not pd.api.types.is_bool_dtype(cser):
            cstr = cser.astype("string").fillna("<NA>")
            vc = cstr.value_counts(dropna=False)
            if len(vc) > 30:
                top = set(vc.head(20).index.astype(str).tolist())
                plot_df[color_col] = cstr.astype(str).where(cstr.astype(str).isin(top), other="Other")
            else:
                plot_df[color_col] = cstr
        color_col_eff = color_col

    plot_df = plot_df.dropna(subset=[tod_col, y_col])
    if plot_df.empty:
        st.info("No rows available after dropping null x/y values.")
        return

    if max_points > 0 and len(plot_df) > int(max_points):
        plot_df = plot_df.sample(n=int(max_points), random_state=int(seed))

    try:
        fig = px.scatter(
            plot_df,
            x=tod_col,
            y=y_col,
            color=color_col_eff,
            hover_data={time_col: True} if time_col in plot_df.columns else None,
            labels={tod_col: "Hour of day"},
            title=(
                f"Time of day: {y_col}"
                + (f" (colored by {color_col_eff})" if color_col_eff else "")
            ),
        )
    except Exception as exc:
        st.error(f"Could not render timeline scatter plot: {exc}")
        return

    fig.update_traces(marker={"size": 6, "opacity": 0.6})
    fig.update_xaxes(range=[0, 24])
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="OMÉ Dataset Explorer", layout="wide")

    st.title("OMÉ dataset explorer")
    st.caption("Interactive dataset summaries & charts (Streamlit + Plotly).")

    with st.sidebar:
        st.header("Data")
        data_dir = st.text_input("Data directory", value="data")
        pattern = st.text_input("File pattern", value="*.parquet")

        st.header("Sampling")
        sample = st.number_input(
            "Sample rows per file (0 = disable)",
            min_value=0,
            max_value=2_000_000,
            value=5000,
            step=1000,
        )
        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42)

        st.header("View")
        view = st.radio("Dataset", options=["merged", "single file"], index=0)

        st.header("Columns")
        top_n = st.slider("Top N categories", min_value=5, max_value=100, value=20, step=5)
        bins = st.slider("Histogram bins", min_value=10, max_value=200, value=50, step=10)

    files = _list_parquet_files(data_dir, pattern)
    if not files:
        st.error(f"No parquet files found in '{data_dir}' (pattern: {pattern}).")
        st.info("Tip: run `python scripts/download_dataset.py` first.")
        return

    spec = LoadSpec(data_dir=data_dir, pattern=pattern, sample=int(sample), seed=int(seed))
    merged, per_file = _load_dataset(spec)

    selected_name: str
    if view == "single file":
        selected_name = st.selectbox("Choose a file", options=sorted(per_file.keys()))
        df = per_file[selected_name]
        dataset_name = selected_name
    else:
        df = merged
        dataset_name = "merged"

    if df.empty:
        st.warning("No rows loaded.")
        return

    st.subheader(f"Dataset: {dataset_name}")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", f"{len(df):,}")
    col_b.metric("Columns", f"{len(df.columns):,}")
    col_c.metric("Files", f"{len(per_file):,}")

    st.divider()

    st.markdown("#### Quick peek")
    st.dataframe(df.head(50), use_container_width=True)

    st.divider()

    # Column inference with user override
    inferred_cat = _first_existing_column(df, DEFAULT_CATEGORICAL_COLUMNS)
    inferred_text = _first_existing_column(df, DEFAULT_TEXT_COLUMNS)
    inferred_num = _first_existing_column(df, DEFAULT_NUMERIC_COLUMNS)

    all_cols = list(df.columns)

    st.markdown("#### Custom scatter")
    st.caption("Pick any two columns for x/y and an optional color column.")

    numeric_candidates = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

    # Requested defaults
    preferred_x = "duration_seconds"
    preferred_y = "channel_name"
    preferred_color = "category"

    default_scatter_x = (
        preferred_x
        if preferred_x in all_cols
        else (inferred_num if inferred_num in all_cols else (numeric_candidates[0] if numeric_candidates else None))
    )
    default_scatter_y = (
        preferred_y
        if preferred_y in all_cols
        else (inferred_cat if inferred_cat in all_cols else default_scatter_x)
    )
    default_scatter_color = (
        preferred_color
        if preferred_color in all_cols
        else (inferred_cat if inferred_cat in all_cols else (SPLIT_COLUMN if SPLIT_COLUMN in all_cols else None))
    )

    sx, sy, sc = st.columns(3)
    with sx:
        scatter_x = st.selectbox(
            "X axis",
            options=["(none)"] + all_cols,
            index=(1 + all_cols.index(default_scatter_x)) if default_scatter_x in all_cols else 0,
        )
    with sy:
        scatter_y = st.selectbox(
            "Y axis",
            options=["(none)"] + all_cols,
            index=(1 + all_cols.index(default_scatter_y)) if default_scatter_y in all_cols else 0,
        )
    with sc:
        scatter_color = st.selectbox(
            "Color",
            options=["(none)"] + all_cols,
            index=(1 + all_cols.index(default_scatter_color)) if default_scatter_color in all_cols else 0,
        )

    scatter_max_points = st.slider(
        "Max points (scatter)",
        min_value=200,
        max_value=50_000,
        value=5000,
        step=500,
        help="Downsamples for responsiveness.",
    )

    if scatter_x != "(none)" and scatter_y != "(none)":
        _maybe_render_custom_scatter(
            df,
            x_col=scatter_x,
            y_col=scatter_y,
            color_col=(None if scatter_color == "(none)" else scatter_color),
            max_points=int(scatter_max_points),
            seed=int(seed),
        )

    st.divider()

    # Timeline view
    time_col = "segment_start"
    if time_col in df.columns:
        st.markdown("#### Timeline")
        st.caption(
            "Scatter plot by time of day (derived from segment_start). Choose a y-axis and an optional color column."
        )

        default_tl_y = (
            "duration_seconds"
            if "duration_seconds" in all_cols
            else (numeric_candidates[0] if numeric_candidates else all_cols[0])
        )
        default_tl_color = "category" if "category" in all_cols else (SPLIT_COLUMN if SPLIT_COLUMN in all_cols else None)

        t1, t2, t3 = st.columns(3)
        with t1:
            tl_y = st.selectbox(
                "Y axis",
                options=["(none)"] + [c for c in all_cols if c != time_col],
                index=(1 + all_cols.index(default_tl_y)) if default_tl_y in all_cols else 0,
            )
        with t2:
            tl_color = st.selectbox(
                "Color",
                options=["(none)"] + [c for c in all_cols if c != time_col],
                index=(1 + all_cols.index(default_tl_color)) if default_tl_color in all_cols else 0,
            )
        with t3:
            tl_max_points = st.slider(
                "Max points (timeline)",
                min_value=200,
                max_value=50_000,
                value=10_000,
                step=500,
                help="Downsamples for responsiveness.",
            )

        if tl_y != "(none)":
            _maybe_render_timeline(
                df,
                time_col=time_col,
                y_col=tl_y,
                color_col=(None if tl_color == "(none)" else tl_color),
                max_points=int(tl_max_points),
                seed=int(seed),
            )
        else:
            st.info("Pick a y-axis to render the timeline scatter plot.")

        st.divider()

    themes_default = _first_existing_column(df, DEFAULT_THEMES_COLUMNS)
    if themes_default is not None:
        st.markdown("#### Themes")
        st.caption(
            "The column contains comma-separated values; this view splits them and counts each sub-theme."
        )

        with st.expander("Themes settings", expanded=False):
            themes_top_n = st.slider(
                "Number of themes to display",
                min_value=5,
                max_value=100,
                value=min(int(top_n), 100),
                step=5,
            )

        _maybe_render_themes(
            df,
            themes_col=themes_default,
            top_n=int(themes_top_n),
            sep=",",
            show_by_split=False,
            show_cooccurrence=False,
        )

        st.divider()




if __name__ == "__main__":
    main()
