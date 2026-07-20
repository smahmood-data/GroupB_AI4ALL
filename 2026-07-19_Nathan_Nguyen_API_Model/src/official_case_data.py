"""Official dengue case ingestion and revision tracking.

The operational forecasting system needs two different kinds of case data:

* provisional recent counts, which may improve current-week predictions; and
* sufficiently stable historical counts, which may become training labels.

Those uses must not be conflated.  Public-health surveillance data are revised
as laboratory results and investigations arrive, so this module records every
changed weekly count before replacing the current normalized table.

Puerto Rico
------------
The Puerto Rico Department of Health exposes a structured catalog API.  The
``arbovirus_cases_summary`` catalog contains daily Puerto Rico-wide PCR and IgM
dengue counts.  The API first returns a short-lived signed download URL; this
module resolves and consumes it without ever writing the signed URL to Git.

Peru / Iquitos
--------------
Peru's official open-data catalog describes a district-level dengue CSV.  At
the time this pipeline was implemented, the catalog metadata was automatable
but the 100+ MB file host rejected non-browser downloads and the published file
ended in 2024.  We therefore support a deliberate manual import of that
official file.  Only a compact Iquitos weekly aggregate is retained afterward.
"""

from __future__ import annotations

import gzip
import io
import json
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import brotli


PUERTO_RICO_API_BASE = "https://biostatistics.salud.pr.gov"
PUERTO_RICO_SOURCE_PAGE = "https://datos.salud.pr.gov"
PERU_PACKAGE_ID = "2c3b2955-6252-4b9d-abd3-caeb61ccda39"
PERU_PACKAGE_URL = (
    "https://www.datosabiertos.gob.pe/api/3/action/package_show?"
    + urlencode({"id": PERU_PACKAGE_ID})
)
PERU_SOURCE_PAGE = (
    "https://www.datosabiertos.gob.pe/dataset/"
    "vigilancia-epidemiol%C3%B3gica-de-dengue"
)

CASE_TABLE_COLUMNS = [
    "geography",
    "week_start_date",
    "total_cases",
    "pcr_cases",
    "igm_cases",
    "hospitalized_cases",
    "complete_week",
    "source_file_id",
    "source_publication_date",
    "retrieved_at_utc",
    "source_page",
]


@dataclass(frozen=True)
class PuertoRicoSnapshot:
    """Normalized PR weekly cases plus source metadata for auditability."""

    weekly_cases: pd.DataFrame
    source_file_id: str
    publication_date: str
    retrieved_at_utc: str


def _request_bytes(
    url: str,
    timeout_seconds: int = 90,
    source_name: str | None = None,
) -> tuple[bytes, dict[str, str]]:
    """Download bytes with explicit headers and concise error context."""

    request = Request(
        url,
        headers={
            "User-Agent": "dengue-forecasting-model/1.0",
            "Accept": "application/json,text/csv,*/*",
            "Accept-Encoding": "gzip",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            return response.read(), headers
    except (HTTPError, URLError, TimeoutError) as exc:
        safe_source = source_name or url
        raise RuntimeError(
            f"Unable to download official case data from {safe_source}: {exc}"
        ) from exc


def _decode_payload(payload: bytes, headers: dict[str, str]) -> bytes:
    """Decompress official files whether gzip/Brotli is declared or detectable."""

    encoding = headers.get("content-encoding", "").lower()
    if encoding == "br":
        return brotli.decompress(payload)
    is_gzip = encoding == "gzip"
    has_gzip_magic = payload[:2] == b"\x1f\x8b"
    if is_gzip or has_gzip_magic:
        return gzip.decompress(payload)
    return payload


def fetch_json(url: str, source_name: str | None = None) -> Any:
    """Fetch and decode a JSON document from an official endpoint."""

    payload, headers = _request_bytes(url, source_name=source_name)
    decoded = _decode_payload(payload, headers)
    try:
        return json.loads(decoded.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        safe_source = source_name or url
        raise ValueError(f"Official endpoint returned invalid JSON: {safe_source}") from exc


def normalize_puerto_rico_daily_cases(
    records: list[dict[str, Any]],
    source_file_id: str,
    publication_date: str,
    retrieved_at_utc: str,
) -> pd.DataFrame:
    """Aggregate PR daily PCR + IgM counts into complete Monday-start weeks.

    The structured source contains one row per diagnostic date.  A missing date
    is not assumed to mean zero.  Instead, its week is marked incomplete and is
    excluded from the normalized table so partial totals cannot become labels.
    """

    if not records:
        raise ValueError("Puerto Rico arbovirus source returned no records")

    frame = pd.DataFrame(records)
    required = {
        "diagnosticDate",
        "totalCasesPcrCount",
        "totalCasesIgMCount",
        "totalHospitalizedCount",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Puerto Rico source is missing required fields: {sorted(missing)}")

    frame["diagnostic_date"] = pd.to_datetime(
        frame["diagnosticDate"], errors="raise"
    ).dt.normalize()
    if frame["diagnostic_date"].duplicated().any():
        raise ValueError("Puerto Rico source contains duplicate diagnostic dates")

    frame["pcr_cases"] = pd.to_numeric(frame["totalCasesPcrCount"], errors="raise")
    frame["igm_cases"] = pd.to_numeric(frame["totalCasesIgMCount"], errors="raise")
    frame["hospitalized_cases"] = pd.to_numeric(
        frame["totalHospitalizedCount"], errors="raise"
    )
    if (frame[["pcr_cases", "igm_cases", "hospitalized_cases"]] < 0).any().any():
        raise ValueError("Puerto Rico case counts cannot be negative")

    # Reindexing reveals missing calendar dates.  Values remain NaN, allowing
    # the weekly completeness rule below to reject the affected week.
    calendar = pd.date_range(
        frame["diagnostic_date"].min(),
        frame["diagnostic_date"].max(),
        freq="D",
    )
    daily = frame.set_index("diagnostic_date")[
        ["pcr_cases", "igm_cases", "hospitalized_cases"]
    ].reindex(calendar)
    daily.index.name = "diagnostic_date"
    daily["week_start_date"] = daily.index - pd.to_timedelta(daily.index.weekday, unit="D")

    grouped = daily.groupby("week_start_date", as_index=False).agg(
        pcr_cases=("pcr_cases", "sum"),
        igm_cases=("igm_cases", "sum"),
        hospitalized_cases=("hospitalized_cases", "sum"),
        pcr_days=("pcr_cases", "count"),
        igm_days=("igm_cases", "count"),
        hospitalized_days=("hospitalized_cases", "count"),
    )
    grouped["complete_week"] = (
        (grouped["pcr_days"] == 7)
        & (grouped["igm_days"] == 7)
        & (grouped["hospitalized_days"] == 7)
    )
    grouped = grouped[grouped["complete_week"]].copy()
    grouped["total_cases"] = grouped["pcr_cases"] + grouped["igm_cases"]
    grouped["geography"] = "pr"
    grouped["source_file_id"] = source_file_id
    grouped["source_publication_date"] = publication_date
    grouped["retrieved_at_utc"] = retrieved_at_utc
    grouped["source_page"] = PUERTO_RICO_SOURCE_PAGE

    # Integer case columns make diffs and revision logs easier to inspect.
    for column in ["pcr_cases", "igm_cases", "hospitalized_cases", "total_cases"]:
        grouped[column] = grouped[column].round().astype(int)
    grouped["week_start_date"] = pd.to_datetime(grouped["week_start_date"]).dt.normalize()
    return grouped[CASE_TABLE_COLUMNS].sort_values("week_start_date").reset_index(drop=True)


def fetch_puerto_rico_snapshot(catalog_id: str) -> PuertoRicoSnapshot:
    """Resolve the latest official catalog file and return normalized weeks."""

    catalog_url = f"{PUERTO_RICO_API_BASE}/catalogs/{catalog_id}"
    catalog = fetch_json(catalog_url)
    last_file = catalog.get("lastFile") or {}
    file_id = str(last_file.get("id", ""))
    publication_date = str(last_file.get("publicationDate", ""))
    if not file_id or not publication_date:
        raise ValueError("Puerto Rico catalog metadata has no published file")

    signed = fetch_json(f"{PUERTO_RICO_API_BASE}/catalogs/{catalog_id}/last-file-url")
    signed_url = signed.get("url")
    if not signed_url:
        raise ValueError("Puerto Rico API did not return a signed file URL")

    # The URL contains a short-lived access signature.  A safe source label
    # prevents that transient credential from appearing in error logs.
    records = fetch_json(
        str(signed_url), source_name="Puerto Rico signed arbovirus catalog file"
    )
    if not isinstance(records, list):
        raise ValueError("Puerto Rico arbovirus file is not a JSON record list")

    retrieved_at = datetime.now(timezone.utc).isoformat()
    weekly = normalize_puerto_rico_daily_cases(
        records,
        source_file_id=file_id,
        publication_date=publication_date,
        retrieved_at_utc=retrieved_at,
    )
    return PuertoRicoSnapshot(
        weekly_cases=weekly,
        source_file_id=file_id,
        publication_date=publication_date,
        retrieved_at_utc=retrieved_at,
    )


def _read_case_table(path: Path) -> pd.DataFrame:
    """Read a normalized weekly case table with consistent date handling."""

    if not path.exists():
        return pd.DataFrame(columns=CASE_TABLE_COLUMNS)
    frame = pd.read_csv(path, parse_dates=["week_start_date"])
    # Before schema v2 the canonical table did not retain hospitalizations.
    # Reindexing lets one fresh official ingest migrate that file without a
    # destructive manual conversion; comparison logic will then replace it.
    frame = frame.reindex(columns=CASE_TABLE_COLUMNS)
    return frame.sort_values(["geography", "week_start_date"]).reset_index(drop=True)


def merge_case_snapshot(
    new_cases: pd.DataFrame,
    current_path: Path,
    revisions_path: Path,
) -> pd.DataFrame:
    """Replace the canonical table and append only new or revised week events.

    The official PR file is a complete historical series.  A sudden large row
    loss is treated as a source failure rather than a valid deletion.  Small
    revisions are expected and recorded with old/new values before replacement.
    """

    required = set(CASE_TABLE_COLUMNS)
    missing = required.difference(new_cases.columns)
    if missing:
        raise ValueError(f"Normalized case snapshot is missing: {sorted(missing)}")
    if new_cases.duplicated(["geography", "week_start_date"]).any():
        raise ValueError("Normalized case snapshot has duplicate geography/week rows")

    current = _read_case_table(current_path)
    if len(current) and len(new_cases) < 0.90 * len(current):
        raise ValueError(
            "Official snapshot lost more than 10% of existing weekly rows; refusing replacement"
        )

    old_values = current.set_index(["geography", "week_start_date"])[
        ["total_cases", "hospitalized_cases"]
    ].to_dict(orient="index")
    revision_rows: list[dict[str, Any]] = []
    for _, row in new_cases.sort_values(["geography", "week_start_date"]).iterrows():
        key = (row["geography"], pd.Timestamp(row["week_start_date"]))
        old_value = old_values.get(key)
        new_total = int(row["total_cases"])
        new_hospitalized = int(row["hospitalized_cases"])
        old_total = None if old_value is None else old_value["total_cases"]
        old_hospitalized = (
            None if old_value is None else old_value["hospitalized_cases"]
        )
        total_changed = old_total is None or int(old_total) != new_total
        # A NaN old hospitalization means the file is undergoing the one-time
        # schema migration.  Backfilling that previously untracked field should
        # not create hundreds of artificial revision events.
        hospitalization_changed = (
            old_value is not None
            and pd.notna(old_hospitalized)
            and int(old_hospitalized) != new_hospitalized
        )
        if total_changed or hospitalization_changed:
            revision_rows.append(
                {
                    "geography": row["geography"],
                    "week_start_date": pd.Timestamp(row["week_start_date"]).date().isoformat(),
                    "old_total_cases": (
                        None if old_total is None else int(old_total)
                    ),
                    "new_total_cases": new_total,
                    "old_hospitalized_cases": (
                        None
                        if old_hospitalized is None or pd.isna(old_hospitalized)
                        else int(old_hospitalized)
                    ),
                    "new_hospitalized_cases": new_hospitalized,
                    "change_type": "new" if old_value is None else "revision",
                    "observed_at_utc": row["retrieved_at_utc"],
                    "source_file_id": row["source_file_id"],
                    "source_publication_date": row["source_publication_date"],
                }
            )

    # Re-fetching the exact same published source file should not rewrite every
    # historical row merely to update ``retrieved_at_utc``. The source-status
    # artifact records each API check separately, while this canonical table
    # changes only when the official file or its weekly contents change. This
    # keeps scheduled bot commits small and reviewable.
    if len(current) and not revision_rows and len(current) == len(new_cases):
        comparison_columns = [
            column for column in CASE_TABLE_COLUMNS if column != "retrieved_at_utc"
        ]
        current_values = current[comparison_columns].reset_index(drop=True)
        new_values = (
            new_cases.sort_values(["geography", "week_start_date"])[comparison_columns]
            .reset_index(drop=True)
        )
        if current_values.equals(new_values):
            return pd.DataFrame(revision_rows)

    current_path.parent.mkdir(parents=True, exist_ok=True)
    new_cases.sort_values(["geography", "week_start_date"]).to_csv(current_path, index=False)

    if revision_rows:
        revisions_path.parent.mkdir(parents=True, exist_ok=True)
        additions = pd.DataFrame(revision_rows)
        if revisions_path.exists():
            previous = pd.read_csv(revisions_path)
            additions = pd.concat([previous, additions], ignore_index=True)
        additions.to_csv(revisions_path, index=False)
    return pd.DataFrame(revision_rows)


def _normalize_text(value: Any) -> str:
    """Uppercase and strip accents for stable Spanish location matching."""

    normalized = unicodedata.normalize("NFKD", str(value))
    without_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(without_accents.upper().strip().split())


def _read_large_peru_csv(path: Path) -> pd.DataFrame:
    """Read the manually supplied official file across common encodings."""

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Unable to parse Peru CSV; " + " | ".join(errors))


def normalize_peru_iquitos_csv(path: Path, retrieved_at_utc: str | None = None) -> pd.DataFrame:
    """Convert the official Peru case-level CSV into Iquitos weekly counts.

    Confirmed and probable dengue rows are counted.  Suspected rows are kept out
    so the target reflects the more stable classifications described in the
    source metadata.  The raw 100+ MB file is never copied into the repository.
    """

    if not path.exists():
        raise FileNotFoundError(f"Peru official CSV does not exist: {path}")
    frame = _read_large_peru_csv(path)
    frame.columns = [_normalize_text(column).lower() for column in frame.columns]

    aliases = {
        "ano": next((column for column in frame.columns if column in {"ano", "año"}), None),
        "semana": next((column for column in frame.columns if column == "semana"), None),
        "departamento": next((column for column in frame.columns if column == "departamento"), None),
        "provincia": next((column for column in frame.columns if column == "provincia"), None),
        "distrito": next((column for column in frame.columns if column == "distrito"), None),
        "tipo_dx": next((column for column in frame.columns if column in {"tipo_dx", "tipo dx"}), None),
    }
    missing = [name for name, column in aliases.items() if column is None]
    if missing:
        raise ValueError(f"Peru official CSV is missing expected fields: {missing}")

    for name in ["departamento", "provincia", "distrito", "tipo_dx"]:
        frame[name] = frame[aliases[name]].map(_normalize_text)
    frame["year"] = pd.to_numeric(frame[aliases["ano"]], errors="coerce")
    frame["weekofyear"] = pd.to_numeric(frame[aliases["semana"]], errors="coerce")

    iquitos = frame[
        (frame["departamento"] == "LORETO")
        & (frame["provincia"] == "MAYNAS")
        & (frame["distrito"] == "IQUITOS")
        & (frame["tipo_dx"].isin(["C", "P"]))
        & frame["year"].notna()
        & frame["weekofyear"].notna()
    ].copy()
    if iquitos.empty:
        raise ValueError("No confirmed/probable Iquitos, Maynas, Loreto rows were found")

    observed = (
        iquitos.groupby(["year", "weekofyear"], as_index=False)
        .size()
        .rename(columns={"size": "total_cases"})
    )

    def week_start(row: pd.Series) -> pd.Timestamp:
        year = int(row["year"])
        week = int(row["weekofyear"])
        try:
            return pd.Timestamp(date.fromisocalendar(year, week, 1))
        except ValueError as exc:
            raise ValueError(f"Invalid Peru epidemiological year/week: {year}/{week}") from exc

    observed["week_start_date"] = observed.apply(week_start, axis=1)

    # This is a case-level source, so an absent year/week inside the published
    # time span means no matching confirmed/probable Iquitos records.  Insert
    # those zero weeks explicitly; otherwise lags would incorrectly jump across
    # time and a long quiet period would disappear from model training.
    calendar = pd.DataFrame(
        {
            "week_start_date": pd.date_range(
                observed["week_start_date"].min(),
                observed["week_start_date"].max(),
                freq="7D",
            )
        }
    )
    grouped = calendar.merge(
        observed[["week_start_date", "total_cases"]],
        on="week_start_date",
        how="left",
        validate="one_to_one",
    )
    grouped["total_cases"] = grouped["total_cases"].fillna(0).astype(int)
    retrieved = retrieved_at_utc or datetime.now(timezone.utc).isoformat()
    grouped["geography"] = "iq"
    grouped["pcr_cases"] = np.nan
    grouped["igm_cases"] = np.nan
    grouped["complete_week"] = True
    grouped["source_file_id"] = path.name
    grouped["source_publication_date"] = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
    grouped["retrieved_at_utc"] = retrieved
    grouped["source_page"] = PERU_SOURCE_PAGE
    return grouped[CASE_TABLE_COLUMNS].sort_values("week_start_date").reset_index(drop=True)


def fetch_peru_catalog_status() -> dict[str, Any]:
    """Return official Peru metadata without attempting the blocked large file."""

    payload = fetch_json(PERU_PACKAGE_URL)
    if not payload.get("success"):
        raise ValueError("Peru CKAN metadata endpoint reported failure")
    result = payload.get("result")
    if isinstance(result, list):
        result = result[0] if result else {}
    resources = result.get("resources", []) if isinstance(result, dict) else []
    resource = resources[0] if resources else {}
    return {
        "source": "Peru CDC/MINSA open-data catalog",
        "dataset_id": PERU_PACKAGE_ID,
        "metadata_modified": result.get("metadata_modified") if isinstance(result, dict) else None,
        "resource_name": resource.get("name"),
        "resource_last_modified": resource.get("last_modified"),
        "resource_size": resource.get("size"),
        "automation_status": "metadata_only_manual_csv_required",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": PERU_SOURCE_PAGE,
    }


def write_source_status(path: Path, status: dict[str, Any]) -> None:
    """Write deterministic, human-readable source health metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
