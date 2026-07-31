"""Anonymous BLS API and bulk-file utilities for labor signals."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from app.venture.sources.errors import SourceParseError
from app.venture.sources.http import (
    BLS_BED_NAICS_23,
    BLS_BED_NAICS_51,
    BLS_BED_NAICS_54,
    BLS_BED_NAICS_56,
    BLS_BED_NAICS_62,
    BLS_BED_NAICS_81,
    BLS_ECI_BULK,
    BLS_JOLTS_BULK,
    BLS_TIMESERIES,
    FixedEndpoint,
    RawCapture,
    SafeHttpClient,
)
from app.venture.sources.parsing import (
    JsonObject,
    json_object,
    object_list,
    object_value,
    optional_decimal,
    optional_text,
    required_text,
)

JOLTS_TOTAL_NONFARM_JOB_OPENINGS: Final = "JTS000000000000000JOL"
ECI_PRIVATE_INDUSTRY_COMPENSATION_12_MONTH: Final = "CIU2010000000000A"
BED_ESTABLISHMENT_SURVIVAL_NOTE: Final = (
    "BLS Business Employment Dynamics survival of private-sector establishments "
    "in an opening-year cohort; this is establishment survival, not firm survival."
)
_SERIES_ID: Final = re.compile(r"^[A-Z0-9_#-]+$")
_BED_COHORT: Final = re.compile(r"^\s*Year ended:\s*March\s+(\d{4})\s*$")
_BED_ROW: Final = re.compile(
    r"^\s*March\s+(?P<year>\d{4})\s+"
    r"(?P<establishments>[\d,]+)\s+"
    r"(?P<employment>[\d,]+)\s+"
    r"(?P<since_birth>\d+(?:\.\d+)?)\s+"
    r"(?P<previous_year>_|\d+(?:\.\d+)?)\s+"
    r"(?P<average_employment>\d+(?:\.\d+)?)\s*$"
)
_MAX_ANONYMOUS_SERIES: Final = 25
_MAX_ANONYMOUS_YEARS: Final = 10
_BED_ENDPOINTS: Final = {
    "23": BLS_BED_NAICS_23,
    "51": BLS_BED_NAICS_51,
    "54": BLS_BED_NAICS_54,
    "56": BLS_BED_NAICS_56,
    "62": BLS_BED_NAICS_62,
    "81": BLS_BED_NAICS_81,
}


@dataclass(frozen=True, slots=True)
class BlsFootnote:
    """A BLS status or qualification attached to an observation."""

    code: str | None
    text: str | None


@dataclass(frozen=True, slots=True)
class BlsSeriesPoint:
    """One BLS series observation from the API or a bulk file."""

    series_id: str
    year: int
    period: str
    period_name: str | None
    value: Decimal | None
    latest: bool
    footnotes: tuple[BlsFootnote, ...]


@dataclass(frozen=True, slots=True)
class BlsSeries:
    """One requested BLS series and its returned observations."""

    series_id: str
    points: tuple[BlsSeriesPoint, ...]


@dataclass(frozen=True, slots=True)
class BlsSeriesResult:
    """Frozen API request/response and normalized series."""

    capture: RawCapture
    series: tuple[BlsSeries, ...]
    messages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlsBulkResult:
    """Frozen BLS bulk response and selected normalized rows."""

    capture: RawCapture
    points: tuple[BlsSeriesPoint, ...]


@dataclass(frozen=True, slots=True)
class BlsBedEstablishmentSurvival:
    """One mature BED opening cohort's establishment-survival percentages."""

    naics_code: str
    cohort_year: int
    one_year_establishment_survival_percent: Decimal
    five_year_establishment_survival_percent: Decimal
    measurement_note: str = BED_ESTABLISHMENT_SURVIVAL_NOTE


@dataclass(frozen=True, slots=True)
class BlsBedSurvivalResult:
    """Frozen official BED table plus complete one/five-year cohort rates."""

    capture: RawCapture
    naics_code: str
    cohorts: tuple[BlsBedEstablishmentSurvival, ...]


class BlsSource:
    """Fetch anonymous official BLS time series or fixed bulk files."""

    def __init__(self, client: SafeHttpClient | None = None) -> None:
        self._client = client or SafeHttpClient()

    def fetch_series(
        self,
        series_ids: Collection[str],
        *,
        start_year: int,
        end_year: int,
    ) -> BlsSeriesResult:
        """Fetch up to the anonymous API's series/year limits without a key."""
        normalized_ids = self._validate_series_request(series_ids, start_year, end_year)
        query: dict[str, object] = {
            "endyear": str(end_year),
            "seriesid": list(normalized_ids),
            "startyear": str(start_year),
        }
        capture = self._client.request(BLS_TIMESERIES, json_body=query)
        payload = json_object(capture.raw_bytes, source=BLS_TIMESERIES.name)
        if payload.get("status") != "REQUEST_SUCCEEDED":
            raise SourceParseError("BLS request did not report REQUEST_SUCCEEDED")
        messages = self._parse_messages(payload.get("message", []))
        results = object_value(payload.get("Results"), field="Results")
        parsed_series = tuple(
            self._parse_series(row)
            for row in object_list(results.get("series"), field="Results.series")
        )
        returned_ids = {series.series_id for series in parsed_series}
        unexpected = returned_ids.difference(normalized_ids)
        if unexpected:
            raise SourceParseError(f"BLS returned unrequested series: {sorted(unexpected)!r}")
        return BlsSeriesResult(capture=capture, series=parsed_series, messages=messages)

    def fetch_job_openings(self, *, start_year: int, end_year: int) -> BlsSeriesResult:
        """Fetch total-nonfarm JOLTS job openings, reported in thousands."""
        return self.fetch_series(
            [JOLTS_TOTAL_NONFARM_JOB_OPENINGS],
            start_year=start_year,
            end_year=end_year,
        )

    def fetch_labor_cost_growth(self, *, start_year: int, end_year: int) -> BlsSeriesResult:
        """Fetch private-industry total-compensation 12-month percent change."""
        return self.fetch_series(
            [ECI_PRIVATE_INDUSTRY_COMPENSATION_12_MONTH],
            start_year=start_year,
            end_year=end_year,
        )

    def fetch_establishment_survival(self, naics_code: str) -> BlsBedSurvivalResult:
        """Fetch one approved two-digit NAICS BED establishment-survival table."""
        normalized = naics_code.strip()
        try:
            endpoint = _BED_ENDPOINTS[normalized]
        except KeyError as exc:
            supported = ", ".join(sorted(_BED_ENDPOINTS))
            raise ValueError(f"NAICS code must be one of: {supported}") from exc
        capture = self._client.request(endpoint)
        cohorts = parse_bed_establishment_survival(
            capture.raw_bytes,
            naics_code=normalized,
        )
        return BlsBedSurvivalResult(
            capture=capture,
            naics_code=normalized,
            cohorts=cohorts,
        )

    def fetch_jolts_bulk(
        self,
        *,
        series_ids: Collection[str] = (JOLTS_TOTAL_NONFARM_JOB_OPENINGS,),
    ) -> BlsBulkResult:
        """Fetch the fixed JOLTS bulk table and retain selected series."""
        return self._fetch_bulk(BLS_JOLTS_BULK, series_ids=series_ids)

    def fetch_eci_bulk(
        self,
        *,
        series_ids: Collection[str] = (ECI_PRIVATE_INDUSTRY_COMPENSATION_12_MONTH,),
    ) -> BlsBulkResult:
        """Fetch the fixed Employment Cost Index bulk table and retain selected series."""
        return self._fetch_bulk(BLS_ECI_BULK, series_ids=series_ids)

    def _fetch_bulk(
        self,
        endpoint: FixedEndpoint,
        *,
        series_ids: Collection[str],
    ) -> BlsBulkResult:
        normalized = self._normalize_series_ids(series_ids)
        capture = self._client.request(endpoint)
        points = parse_bls_bulk(capture.raw_bytes, series_ids=normalized)
        return BlsBulkResult(capture=capture, points=points)

    @classmethod
    def _validate_series_request(
        cls,
        series_ids: Collection[str],
        start_year: int,
        end_year: int,
    ) -> tuple[str, ...]:
        normalized = cls._normalize_series_ids(series_ids)
        if len(normalized) > _MAX_ANONYMOUS_SERIES:
            raise ValueError(f"anonymous BLS requests allow at most {_MAX_ANONYMOUS_SERIES} series")
        if start_year < 1900 or end_year > 9999 or end_year < start_year:
            raise ValueError("start_year and end_year must form a valid four-digit range")
        if end_year - start_year + 1 > _MAX_ANONYMOUS_YEARS:
            raise ValueError(f"anonymous BLS requests allow at most {_MAX_ANONYMOUS_YEARS} years")
        return normalized

    @staticmethod
    def _normalize_series_ids(series_ids: Collection[str]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(series_id.strip().upper() for series_id in series_ids))
        if not normalized:
            raise ValueError("at least one BLS series ID is required")
        if any(not _SERIES_ID.fullmatch(series_id) for series_id in normalized):
            raise ValueError(
                "BLS series IDs may contain only uppercase letters, digits, _, #, or -"
            )
        return normalized

    @classmethod
    def _parse_series(cls, row: JsonObject) -> BlsSeries:
        series_id = required_text(row.get("seriesID"), field="Results.series.seriesID")
        points = tuple(
            cls._parse_api_point(item, series_id=series_id)
            for item in object_list(row.get("data"), field=f"Results.series[{series_id}].data")
        )
        return BlsSeries(series_id=series_id, points=points)

    @classmethod
    def _parse_api_point(cls, row: JsonObject, *, series_id: str) -> BlsSeriesPoint:
        year_text = required_text(row.get("year"), field="BLS point year")
        if not year_text.isdigit() or len(year_text) != 4:
            raise SourceParseError("BLS point year must be four digits")
        latest_value = row.get("latest", False)
        if latest_value not in (True, False, "true", "false"):
            raise SourceParseError("BLS point latest must be boolean-like")
        footnotes = tuple(
            BlsFootnote(
                code=optional_text(note.get("code"), field="BLS footnote code"),
                text=optional_text(note.get("text"), field="BLS footnote text"),
            )
            for note in object_list(row.get("footnotes", []), field="BLS point footnotes")
        )
        return BlsSeriesPoint(
            series_id=series_id,
            year=int(year_text),
            period=required_text(row.get("period"), field="BLS point period"),
            period_name=optional_text(row.get("periodName"), field="BLS point periodName"),
            value=optional_decimal(row.get("value"), field="BLS point value"),
            latest=latest_value in (True, "true"),
            footnotes=footnotes,
        )

    @staticmethod
    def _parse_messages(value: object) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise SourceParseError("BLS message must be a list of strings")
        return tuple(value)


def parse_bls_bulk(
    raw_bytes: bytes,
    *,
    series_ids: Collection[str] | None = None,
) -> tuple[BlsSeriesPoint, ...]:
    """Parse an official tab-separated BLS bulk data table.

    Bulk rows have no period name or expanded footnote text; the source's
    footnote-code column is retained as a code-only footnote.
    """
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceParseError("BLS bulk response is not UTF-8 text") from exc
    lines = text.splitlines()
    if not lines:
        raise SourceParseError("BLS bulk response is empty")
    headers = tuple(part.strip() for part in lines[0].split("\t"))
    required = {"series_id", "year", "period", "value", "footnote_codes"}
    if not required.issubset(headers):
        raise SourceParseError("BLS bulk response is missing required columns")
    positions = {header: index for index, header in enumerate(headers)}
    selected = set(BlsSource._normalize_series_ids(series_ids)) if series_ids is not None else None
    points: list[BlsSeriesPoint] = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split("\t")]
        if len(cells) < len(headers):
            raise SourceParseError(f"BLS bulk row {line_number} has too few columns")
        series_id = cells[positions["series_id"]]
        if selected is not None and series_id not in selected:
            continue
        if not _SERIES_ID.fullmatch(series_id):
            raise SourceParseError(f"BLS bulk row {line_number} has an invalid series ID")
        year_text = cells[positions["year"]]
        if not year_text.isdigit() or len(year_text) != 4:
            raise SourceParseError(f"BLS bulk row {line_number} has an invalid year")
        footnote_code = cells[positions["footnote_codes"]] or None
        points.append(
            BlsSeriesPoint(
                series_id=series_id,
                year=int(year_text),
                period=cells[positions["period"]],
                period_name=None,
                value=optional_decimal(
                    cells[positions["value"]],
                    field=f"BLS bulk row {line_number} value",
                ),
                latest=False,
                footnotes=(
                    (BlsFootnote(code=footnote_code, text=None),)
                    if footnote_code is not None
                    else ()
                ),
            )
        )
    return tuple(points)


def parse_bed_establishment_survival(
    raw_bytes: bytes,
    *,
    naics_code: str,
) -> tuple[BlsBedEstablishmentSurvival, ...]:
    """Extract complete one- and five-year establishment survival percentages.

    Table 7's third numeric column is ``Survival Rates Since Birth``.  The
    previous-year survival column is intentionally ignored.  Recent cohorts
    without five years of follow-up are omitted; a mature cohort missing either
    required observation is treated as malformed rather than silently imputed.
    """
    if naics_code not in _BED_ENDPOINTS:
        supported = ", ".join(sorted(_BED_ENDPOINTS))
        raise ValueError(f"NAICS code must be one of: {supported}")
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceParseError("BLS BED table is not UTF-8 text") from exc
    if "Table 7. Survival of private sector establishments by opening year" not in text:
        raise SourceParseError("BLS BED response is not establishment-survival Table 7")

    observations: dict[int, dict[int, Decimal]] = {}
    current_cohort: int | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        cohort_match = _BED_COHORT.match(line)
        if cohort_match is not None:
            current_cohort = int(cohort_match.group(1))
            if current_cohort in observations:
                raise SourceParseError(
                    f"BLS BED table repeats cohort {current_cohort} at line {line_number}"
                )
            observations[current_cohort] = {}
            continue
        row_match = _BED_ROW.match(line)
        if row_match is None:
            if current_cohort is not None and line.lstrip().startswith("March "):
                raise SourceParseError(f"BLS BED table has a malformed row at line {line_number}")
            continue
        if current_cohort is None:
            continue
        observation_year = int(row_match.group("year"))
        if observation_year < current_cohort:
            raise SourceParseError(f"BLS BED observation precedes cohort at line {line_number}")
        cohort_observations = observations[current_cohort]
        if observation_year in cohort_observations:
            raise SourceParseError(
                f"BLS BED table repeats observation year {observation_year} "
                f"for cohort {current_cohort}"
            )
        rate = Decimal(row_match.group("since_birth"))
        if not Decimal(0) <= rate <= Decimal(100):
            raise SourceParseError(f"BLS BED survival rate is outside 0-100 at line {line_number}")
        cohort_observations[observation_year] = rate

    if not observations:
        raise SourceParseError("BLS BED table contains no opening-year cohorts")
    latest_observation = max(
        observation_year
        for cohort_observations in observations.values()
        for observation_year in cohort_observations
    )
    complete: list[BlsBedEstablishmentSurvival] = []
    for cohort_year, cohort_observations in sorted(observations.items()):
        if cohort_year + 5 > latest_observation:
            continue
        one_year = cohort_observations.get(cohort_year + 1)
        five_year = cohort_observations.get(cohort_year + 5)
        if one_year is None or five_year is None:
            raise SourceParseError(
                f"BLS BED mature cohort {cohort_year} is missing one- or five-year survival"
            )
        complete.append(
            BlsBedEstablishmentSurvival(
                naics_code=naics_code,
                cohort_year=cohort_year,
                one_year_establishment_survival_percent=one_year,
                five_year_establishment_survival_percent=five_year,
            )
        )
    if not complete:
        raise SourceParseError("BLS BED table contains no complete five-year cohorts")
    return tuple(complete)


__all__ = [
    "BED_ESTABLISHMENT_SURVIVAL_NOTE",
    "ECI_PRIVATE_INDUSTRY_COMPENSATION_12_MONTH",
    "JOLTS_TOTAL_NONFARM_JOB_OPENINGS",
    "BlsBedEstablishmentSurvival",
    "BlsBedSurvivalResult",
    "BlsBulkResult",
    "BlsFootnote",
    "BlsSeries",
    "BlsSeriesPoint",
    "BlsSeriesResult",
    "BlsSource",
    "parse_bed_establishment_survival",
    "parse_bls_bulk",
]
