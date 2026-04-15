"""
Raw-data manifest: documents the provenance of every file in ``data/raw/``
and verifies that what's checked into git still matches the version the
analysis was built against.

The six source files are committed to the repo (they're small — 7.5 MB
total) so a replicator does not need network access to reproduce the
pipeline. This script exists so that:

1. A future maintainer who re-downloads fresh vintages can compare them
   against the pinned checksums and see whether results might drift.
2. Anyone cloning the repo with ``--sparse`` or ``--depth=1`` can confirm
   the raw files arrived intact.
3. The public URLs and dataset versions are recorded in one canonical
   place instead of living only in commit history.

Usage
-----
    python scripts/download_raw_data.py           # verify checksums
    python scripts/download_raw_data.py --show    # print the manifest
    python scripts/download_raw_data.py --fetch   # download any missing
                                                  # files (uses the URLs
                                                  # listed in the manifest)

The script intentionally does NOT overwrite existing files on ``--fetch``
— if a file is present and its checksum mismatches, it flags the drift
and exits non-zero so the replicator can decide what to do manually.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class RawFile:
    """One raw data file with its provenance and pinned SHA-256."""

    filename: str
    description: str
    source_url: str | None
    sha256: str
    size_bytes: int


# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------
# Checksums pinned from the copies committed to the repo at branch creation
# time. Source URLs point to the public landing page for each dataset; the
# exact download link format is not guaranteed stable, so ``--fetch`` falls
# back to printing the landing page when direct download is not possible.
MANIFEST: tuple[RawFile, ...] = (
    RawFile(
        filename="cpds_raw.xlsx",
        description="Comparative Political Data Set (Armingeon et al.) — social policy variables",
        source_url="https://www.cpds-data.org/",
        sha256="480b9dbb59cc3ddb02bc2ceaebf865e6114eb40c9c16b6d3fbb461e70ad7a74b",
        size_bytes=3_441_331,
    ),
    RawFile(
        filename="KOF_index_raw.xlsx",
        description="KOF Globalisation Index (ETH Zürich) — economic / social / political globalisation sub-indices",
        source_url="https://kof.ethz.ch/en/forecasts-and-indicators/indicators/kof-globalisation-index.html",
        sha256="75d361d17df3055758aae2838bb8d980c12346ca0aaa2b692091305f8fd32a3e",
        size_bytes=4_060_531,
    ),
    RawFile(
        filename="GDP_per_capita.xlsx",
        description="World Bank — GDP per capita (constant 2015 US$), indicator NY.GDP.PCAP.KD",
        source_url="https://data.worldbank.org/indicator/NY.GDP.PCAP.KD",
        sha256="3a0ce17253eaed0be1d017588270da2646ada43a1038437a15da81e918bf89ea",
        size_bytes=102_201,
    ),
    RawFile(
        filename="Inflation_cpi.xlsx",
        description="World Bank — Inflation, consumer prices (annual %), indicator FP.CPI.TOTL.ZG",
        source_url="https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG",
        sha256="f8605d7cac8fd03ac19562b209957f37346a70dcc8fc9d5d7678fe55cce73682",
        size_bytes=52_193,
    ),
    RawFile(
        filename="Population_raw.xlsx",
        description="World Bank — Total population, indicator SP.POP.TOTL",
        source_url="https://data.worldbank.org/indicator/SP.POP.TOTL",
        sha256="664828b097a2b29da03eedb5029162381542a3c05cac05135828d0e33e340d07",
        size_bytes=15_402,
    ),
    RawFile(
        filename="Dependency_ratio.xlsx",
        description="World Bank — Age dependency ratio (% working-age population), indicator SP.POP.DPND",
        source_url="https://data.worldbank.org/indicator/SP.POP.DPND",
        sha256="4cde070cc8b6566efb781f19b3d9c704c326aa453c4a45ed860a235d3f8660b9",
        size_bytes=27_246,
    ),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(strict: bool = True) -> int:
    """Check every manifest entry against what's on disk.

    Returns 0 when all checksums match, 1 otherwise. Missing files are a
    hard error when ``strict=True`` (the default for CI use).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    for entry in MANIFEST:
        path = RAW_DIR / entry.filename
        if not path.exists():
            msg = f"MISSING: {entry.filename}"
            print(f"  {msg}")
            if strict:
                problems.append(msg)
            continue

        actual = _sha256(path)
        if actual != entry.sha256:
            problems.append(
                f"CHECKSUM MISMATCH: {entry.filename}\n"
                f"    expected: {entry.sha256}\n"
                f"    actual:   {actual}"
            )
            print(f"  ❌ {entry.filename}: checksum drift")
        else:
            print(f"  ✓  {entry.filename}: ok ({entry.size_bytes:,} bytes)")

    if problems:
        print("\nVerification failed:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll raw data files verified against manifest.")
    return 0


def fetch_missing() -> int:
    """Best-effort download of any missing files.

    Several sources (World Bank, CPDS) don't expose stable direct-download
    URLs — they require clicking through a landing page. For those we
    print the landing URL and let the user download manually. Only KOF's
    current direct-link format is baked in.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    still_missing: list[RawFile] = []

    for entry in MANIFEST:
        path = RAW_DIR / entry.filename
        if path.exists():
            continue

        if entry.source_url and entry.source_url.endswith(".xlsx"):
            print(f"Downloading {entry.filename} from {entry.source_url} ...")
            try:
                urllib.request.urlretrieve(entry.source_url, path)
                print(f"  saved to {path}")
            except Exception as e:  # noqa: BLE001 — surface any failure
                print(f"  download failed: {e}")
                still_missing.append(entry)
        else:
            still_missing.append(entry)

    if still_missing:
        print("\nThe following files could not be downloaded automatically.")
        print("Please visit each landing page and place the file in data/raw/:")
        for entry in still_missing:
            print(f"  - {entry.filename}")
            print(f"      {entry.source_url or '(no URL on file)'}")
        return 1
    return 0


def show_manifest() -> int:
    print(f"{'filename':<25}  {'bytes':>10}  description")
    print(f"{'-' * 25}  {'-' * 10}  {'-' * 40}")
    for entry in MANIFEST:
        print(f"{entry.filename:<25}  {entry.size_bytes:>10,}  {entry.description}")
        if entry.source_url:
            print(f"{'':<25}  {'':>10}  ↳ {entry.source_url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--show", action="store_true", help="Print the manifest and exit (no verification)."
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Attempt to download files listed in the manifest that are missing locally.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Treat MISSING files as warnings instead of errors.",
    )
    args = parser.parse_args(argv)

    if args.show:
        return show_manifest()

    if args.fetch:
        rc = fetch_missing()
        if rc != 0:
            return rc

    return verify(strict=not args.no_strict)


if __name__ == "__main__":
    sys.exit(main())
