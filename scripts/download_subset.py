"""Selective download of the minimal training subset from the Kaggle competition.

Fetches only scenes/train.zarr, semantic_map/, and meta.json, the smallest set
that supports real training within a small disk budget; the aerial map and the
other zarr archives are deliberately excluded. Requires having joined the
competition (accepting its rules) and an authenticated Kaggle CLI.

Resumable: files already present with the size the manifest expects are
skipped, so rerunning after an interruption continues where it left off.

Usage:
  uv run python scripts/download_subset.py --report    # manifest + size check only
  uv run python scripts/download_subset.py --download  # fetch the subset into data/
"""

import argparse
import csv
import json
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COMPETITION = "lyft-motion-prediction-autonomous-vehicles"
SUBSET_PREFIXES = (
    "scenes/train.zarr/",
    "scenes/sample.zarr/",
    "semantic_map/",
    "meta.json",
)
SMALL_PREFIXES = ("scenes/sample.zarr/", "semantic_map/", "meta.json")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA_DIR / "manifest.csv"
MANIFEST_PARTIAL = DATA_DIR / "manifest.partial.csv"
MANIFEST_STATE = DATA_DIR / "manifest.state.json"
PAGE_SIZE = 200
PAGE_PAUSE = 1.2
WORKERS = 4
RETRIES = 6

_thread_local = threading.local()


def make_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def thread_api():
    if not hasattr(_thread_local, "api"):
        _thread_local.api = make_api()
    return _thread_local.api


def call_with_backoff(description: str, call):
    """Run an API call, waiting out 429 rate limits with growing pauses."""
    for attempt in range(RETRIES + 1):
        try:
            return call()
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status != 429 or attempt == RETRIES:
                raise
            pause = min(120, 20 * (attempt + 1))
            print(f"  rate limited on {description}, waiting {pause}s...", flush=True)
            time.sleep(pause)


def build_manifest(refresh: bool = False) -> list[tuple[str, int]]:
    """Page through the full competition file listing once, checkpointing every
    page so a crash or rate-limit abort resumes instead of starting over."""
    if MANIFEST.exists() and not refresh:
        with MANIFEST.open() as fh:
            return [(name, int(size)) for name, size in csv.reader(fh)]
    DATA_DIR.mkdir(exist_ok=True)
    entries: list[tuple[str, int]] = []
    token = None
    if MANIFEST_PARTIAL.exists() and MANIFEST_STATE.exists() and not refresh:
        with MANIFEST_PARTIAL.open() as fh:
            entries = [(name, int(size)) for name, size in csv.reader(fh)]
        token = json.loads(MANIFEST_STATE.read_text())["token"]
        print(f"  resuming listing at {len(entries)} files", flush=True)
    api = make_api()
    with MANIFEST_PARTIAL.open("a", newline="") as fh:
        writer = csv.writer(fh)
        while True:
            response = call_with_backoff(
                "file listing",
                lambda token=token: api.competition_list_files(
                    COMPETITION, page_token=token, page_size=PAGE_SIZE
                ),
            )
            page = [(f.name, int(f.total_bytes)) for f in response.files]
            writer.writerows(page)
            fh.flush()
            entries.extend(page)
            token = response.next_page_token or None
            MANIFEST_STATE.write_text(json.dumps({"token": token}))
            if len(entries) % 5000 < PAGE_SIZE:
                print(f"  listed {len(entries)} files...", flush=True)
            if token is None:
                break
            time.sleep(PAGE_PAUSE)
    deduped = dict(entries)
    with MANIFEST.open("w", newline="") as fh:
        csv.writer(fh).writerows(sorted(deduped.items()))
    MANIFEST_PARTIAL.unlink()
    MANIFEST_STATE.unlink()
    return sorted(deduped.items())


def subset_of(entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return [e for e in entries if e[0].startswith(SUBSET_PREFIXES)]


def report(entries: list[tuple[str, int]]) -> None:
    groups: dict[str, list[int]] = {}
    for name, size in entries:
        parts = name.split("/")
        key = "/".join(parts[:2]) if parts[0] == "scenes" else parts[0]
        groups.setdefault(key, []).append(size)
    print(f"{'group':40} {'files':>8} {'GiB':>8}")
    for key in sorted(groups):
        sizes = groups[key]
        print(f"{key:40} {len(sizes):>8} {sum(sizes) / 2**30:>8.2f}")
    chosen = subset_of(entries)
    total = sum(size for _, size in chosen)
    free = shutil.disk_usage(DATA_DIR.parent).free
    print(f"\nsubset {SUBSET_PREFIXES}")
    print(f"  {len(chosen)} files, {total / 2**30:.2f} GiB")
    print(f"  free disk: {free / 2**30:.2f} GiB")
    if total * 1.1 > free:
        print("  WARNING: less than 10% headroom after download")


def fetch_one(name: str, size: int) -> int:
    dest = DATA_DIR / name
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            call_with_backoff(
                f"download {name}",
                lambda: thread_api().competition_download_file(
                    COMPETITION, name, path=str(dest.parent), force=True, quiet=True
                ),
            )
            zipped = dest.with_name(dest.name + ".zip")
            if not dest.exists() and zipped.exists():
                with zipfile.ZipFile(zipped) as zf:
                    zf.extractall(dest.parent)
                zipped.unlink()
            if dest.exists() and dest.stat().st_size == size:
                return size
            raise OSError(
                f"post-download check failed for {name}: "
                f"exists={dest.exists()} size={dest.stat().st_size if dest.exists() else None}"
            )
        except Exception as error:
            last_error = error
            time.sleep(2 ** (attempt + 1))
    raise OSError(f"giving up on {name}: {last_error}")


def download(
    entries: list[tuple[str, int]], prefixes: tuple[str, ...] = SUBSET_PREFIXES
) -> None:
    chosen = [e for e in entries if e[0].startswith(prefixes)]
    pending = [
        (n, s)
        for n, s in chosen
        if not ((DATA_DIR / n).exists() and (DATA_DIR / n).stat().st_size == s)
    ]
    pending.sort(key=lambda e: e[0].startswith("scenes/train.zarr/"))
    total_bytes = sum(s for _, s in pending)
    print(
        f"{len(pending)} of {len(chosen)} files to fetch ({total_bytes / 2**30:.2f} GiB)"
    )
    done_files = 0
    done_bytes = 0
    failures: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, n, s): n for n, s in pending}
        for future in as_completed(futures):
            try:
                done_bytes += future.result()
            except Exception as error:
                failures.append(str(error))
            done_files += 1
            if done_files % 200 == 0 or done_files == len(pending):
                rate = done_bytes / max(time.time() - started, 1) / 2**20
                print(
                    f"  {done_files}/{len(pending)} files, "
                    f"{done_bytes / 2**30:.2f} GiB, {rate:.1f} MiB/s",
                    flush=True,
                )
    missing = [
        n
        for n, s in chosen
        if not ((DATA_DIR / n).exists() and (DATA_DIR / n).stat().st_size == s)
    ]
    if failures or missing:
        print(f"FAILED: {len(failures)} errors, {len(missing)} files incomplete")
        for line in failures[:10]:
            print("  " + line)
        sys.exit(1)
    print(
        f"complete: subset verified, {sum(s for _, s in chosen) / 2**30:.2f} GiB on disk"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="fetch only sample.zarr + semantic map + meta.json (~60 MiB)",
    )
    parser.add_argument("--refresh-manifest", action="store_true")
    args = parser.parse_args()
    entries = build_manifest(refresh=args.refresh_manifest)
    if args.report or not (args.download or args.sample_only):
        report(entries)
    if args.sample_only:
        download(entries, prefixes=SMALL_PREFIXES)
    elif args.download:
        download(entries)


if __name__ == "__main__":
    main()
