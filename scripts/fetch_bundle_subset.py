"""Fetch the training subset by reading the competition bundle zip remotely.

Kaggle's per-file download API rate-limits to a crawl on a 52k-file zarr. The
bundle download URL, however, is a signed Google Cloud Storage object that
supports HTTP range requests, so this script opens the remote zip without
downloading it, reads the central directory, and streams out only the members
under the subset prefixes: one URL authorization plus a few hundred large
range reads instead of 52k throttled API calls. Members are CRC-checked by
zipfile during extraction; existing complete files are skipped, so the script
is resumable.

Usage:
  uv run python scripts/fetch_bundle_subset.py
"""

import io
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_subset import COMPETITION, DATA_DIR, SUBSET_PREFIXES  # noqa: E402

BLOCK_SIZE = 32 * 2**20
PROGRESS_EVERY = 2000


def bundle_url() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import (
        ApiDownloadDataFilesRequest,
    )

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = COMPETITION
        response = kaggle.competitions.competition_api_client.download_data_files(
            request
        )
    url = response.request.url
    response.close()
    return url


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of a remote object via HTTP range requests,
    with a single block cache sized for zipfile's sequential access pattern."""

    def __init__(self, refresh_url):
        self._refresh_url = refresh_url
        self._url = refresh_url()
        self._session = requests.Session()
        self._size = self._probe_size()
        self._pos = 0
        self._cache_start = -1
        self._cache = b""
        self.bytes_fetched = 0
        self.requests_made = 0

    def _probe_size(self) -> int:
        response = self._session.get(
            self._url, headers={"Range": "bytes=0-0"}, timeout=60
        )
        response.raise_for_status()
        return int(response.headers["Content-Range"].split("/")[-1])

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self._size) - 1
        for attempt in range(5):
            response = self._session.get(
                self._url, headers={"Range": f"bytes={start}-{end}"}, timeout=300
            )
            if response.status_code in (403, 410):
                self._url = self._refresh_url()
                continue
            if response.status_code in (206, 200):
                self.bytes_fetched += len(response.content)
                self.requests_made += 1
                return response.content
            time.sleep(2**attempt)
        raise OSError(f"range fetch failed at {start}: HTTP {response.status_code}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        return self._pos

    def readinto(self, buffer) -> int:
        if self._pos >= self._size:
            return 0
        want = min(len(buffer), self._size - self._pos)
        in_cache = (
            self._cache_start >= 0
            and self._cache_start <= self._pos < self._cache_start + len(self._cache)
        )
        if not in_cache:
            self._cache_start = self._pos
            self._cache = self._fetch(self._pos, max(want, BLOCK_SIZE))
        offset = self._pos - self._cache_start
        chunk = self._cache[offset : offset + want]
        buffer[: len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)


def main() -> None:
    remote = RemoteFile(bundle_url)
    print(f"remote bundle: {remote._size / 2**30:.2f} GiB")
    archive = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=2**20))
    members = [
        info
        for info in archive.infolist()
        if info.filename.startswith(SUBSET_PREFIXES) and not info.is_dir()
    ]
    members.sort(key=lambda info: info.header_offset)
    pending = [
        info
        for info in members
        if not (
            (DATA_DIR / info.filename).exists()
            and (DATA_DIR / info.filename).stat().st_size == info.file_size
        )
    ]
    total = sum(info.file_size for info in pending)
    print(
        f"{len(pending)} of {len(members)} members to extract ({total / 2**30:.2f} GiB)"
    )
    done_bytes = 0
    started = time.time()
    for index, info in enumerate(pending, 1):
        dest = DATA_DIR / info.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out, 2**20)
        if dest.stat().st_size != info.file_size:
            raise OSError(f"size mismatch after extracting {info.filename}")
        done_bytes += info.file_size
        if index % PROGRESS_EVERY == 0 or index == len(pending):
            rate = remote.bytes_fetched / max(time.time() - started, 1) / 2**20
            print(
                f"  {index}/{len(pending)} files, {done_bytes / 2**30:.2f} GiB out, "
                f"{remote.bytes_fetched / 2**30:.2f} GiB fetched "
                f"({remote.requests_made} requests, {rate:.1f} MiB/s)",
                flush=True,
            )
    missing = [
        info.filename
        for info in members
        if not (
            (DATA_DIR / info.filename).exists()
            and (DATA_DIR / info.filename).stat().st_size == info.file_size
        )
    ]
    if missing:
        print(f"FAILED: {len(missing)} members incomplete, e.g. {missing[:3]}")
        sys.exit(1)
    print(f"complete: {len(members)} members verified on disk")


if __name__ == "__main__":
    main()
