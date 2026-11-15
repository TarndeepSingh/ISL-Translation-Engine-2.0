"""Read a remote ZIP archive's index and individual members with HTTP Range requests.

A ZIP stores its table of contents at the END of the file, so ``zipfile`` can list and extract single
members from a seekable file-like object without the rest of the archive being fetched. ``RangeFile`` is such
an object backed by ``Range:`` requests. For INCLUDE this means: ask Zenodo for ~100 kB of index per archive
and then only the videos you want, instead of 56.8 GB of zips.

Safety properties
* A server that ignores ``Range`` (answers 200 with the whole body) raises ``RangeNotSupported``; the body
  is never read, so a 1 GB archive cannot be pulled by accident.
* Every response length is checked against the requested range; short reads are retried.
* ``zipfile`` verifies each member's CRC-32 while reading, so a corrupt range surfaces as ``BadZipFile``.
  (The archive-level MD5 is NOT checked in this mode, because the whole file is never downloaded.)
"""
from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.request

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


class RangeNotSupported(RuntimeError):
    """The server did not honour an HTTP Range request."""


class RangeFile(io.RawIOBase):
    """Read-only, seekable file backed by HTTP Range requests, with a one-block cache."""

    def __init__(self, url: str, size: int | None = None, *, block: int = 1 << 20, tries: int = 6,
                 timeout: float = 60.0, opener=None, sleep=time.sleep):
        super().__init__()
        self.url, self.block, self.tries, self.timeout = url, max(4096, int(block)), tries, timeout
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep
        self._pos = 0
        self._cache_start, self._cache = 0, b""
        self.requests = 0
        self.bytes_fetched = 0
        self.size = int(size) if size else self._probe_size()

    # ---- raw HTTP -------------------------------------------------------------------------------
    def _get(self, start: int, end: int) -> bytes:
        """Fetch bytes [start, end] inclusive with retries; returns exactly that range."""
        last_error: Exception | None = None
        for attempt in range(self.tries):
            try:
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with self._open(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", None) or response.getcode()
                    if status != 206:
                        raise RangeNotSupported(f"server answered HTTP {status} to a Range request for {self.url}")
                    data = response.read()
                self.requests += 1
                if len(data) != end - start + 1:
                    raise OSError(f"short read: wanted {end - start + 1} bytes, got {len(data)}")
                self.bytes_fetched += len(data)
                return data
            except RangeNotSupported:
                raise
            except (OSError, urllib.error.URLError) as exc:  # includes HTTPError, timeouts, short reads
                last_error = exc
                self._sleep(min(30.0, 2.0 ** attempt))
        raise OSError(f"range request failed after {self.tries} tries: {last_error}")

    def _probe_size(self) -> int:
        request = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with self._open(request, timeout=self.timeout) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 206:
                raise RangeNotSupported(f"server answered HTTP {status} to a Range request for {self.url}")
            match = _CONTENT_RANGE.match(response.headers.get("Content-Range", ""))
            response.read()
        if not match or match.group(3) == "*":
            raise RangeNotSupported("server did not report the total size in Content-Range")
        return int(match.group(3))

    # ---- file protocol --------------------------------------------------------------------------
    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def tell(self) -> int: return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def prefetch(self, start: int, length: int) -> None:
        """Load [start, start+length) into the cache with ONE request (call before reading a member)."""
        start = max(0, int(start))
        end = min(self.size, start + max(1, int(length))) - 1
        if end >= start and not self._cached(start, end + 1):
            self._cache_start, self._cache = start, self._get(start, end)

    def _cached(self, start: int, stop: int) -> bool:
        return self._cache_start <= start and stop <= self._cache_start + len(self._cache)

    def readinto(self, buffer) -> int:
        want = min(len(buffer), max(0, self.size - self._pos))
        if want == 0:
            return 0
        if not self._cached(self._pos, self._pos + 1):
            fetch = min(max(want, self.block), self.size - self._pos)
            self._cache_start, self._cache = self._pos, self._get(self._pos, self._pos + fetch - 1)
        offset = self._pos - self._cache_start
        chunk = self._cache[offset:offset + want]
        buffer[:len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)
