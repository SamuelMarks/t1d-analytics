"""Google Cloud Storage artifact synchronization, MD5 verification, and checkpoint download utilities."""

import asyncio
import base64
import hashlib
import importlib
import time
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from t1d_analytics.errors import (
    ChecksumMismatchError,
    DatabaseNotFoundError,
    GcsStorageError,
    InvalidGcsUriError,
    MissingSdkError,
)


class GcsBlobProtocol(Protocol):
    """Protocol representing a Google Cloud Storage Blob."""

    name: str
    md5_hash: Optional[str]

    def download_to_filename(self, filename: str) -> None:
        """Download blob contents to local file."""
        ...

    def upload_from_filename(self, filename: str) -> None:
        """Upload local file contents to blob."""
        ...


@runtime_checkable
class Md5BlobProtocol(Protocol):
    """Runtime-checkable protocol for GCS blobs exposing md5_hash."""

    @property
    def md5_hash(self) -> Optional[str]:
        """Return base64-encoded MD5 hash string."""
        ...


class GcsBucketProtocol(Protocol):
    """Protocol representing a Google Cloud Storage Bucket."""

    def blob(self, blob_name: str) -> GcsBlobProtocol:
        """Create a blob reference within this bucket."""
        ...


class GcsClientProtocol(Protocol):
    """Protocol representing a Google Cloud Storage Client."""

    def bucket(self, bucket_name: str) -> GcsBucketProtocol:
        """Return bucket reference by name."""
        ...

    def list_blobs(
        self, bucket_or_name: Union[GcsBucketProtocol, str], prefix: str = ""
    ) -> Iterable[GcsBlobProtocol]:
        """List blobs in bucket matching prefix."""
        ...


def _compute_md5_base64(payload: bytes) -> str:
    """
    Compute the base64-encoded MD5 digest of a byte payload for GCS integrity verification.

    Args:
    ----
        payload: Raw file bytes to hash.

    Returns:
    -------
        str: ASCII base64-encoded MD5 digest string.

    """
    return base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()
    ).decode("ascii")


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Validate a `gs://` URI and return `(bucket_name, prefix)`.

    Args:
    ----
        gcs_uri: GCS URI string starting with `gs://`.

    Returns:
    -------
        Tuple[str, str]: `(bucket_name, object_prefix)`.

    Raises:
    ------
        InvalidGcsUriError: If `gcs_uri` does not start with `gs://`.

    """
    if not gcs_uri.startswith("gs://"):
        raise InvalidGcsUriError(
            f"Invalid GCS bucket URI '{gcs_uri}'. Must start with 'gs://'."
        )
    parts = gcs_uri[5:].rstrip("/").split("/")
    bucket_name = parts[0]
    prefix = "/".join(parts[1:])
    return bucket_name, prefix


def _resolve_gcs_client(gcs_client: Optional[GcsClientProtocol]) -> GcsClientProtocol:
    """
    Return the provided `gcs_client` or dynamically instantiate `google.cloud.storage.Client()`.

    Args:
    ----
        gcs_client: Optional injectable GCS client instance.

    Returns:
    -------
        GcsClientProtocol: Active GCS client.

    Raises:
    ------
        MissingSdkError: If `google-cloud-storage` is not installed.

    """
    if gcs_client is not None:
        return gcs_client
    try:
        gcs_pkg = importlib.import_module("google.cloud.storage")
    except ImportError as e:
        raise MissingSdkError(
            "The 'google-cloud-storage' Python package is required. "
            "Install via 'pip install google-cloud-storage'."
        ) from e
    client: GcsClientProtocol = gcs_pkg.Client()
    return client


def verify_gcs_blob_md5(
    blob: GcsBlobProtocol,
    dest_file: Path,
    validate_checksum: bool = True,
) -> None:
    """
    Verify that a downloaded file's MD5 digest matches the remote GCS blob's `md5_hash`.

    Args:
    ----
        blob: Downloaded GCS blob reference.
        dest_file: Local destination file path.
        validate_checksum: Whether checksum validation is enabled.

    Raises:
    ------
        ChecksumMismatchError: If the computed local MD5 does not match `blob.md5_hash`.

    """
    remote_md5 = (
        blob.md5_hash
        if validate_checksum and isinstance(blob, Md5BlobProtocol)
        else None
    )
    if isinstance(remote_md5, str) and remote_md5 and dest_file.exists():
        local_md5 = _compute_md5_base64(dest_file.read_bytes())
        if local_md5 != remote_md5:
            raise ChecksumMismatchError(
                f"Checksum mismatch for '{dest_file.name}': expected {remote_md5}, got {local_md5}"
            )


async def download_checkpoints_from_gcs(
    bucket_url: str,
    output_dir: Path,
    max_retries: int = 3,
    validate_checksum: bool = True,
    gcs_client: Optional[GcsClientProtocol] = None,
) -> Path:
    """
    Download trained model checkpoints from Google Cloud Storage to a local directory.

    Args:
    ----
        bucket_url: GCS bucket URI containing model checkpoints.
        output_dir: Destination local directory path.
        max_retries: Maximum download retry attempts.
        validate_checksum: Whether to verify download checksum against manifest.
        gcs_client: Optional injectable GCS client instance.

    Returns:
    -------
        Path: Path to the local directory containing downloaded checkpoints.

    Raises:
    ------
        InvalidGcsUriError: If bucket URI does not begin with 'gs://'.
        ChecksumMismatchError: If checksum mismatches.
        GcsStorageError: If download fails after retries.

    """
    bucket_name, prefix = parse_gcs_uri(bucket_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _resolve_gcs_client(gcs_client)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            bucket = client.bucket(bucket_name)
            blobs = list(client.list_blobs(bucket, prefix=prefix))
            if not blobs:
                raise GcsStorageError(f"No checkpoint files found at '{bucket_url}'.")
            for blob in blobs:
                rel_name = blob.name[len(prefix) :].lstrip("/")
                dest_file = output_dir / rel_name
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(dest_file))
                verify_gcs_blob_md5(blob, dest_file, validate_checksum)
            return output_dir
        except ChecksumMismatchError:
            raise
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

    raise GcsStorageError(
        f"Checkpoint download failed after {max_retries} attempts: {last_err}"
    ) from last_err


def sync_dataset_to_gcs(
    local_path: Path,
    gcs_bucket: str,
    max_retries: int = 3,
    validate_checksum: bool = True,
    sleep_fn: Optional[Callable[[float], None]] = None,
    gcs_client: Optional[GcsClientProtocol] = None,
) -> str:
    """
    Synchronize exported dataset file to Google Cloud Storage bucket with retries and checksums.

    Args:
    ----
        local_path: Local dataset filepath.
        gcs_bucket: Destination GCS bucket URL (e.g., 'gs://bucket-name').
        max_retries: Number of upload retry attempts.
        validate_checksum: Whether to compute and verify MD5 checksum.
        sleep_fn: Optional injectable sleep function for retry backoff.
        gcs_client: Optional injectable GCS client instance.

    Returns:
    -------
        str: Remote GCS target URI.

    Raises:
    ------
        InvalidGcsUriError: If bucket URL is invalid.
        DatabaseNotFoundError: If local dataset file does not exist.
        GcsStorageError: If upload fails after retries.

    """
    bucket_name, blob_prefix = parse_gcs_uri(gcs_bucket)
    if not local_path.exists():
        raise DatabaseNotFoundError(f"Local file not found at '{local_path}'.")

    bucket_clean = gcs_bucket.rstrip("/")
    dest_uri = f"{bucket_clean}/{local_path.name}"
    client = _resolve_gcs_client(gcs_client)
    blob_name = f"{blob_prefix}/{local_path.name}".lstrip("/")
    local_md5 = (
        _compute_md5_base64(local_path.read_bytes()) if validate_checksum else ""
    )

    backoff_sleep = sleep_fn or time.sleep
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.md5_hash = local_md5
            blob.upload_from_filename(str(local_path))
            return dest_uri
        except Exception as e:
            last_err = e
            backoff_sleep(0.05 * (2 ** (attempt - 1)))

    raise GcsStorageError(
        f"GCS upload failed after {max_retries} attempts: {last_err}"
    ) from last_err


__all__ = [
    "GcsBlobProtocol",
    "GcsBucketProtocol",
    "GcsClientProtocol",
    "Md5BlobProtocol",
    "_compute_md5_base64",
    "_resolve_gcs_client",
    "download_checkpoints_from_gcs",
    "parse_gcs_uri",
    "sync_dataset_to_gcs",
    "verify_gcs_blob_md5",
]
