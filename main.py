import json
import logging
import mimetypes
import os
import random
import re
import time
from base64 import b64decode, b64encode
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

from blake3 import blake3
from cloudflare import BaseModel as CloudflareBaseModel
from cloudflare import Cloudflare, CloudflareError
from cloudflare.types.pages import Deployment
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("cloudflare").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Constants
BULK_UPLOAD_CONCURRENCY = 3
MAX_ASSET_COUNT = 20000
MAX_ASSET_SIZE = 25 * 1024 * 1024  # 25 MiB
MAX_BUCKET_FILE_COUNT = 1000
MAX_BUCKET_SIZE = 50 * 1024 * 1024  # 50 MiB
MAX_DEPLOYMENT_ATTEMPTS = 5
MAX_UPLOAD_ATTEMPTS = 3

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cloudflare-pages")


class CustomCloudflareResponse(CloudflareBaseModel):
    """Pydantic model for custom cloudflare response."""

    result: Dict[str, Any]


class FileInfo(BaseModel):
    """Pydantic model for file information."""

    path: Path
    content_type: str
    size: int
    hash: str
    start_offset: int = 0


class Bucket(BaseModel):
    """Pydantic model for upload buckets."""

    files: List[FileInfo] = Field(default_factory=list)
    remaining_size: int = MAX_BUCKET_SIZE


class ProgressBar:
    """Progress bar implementation using tqdm."""

    def __init__(self, total_bytes: int, desc: str = "Uploading", unit: str = "B"):
        """Initialize the progress bar.

        Args:
            total_bytes: Total bytes to upload
            desc: Description for the progress bar
            unit: Unit of measurement
        """
        self.total_bytes = total_bytes
        self.pbar = tqdm(
            total=total_bytes,
            desc=desc,
            unit=unit,
            unit_scale=True,
            unit_divisor=1024,
        )
        self._last_update = 0

    def update(self, current_bytes: int, total_bytes: int) -> None:
        """Update the progress bar.

        Args:
            current_bytes: Current bytes uploaded
            total_bytes: Total bytes to upload
        """
        if total_bytes != self.total_bytes:
            self.pbar.total = total_bytes
            self.total_bytes = total_bytes
            self.pbar.refresh()

        increment = current_bytes - self._last_update
        if increment > 0:
            self.pbar.update(increment)
            self._last_update = current_bytes

        if current_bytes == total_bytes:
            self.pbar.close()


class FileMapUploader:
    """Handles uploading files in buckets to Cloudflare Pages."""

    def __init__(
        self,
        cf: Cloudflare,
        filemap: Dict[str, FileInfo],
        account_id: str,
        project_name: str,
        concurrency: int = BULK_UPLOAD_CONCURRENCY,
        max_retries: int = MAX_UPLOAD_ATTEMPTS,
        max_bucket_file_count: int = MAX_BUCKET_FILE_COUNT,
        max_bucket_size: int = MAX_BUCKET_SIZE,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize the bucket uploader.

        Args:
            cf: Cloudflare client instance
            upload_token: Upload token for authentication
            max_retries: Maximum number of upload retry attempts
            concurrency: Number of concurrent uploads
            on_progress: Optional callback for progress updates (current_bytes, total_bytes)
        """
        self.cf = cf.with_options(max_retries=max_retries)
        self.filemap = filemap
        self.account_id = account_id
        self.project_name = project_name
        self.concurrency = concurrency
        self.max_bucket_file_count = max_bucket_file_count
        self.max_bucket_size = max_bucket_size
        self._total_bytes = sum(file.size for file in filemap.values())
        self._uploaded_bytes = 0

        # Create progress bar if no progress callback is provided
        if on_progress is None:
            progress_bar = ProgressBar(self._total_bytes)
            self.on_progress = progress_bar.update
        else:
            self.on_progress = on_progress

        self.__upload_token = None
        self.__upload_token_lock = Lock()

    @property
    def _upload_token(self) -> str:
        """Get upload token."""
        return self._fetch_upload_token()

    def _fetch_upload_token(self) -> str:
        """Fetch upload token or use cached token if not expired."""
        with self.__upload_token_lock:
            if self.__upload_token and not self._is_jwt_expired(self.__upload_token):
                return self.__upload_token
            else:
                response = self.cf.get(
                    f"/accounts/{self.account_id}/pages/projects/{self.project_name}/upload-token",
                    cast_to=CustomCloudflareResponse,
                )
                self.__upload_token = response.result["jwt"]
                return self.__upload_token

    def upload_files(self) -> None:
        """Upload files in parallel using buckets.

        Args:
            files: List of FileInfo objects to upload
        """
        # Sort files by size (largest first)
        sorted_files = sorted(self.filemap.values(), key=lambda f: f.size, reverse=True)

        # Create initial buckets
        buckets = [Bucket() for _ in range(self.concurrency)]

        # Distribute files across buckets
        bucket_offset = 0
        for file in sorted_files:
            inserted = False

            for i in range(len(buckets)):
                bucket_index = (i + bucket_offset) % len(buckets)
                bucket = buckets[bucket_index]

                if (
                    bucket.remaining_size >= file.size
                    and len(bucket.files) < self.max_bucket_file_count
                ):
                    bucket.files.append(file)
                    bucket.remaining_size -= file.size
                    inserted = True
                    break

            if not inserted:
                new_bucket = Bucket(
                    files=[file], remaining_size=self.max_bucket_size - file.size
                )
                buckets.append(new_bucket)

            bucket_offset += 1

        # Process buckets in parallel
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # Filter out empty buckets
            populated_buckets = [b for b in buckets if b.files]

            futures: List[Future] = []
            for bucket in populated_buckets:
                futures.append(executor.submit(self._upload_bucket, bucket))

            # Wait for all uploads to complete
            for future in futures:
                future.result()

    def _upload_bucket(self, bucket: Bucket) -> None:
        """Upload a bucket of files."""
        try:
            # Prepare payload
            payload = []
            bucket_size = 0
            for file in bucket.files:
                with file.path.open("rb") as f:
                    file_content = b64encode(f.read()).decode("utf-8")
                    bucket_size += file.size

                payload.append(
                    {
                        "key": file.hash,
                        "value": file_content,
                        "metadata": {"contentType": file.content_type},
                        "base64": True,
                    }
                )

            logger.debug(
                f"Uploading {len(payload)} files ({bucket_size/1024/1024:.1f}MB)"
            )
            response = self.cf.post(
                path="/pages/assets/upload",
                cast_to=CustomCloudflareResponse,
                body=payload,
                options={
                    "headers": {"Authorization": f"Bearer {self._upload_token}"},
                },
            )
            successful_key_count = response.result["successful_key_count"]

            if successful_key_count != len(bucket.files):
                logger.warning(
                    f"Uploaded {successful_key_count} files out of {len(bucket.files)}"
                )

            self._uploaded_bytes += bucket_size
            self.on_progress(self._uploaded_bytes, self._total_bytes)

            logger.debug(
                f"Uploaded {len(payload)} files ({bucket_size/1024/1024:.1f}MB)"
            )
        except Exception as e:
            logger.exception(f"Upload failed: {str(e)}")
            raise e

    def _is_jwt_expired(self, token: str) -> bool:
        """Check if JWT token is expired."""
        try:
            # Decode JWT payload
            payload = token.split(".")[1]
            # Add padding if needed
            payload += "=" * (4 - len(payload) % 4) if len(payload) % 4 else ""
            decoded_jwt = json.loads(b64decode(payload).decode("utf-8"))

            # Check expiration
            date_now = time.time()
            return decoded_jwt.get("exp", 0) <= date_now
        except Exception as e:
            raise CloudflareError(f"Invalid JWT token: {str(e)}")
        
    def _upsert_hashes(self) -> None:
        """Update file hashes in Cloudflare Pages."""
        hashes = [file.hash for file in self.filemap.values()]
        
        try:
            self.cf.post(
                path="/pages/assets/upsert-hashes",
                cast_to=CloudflareBaseModel,
                body={"hashes": hashes},
                options={
                    "headers": {"Authorization": f"Bearer {self._upload_token}"},
                },
            )
        except Exception as error:
            logger.warning(
                f"Failed to update file hashes: {str(error)}\n\n"
                "Failed to update file hashes. Every upload appeared to succeed for this deployment, "
                "but you might need to re-upload for future deployments. This shouldn't have any impact "
                "other than slowing the upload speed of your next deployment."
            )

class CloudflarePages:
    """Python implementation of CFPagesUploader for Cloudflare Pages deployment."""

    def __init__(
        self,
        api_token: Optional[str] = os.getenv("CLOUDFLARE_API_TOKEN"),
        account_id: Optional[str] = os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        account_name: Optional[str] = os.getenv("CLOUDFLARE_ACCOUNT_NAME"),
        project_name: Optional[str] = os.getenv("CLOUDFLARE_PROJECT_NAME"),
    ):
        """Initialize the CloudflarePages client."""
        self.cf = Cloudflare(max_retries=3, api_token=api_token)
        self.account_id = account_id or self._get_account_id(account_name)
        self.project_name = project_name or self._get_project_name(account_name)

    def _get_random_uuid_substring(self, length: int = 8) -> str:
        return uuid4().hex[:length]

    def _get_account_id(self, name: Optional[str] = None) -> str:
        """Get account ID."""

        for account in self.cf.accounts.list():
            if name and account.name.lower() != name.lower():
                continue

            logger.info(f"Found account: {account.name}")
            return account.id

        raise CloudflareError(f"Account {name} not found")

    def _get_project_name(self, name: Optional[str] = None) -> str:
        """Get project name."""
        # fmt: off
        regions = ["east", "west", "south", "north", "southeast", "southwest", "northeast", "northwest"]
        countries = ["us", "eu", "cn", "in", "br", "za", "au", "jp", "kr"]
        # fmt: on
        match_regex = f"({'|'.join(countries)})-({'|'.join(regions)})-({'|'.join(str(i) for i in range(1, 4))})"
        if name:
            match_regex = f"({name})|({match_regex})"

        for project in self.cf.pages.projects.list(account_id=self.account_id):
            if re.match(match_regex, project.name, re.IGNORECASE):
                logger.info(f"Found existing project: {project.name}")
                return project.name

        name = f"{random.choice(countries)}-{random.choice(regions)}-{random.randint(1, 3)}"
        return self._create_new_project(name)

    def _create_new_project(self, name: str) -> str:
        """Create new project."""
        logger.info(f"Creating new project: {name}")
        response = self.cf.pages.projects.create(
            name=name,
            account_id=self.account_id,
            production_branch="main",
        )
        return response.name

    def _deploy(self, directory: str | Path) -> Deployment:
        """Push directory to Cloudflare Pages."""
        directory = Path(directory).resolve()
        if not directory.exists():
            raise CloudflareError(f"Directory {directory} does not exist")

        file_map = self._create_file_map(directory)
        manifest = self._upload_file_map(file_map)

        form_data = {"manifest": json.dumps(manifest)}

        try:
            response = self.cf.pages.projects.deployments.create(
                project_name=self.project_name,
                account_id=self.account_id,
                branch=self._get_random_uuid_substring(),
                extra_body=form_data,
            )
            return response
        except CloudflareError as error:
            logger.error(f"Failed to deploy: {str(error)}")
            raise error

    def _upload_file_map(self, file_map: Dict[str, FileInfo]) -> Dict[str, str]:
        """Upload files to Cloudflare Pages."""
        files = list(file_map.values())

        start = datetime.now()
        missing_hashes = [file.hash for file in files]

        # Sort files by size (largest first)
        sorted_files = sorted(
            [file for file in files if file.hash in missing_hashes],
            key=lambda f: f.size,
            reverse=True,
        )

        # Create bucket uploader
        uploader = FileMapUploader(
            cf=self.cf,
            filemap=file_map,
            account_id=self.account_id,
            project_name=self.project_name,
        )

        # Upload files
        uploader.upload_files()

        elapsed = datetime.now() - start

        skipped = len(file_map) - len(missing_hashes)
        skipped_message = f"({skipped} already uploaded) " if skipped > 0 else ""

        logger.info(
            f"Upload complete: {len(sorted_files)} files "
            f"{skipped_message}{self._format_time(elapsed)}"
        )

        # Create manifest
        return {f"/{filename}": file.hash for filename, file in file_map.items()}


    def _format_time(self, duration: datetime) -> str:
        """Format time from datetime to HH:MM:SS."""
        return str(duration).split(".")[0]

    def _create_file_map(self, path: Path) -> Dict[str, FileInfo]:
        """Validate directory and create file map."""
        file_map: Dict[str, FileInfo] = {}

        if path.is_file():
            filestat = path.stat()
            if filestat.st_size > MAX_ASSET_SIZE:
                raise CloudflareError(
                    f"Error: Pages only supports files up to {MAX_ASSET_SIZE} in size\n"
                    f"{path.name} is {filestat.st_size} in size"
                )

            file_map[path.name] = FileInfo(
                path=path,
                content_type=mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                size=filestat.st_size,
                hash=self._hash_file(path),
            )
            return file_map

        for file in path.rglob("**/*"):
            if file.is_dir():
                continue

            relative_filepath = file.relative_to(path)
            filestat = file.stat()

            # Use forward slashes for consistency
            name = str(relative_filepath).replace(os.path.sep, "/")

            # Check file size
            if filestat.st_size > MAX_ASSET_SIZE:
                raise CloudflareError(
                    f"Error: Pages only supports files up to {MAX_ASSET_SIZE} in size\n"
                    f"{name} is {filestat.st_size} in size"
                )

            # Create file info
            file_map[name] = FileInfo(
                path=file,
                content_type=mimetypes.guess_type(name)[0]
                or "application/octet-stream",
                size=filestat.st_size,
                hash=self._hash_file(file),
            )

        # Check file count
        if len(file_map) > MAX_ASSET_COUNT:
            raise CloudflareError(
                f"Error: Pages only supports up to {MAX_ASSET_COUNT:,} files in a deployment. "
                "Ensure you have specified your build output directory correctly."
            )

        return file_map

    def _hash_file(self, filepath: Path) -> str:
        """Generate hash for a file."""
        with filepath.open("rb") as f:
            contents = f.read()

        base64_contents = b64encode(contents)
        extension = filepath.suffix[1:]  # Remove leading dot

        # Use blake3 for hashing
        hash_obj = blake3(base64_contents + extension.encode())
        return hash_obj.hexdigest()[:32]

    def upload(self, path: str | Path) -> None:
        """Upload a file to Cloudflare Pages."""

        now = datetime.now()
        result = self._deploy(path)
        logger.info(
            f"Uploaded to: {result.url} ({self._format_time(datetime.now() - now)})"
        )
