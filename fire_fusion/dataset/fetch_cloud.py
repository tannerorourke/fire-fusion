"""
Backblaze B2 transit for raw source data and run artifacts.
Processed transfers selected by build step.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..config.dataset_config import DATASET_CONFIGS
from ..config.path_config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LANDFIRE_DIR, NLCD_DIR, GPW_DIR, CROADS_DIR,
    USFS_DIR, FPA_FOD_DIR, PRISM_DIR, AORC_DIR, MODIS_DIR, FIRMS_DIR, USDA_DIR, NCEI_SWDI_DIR,
)


# -- Config ------------------------------------------------------------------
# B2 key namespaces. Local layout round-trips to the same paths: raw sources under
# data/raw/<source>, built cubes under data/processed/<dataset>, runs under runs/.
RAW_PREFIX = "raw"
PROCESSED_PREFIX = "processed"

# Source name -> local directory. Doubles as the B2 key segment.
RAW_SOURCES: Dict[str, Path] = {
    p.name: p for p in [
        LANDFIRE_DIR, NLCD_DIR, GPW_DIR, CROADS_DIR, USFS_DIR, FPA_FOD_DIR,
        PRISM_DIR, AORC_DIR, MODIS_DIR, FIRMS_DIR, USDA_DIR, NCEI_SWDI_DIR,
    ]
}

PROCESSED_STEPS: Dict[str, Tuple[str, ...]] = {
    "staging": ("cube.zarr",),
    "published": ("dataset.zarr", "dataset_manifest.json"),
    "splits": ("train.zarr", "eval.zarr", "test.zarr", "manifest.json"),
}

# -- CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Sync source data and built cubes between local disk and B2.")
parser.add_argument("action", choices=["push", "pull"])
parser.add_argument("--kind", choices=["raw", "processed"], default="raw",
                help="raw sources (data/raw) or built cubes (data/processed)")
parser.add_argument("--sources", nargs="+", choices=sorted(RAW_SOURCES), default=None,
                help="raw only: subset of sources; default all")
parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIGS), default=None,
                help="processed only: dataset names; push defaults to all built locally")
parser.add_argument("--step", nargs="+", choices=sorted(PROCESSED_STEPS), default=["splits"],
                help="processed only: which build steps' outputs to move")
parser.add_argument("--fold", default="full",
                help="processed only: which fold's split stores to move")
parser.add_argument("--overwrite", action="store_true")


class B2Store:
    """ Minimal keyed object store over a Backblaze B2 bucket (S3 API). """

    def __init__(self, bucket: Optional[str] = None, endpoint: Optional[str] = None):
        """ Open an S3 client against the B2 endpoint and verify the bucket is reachable. """
        import boto3
        from botocore.config import Config

        endpoint = endpoint or os.environ["B2_ENDPOINT"]
        self.bucket = bucket or os.environ["B2_BUCKET"]
        # -- B2 signs against the region embedded in the endpoint host
        m = re.search(r"s3\.([^.]+)\.backblazeb2\.com", endpoint)
        region = m.group(1) if m else os.environ.get("B2_REGION", "us-east-005")

        # -- B2 throttles bursty multi-GB transfers; standard-mode retries absorb
        #    the SlowDown and connection-reset errors
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=os.environ["B2_KEY_ID"],
            aws_secret_access_key=os.environ["B2_APP_KEY"],
            config=Config(retries={"max_attempts": 10, "mode": "standard"}),
        )
        self.client.head_bucket(Bucket=self.bucket)

    def _remote_size(self, key: str) -> Optional[int]:
        from botocore.exceptions import ClientError
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        except ClientError:
            return None

    def put_file(self, local: Path, key: str, overwrite: bool = False) -> None:
        size = self._remote_size(key)
        if size is not None and not overwrite and size == local.stat().st_size:
            print(f"[B2] skip {key} (present, same size)")
            return
        self.client.upload_file(str(local), self.bucket, key)
        print(f"[B2] put  {key}")

    def get_file(self, key: str, local: Path, overwrite: bool = False) -> None:
        if local.exists() and not overwrite:
            print(f"[B2] skip {local} (present)")
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local))
        print(f"[B2] get  {key}")

    def put_tree(self, root: Path, key_prefix: str, overwrite: bool = False) -> None:
        if not root.exists():
            print(f"[B2] {root} absent, nothing to push")
            return
        for f in sorted(root.rglob("*")):
            if f.is_file():
                self.put_file(f, f"{key_prefix}/{f.relative_to(root).as_posix()}", overwrite)

    def get_tree(self, key_prefix: str, root: Path, overwrite: bool = False) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{key_prefix}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                self.get_file(key, root / key[len(key_prefix) + 1:], overwrite)


def _resolve_sources(names: Optional[Iterable[str]]) -> Dict[str, Path]:
    if not names:
        return RAW_SOURCES
    return {n: RAW_SOURCES[n] for n in names}


def _processed_datasets(names: Optional[Iterable[str]]) -> List[str]:
    if names:
        return list(names)
    # push with no explicit list -> every built cube on local disk; a pull needs
    # explicit names (the local processed dir may not exist on a fresh node)
    if not PROCESSED_DATA_DIR.exists():
        return []
    return sorted(p.name for p in PROCESSED_DATA_DIR.iterdir() if p.is_dir())


def _step_members(steps: Iterable[str]) -> List[str]:
    seen = {m: None for s in steps for m in PROCESSED_STEPS[s]}
    return list(seen)


def _sync_processed(
    store: "B2Store", action: str, ds: str,
    members: Iterable[str], overwrite: bool, fold: str = "full",
) -> None:
    """ Push or pull each named build member of dataset ds between local disk and B2. """
    # -- split stores of a non-default fold live one directory down, on both sides
    for member in members:
        sub = fold if fold != "full" and member in PROCESSED_STEPS["splits"] else None
        local = PROCESSED_DATA_DIR / ds / sub / member if sub else PROCESSED_DATA_DIR / ds / member
        key = f"{PROCESSED_PREFIX}/{ds}/{sub}/{member}" if sub else f"{PROCESSED_PREFIX}/{ds}/{member}"
        if action == "push":
            if local.is_dir():
                store.put_tree(local, key, overwrite)
            elif local.is_file():
                store.put_file(local, key, overwrite)
            else:
                print(f"[B2] {local} absent, skipping")
        elif member.endswith(".json"):
            store.get_file(key, local, overwrite)
        else:
            store.get_tree(key, local, overwrite)


def main() -> None:
    args = parser.parse_args()

    store = B2Store()
    if args.kind == "raw":
        for name, path in _resolve_sources(args.sources).items():
            prefix = f"{RAW_PREFIX}/{name}"
            if args.action == "push":
                store.put_tree(path, prefix, args.overwrite)
            else:
                store.get_tree(prefix, RAW_DATA_DIR / name, args.overwrite)
    else:
        members = _step_members(args.step)
        for ds in _processed_datasets(args.datasets):
            _sync_processed(store, args.action, ds, members, args.overwrite, args.fold)


if __name__ == "__main__":
    main()
