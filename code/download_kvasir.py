from __future__ import annotations

import argparse
import ssl
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def download_file(url: str, destination: Path, insecure_ssl: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context() if insecure_ssl else None

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_count * block_size
        percent = min(100.0, downloaded * 100.0 / total_size)
        print(f"\rDownloading {percent:5.1f}%", end="")

    try:
        if insecure_ssl:
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
            with opener.open(url) as response, destination.open("wb") as handle:
                total = int(response.headers.get("Content-Length", "0"))
                copied = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    copied += len(chunk)
                    if total:
                        print(f"\rDownloading {copied * 100.0 / total:5.1f}%", end="")
        else:
            urllib.request.urlretrieve(url, destination, reporthook=report)
    except ssl.SSLError:
        print("\nSSL verification failed; retrying with an unverified SSL context.")
        context = ssl._create_unverified_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
        with opener.open(url) as response, destination.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0"))
            copied = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                copied += len(chunk)
                if total:
                    print(f"\rDownloading {copied * 100.0 / total:5.1f}%", end="")
    print()


def extract_zip(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract Kvasir-SEG.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--archive-name", default="kvasir-seg.zip")
    parser.add_argument("--insecure-ssl", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    archive = args.output_dir / args.archive_name
    extracted = args.output_dir / "Kvasir-SEG"

    if args.skip_existing and extracted.exists():
        print(f"Dataset already exists at {extracted}")
        return

    if not archive.exists():
        download_file(args.url, archive, insecure_ssl=args.insecure_ssl)
    else:
        print(f"Using existing archive: {archive}")

    extract_zip(archive, args.output_dir)
    print(f"Extracted dataset to {extracted}")


if __name__ == "__main__":
    main()
