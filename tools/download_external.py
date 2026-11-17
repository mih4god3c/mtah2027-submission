"""Download the external validation data, pinned and checksummed. Standard library only.

    .venv/bin/python tools/download_external.py            # or: make external
    .venv/bin/python tools/download_external.py --verify   # check what is on disk, download nothing

What it fetches: the IDSSE dataset (Bassek et al., Scientific Data 2025) -- seven Bundesliga
matches of DFL position and event data, CC-BY 4.0. These are OTHER games, which RULES.md allows
for training and validation. Nothing here touches the competition match.

Source: the pysport mirror on Hugging Face, pinned to one revision so every teammate gets
byte-identical files. The canonical copy is on figshare (DOI 10.6084/m9.figshare.28196177), but
its download endpoint serves a bot check to scripts, so the mirror is the reproducible route.
Every file is verified against the hash the mirror publishes: sha256 for the large position
files, the git blob sha1 for the small ones.

Files land in external_data/idsse/, which is gitignored. About 2.5 GB.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "external_data", "idsse")

REPO = "pysport/idsse-data"
REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"   # pinned; change deliberately, never silently
TREE_URL = "https://huggingface.co/api/datasets/{}/tree/{}".format(REPO, REVISION)
FILE_URL = "https://huggingface.co/datasets/{}/resolve/{}/{{}}".format(REPO, REVISION)
CHUNK = 1 << 20


def listing():
    with urllib.request.urlopen(TREE_URL) as response:
        entries = json.load(response)
    return [entry for entry in entries
            if entry["type"] == "file" and entry["path"].startswith("DFL_")]


def file_ok(path, entry):
    """True if the file on disk matches the mirror's published hash and size."""
    if not os.path.isfile(path) or os.path.getsize(path) != entry["size"]:
        return False
    if entry.get("lfs"):
        digest = hashlib.sha256()
        expected = entry["lfs"]["oid"]
    else:
        digest = hashlib.sha1()
        digest.update("blob {}\0".format(entry["size"]).encode())
        expected = entry["oid"]
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def download(entry, path):
    partial = path + ".part"
    with urllib.request.urlopen(FILE_URL.format(entry["path"])) as response, \
            open(partial, "wb") as handle:
        for block in iter(lambda: response.read(CHUNK), b""):
            handle.write(block)
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="check files on disk only")
    args = parser.parse_args()

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    entries = listing()
    bad = 0
    for entry in sorted(entries, key=lambda e: e["path"]):
        path = os.path.join(OUT_DIR, entry["path"])
        if file_ok(path, entry):
            print("ok        {}".format(entry["path"]))
            continue
        if args.verify:
            print("MISSING   {}".format(entry["path"]) if not os.path.isfile(path)
                  else "MISMATCH  {}".format(entry["path"]))
            bad += 1
            continue
        print("download  {}  ({:.0f} MB)".format(entry["path"], entry["size"] / 1e6), flush=True)
        download(entry, path)
        if not file_ok(path, entry):
            print("MISMATCH after download: {}".format(entry["path"]))
            bad += 1
    print("\n{} files, {} problems, in {}".format(len(entries), bad, os.path.relpath(OUT_DIR, ROOT)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
