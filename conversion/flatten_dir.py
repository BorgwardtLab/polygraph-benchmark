#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Tuple


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten a directory tree by moving or copying all files from nested "
            "subdirectories up to the top-level directory, joining path "
            "components with a separator."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".local/data",
        help="Root directory to flatten. Defaults to .local/data",
    )
    parser.add_argument(
        "--sep",
        default="_",
        help="Separator used to join path components. Default: _",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them. Default: move",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without making changes.",
    )
    parser.add_argument(
        "--keep-dirs",
        action="store_true",
        help="Do not remove empty directories after flattening. Default: remove",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories (those starting with .).",
    )
    return parser.parse_args()


def iter_all_files(root_dir: Path, include_hidden: bool) -> Iterable[Path]:
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if not include_hidden:
            parts = path.relative_to(root_dir).parts
            if any(part.startswith(".") for part in parts):
                continue
        yield path


def split_name_and_suffixes(filename: str) -> Tuple[str, str]:
    p = Path(filename)
    suffix = "".join(p.suffixes)
    if suffix:
        stem = filename[: -len(suffix)]
    else:
        stem = filename
    return stem, suffix


def make_flat_name(root_dir: Path, file_path: Path, sep: str) -> str:
    rel_parts = file_path.relative_to(root_dir).parts
    stem, suffix = split_name_and_suffixes(rel_parts[-1])
    flat_stem_parts = list(rel_parts[:-1]) + [stem]
    flat_stem = sep.join(flat_stem_parts)
    return f"{flat_stem}{suffix}"


def ensure_unique_destination(dest_path: Path) -> Path:
    if not dest_path.exists():
        return dest_path
    base = dest_path.stem
    suffix = dest_path.suffix
    counter = 1
    full_suffix = "".join(Path(dest_path.name).suffixes)
    if full_suffix:
        base = dest_path.name[: -len(full_suffix)]
        suffix = full_suffix
    while True:
        candidate = dest_path.with_name(f"{base}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def remove_empty_directories(root_dir: Path) -> None:
    for current_root, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            dir_path = Path(current_root) / dirname
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
            except Exception:
                pass


def main() -> None:
    args = parse_arguments()
    root_dir = Path(args.root).resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(
            f"Root directory does not exist or is not a directory: {root_dir}"
        )

    actions = []
    for file_path in iter_all_files(
        root_dir, include_hidden=args.include_hidden
    ):
        flat_name = make_flat_name(root_dir, file_path, sep=args.sep)
        dest_path = root_dir / flat_name

        if file_path.resolve() == dest_path.resolve():
            continue

        final_dest = ensure_unique_destination(dest_path)

        operation = "COPY" if args.copy else "MOVE"
        actions.append((operation, file_path, final_dest))

    if not actions:
        print("Nothing to do. No files to flatten.")
        return

    for operation, src, dst in actions:
        rel_src = src.relative_to(root_dir)
        rel_dst = dst.relative_to(root_dir)
        if args.dry_run:
            print(f"{operation}: {rel_src} -> {rel_dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if operation == "COPY":
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        print(f"{operation}: {rel_src} -> {rel_dst}")

    if not args.dry_run and not args.keep_dirs:
        remove_empty_directories(root_dir)


if __name__ == "__main__":
    main()
