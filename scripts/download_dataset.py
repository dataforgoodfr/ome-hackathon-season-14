from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASET_ID = "DataForGood/ome-hackathon-season-14"
REMOTE_SPLITS = {
	"train": "data/train-00000-of-00001.parquet",
	"test": "data/test-00000-of-00001.parquet",
}


def download_parquet_split(split: str, data_dir: Path, force: bool) -> Path:
	if split not in REMOTE_SPLITS:
		raise ValueError(f"Unknown split '{split}'. Expected one of: {', '.join(REMOTE_SPLITS)}")

	data_dir.mkdir(parents=True, exist_ok=True)

	remote_rel_path = REMOTE_SPLITS[split]
	remote_uri = f"hf://datasets/{DATASET_ID}/{remote_rel_path}"
	local_path = data_dir / Path(remote_rel_path).name

	if local_path.exists() and not force:
		print(f"[skip] {split}: {local_path} already exists")
		return local_path

	print(f"[download] {split}: {remote_uri}")
	df = pd.read_parquet(remote_uri)

	print(f"[write] {split}: {local_path}")
	df.to_parquet(local_path, index=False)
	return local_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Download OME hackathon dataset parquet files into ./data")
	parser.add_argument(
		"--split",
		choices=["train", "test", "all"],
		default="all",
		help="Which split to download (default: all)",
	)
	parser.add_argument(
		"--data-dir",
		default="data",
		help="Local directory to write parquet files (default: data)",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite existing files",
	)
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	splits = ["train", "test"] if args.split == "all" else [args.split]

	written = [download_parquet_split(s, data_dir=data_dir, force=args.force) for s in splits]
	print("[done] wrote:")
	for p in written:
		print(f"- {p}")


if __name__ == "__main__":
	main()