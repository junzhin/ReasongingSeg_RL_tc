#!/usr/bin/env python3
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List


OLD_ROOT = "/mnt/petrelfs/tangcheng/tangcheng/research/ReasongingSeg/data/"
NEW_ROOT = "/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/data/"


@dataclass
class FileReport:
    path: str
    records: int
    changed_records: int
    image_refs: int
    rewritten_refs: int
    existing_refs: int
    missing_refs: int
    backup_path: str


def iter_jsonl_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.jsonl") if p.is_file())


def rewrite_image_path(path_str: str) -> tuple[str, bool]:
    if path_str.startswith(OLD_ROOT):
        new_path = NEW_ROOT + path_str[len(OLD_ROOT) :]
        return new_path, new_path != path_str
    return path_str, False


def process_file(path: Path, backup_suffix: str) -> tuple[FileReport, list[dict]]:
    records = 0
    changed_records = 0
    image_refs = 0
    rewritten_refs = 0
    existing_refs = 0
    missing_items: list[dict] = []
    output_lines: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            records += 1
            obj = json.loads(line)
            original_images = obj.get("images", [])
            new_images = []
            record_changed = False

            for img_path in original_images:
                image_refs += 1
                new_path, rewritten = rewrite_image_path(img_path)
                if rewritten:
                    rewritten_refs += 1
                    record_changed = True
                if Path(new_path).exists():
                    existing_refs += 1
                else:
                    missing_items.append(
                        {
                            "jsonl": str(path),
                            "line": line_no,
                            "old_path": img_path,
                            "new_path": new_path,
                        }
                    )
                new_images.append(new_path)

            if record_changed:
                changed_records += 1
                obj["images"] = new_images

            output_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

    backup_path = path.with_name(path.name + backup_suffix)
    path.replace(backup_path)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(output_lines)

    report = FileReport(
        path=str(path),
        records=records,
        changed_records=changed_records,
        image_refs=image_refs,
        rewritten_refs=rewritten_refs,
        existing_refs=existing_refs,
        missing_refs=len(missing_items),
        backup_path=str(backup_path),
    )
    return report, missing_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/data/rl_data_final",
        help="Root directory containing JSONL files to fix.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4)),
        help="Number of worker threads used for file processing.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for generated reports. Defaults to <root>/reports.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = iter_jsonl_files(root)
    if not jsonl_files:
        raise SystemExit(f"No JSONL files found under {root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_suffix = f".bak_{timestamp}"

    file_reports: list[FileReport] = []
    missing_items: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, path, backup_suffix) for path in jsonl_files]
        for future in futures:
            report, missing = future.result()
            file_reports.append(report)
            missing_items.extend(missing)

    file_reports.sort(key=lambda item: item.path)
    total_records = sum(item.records for item in file_reports)
    total_changed_records = sum(item.changed_records for item in file_reports)
    total_image_refs = sum(item.image_refs for item in file_reports)
    total_rewritten_refs = sum(item.rewritten_refs for item in file_reports)
    total_existing_refs = sum(item.existing_refs for item in file_reports)

    summary = {
        "timestamp": timestamp,
        "root": str(root),
        "old_root": OLD_ROOT,
        "new_root": NEW_ROOT,
        "workers": args.workers,
        "jsonl_files": len(file_reports),
        "records": total_records,
        "changed_records": total_changed_records,
        "image_refs": total_image_refs,
        "rewritten_refs": total_rewritten_refs,
        "existing_refs": total_existing_refs,
        "missing_refs": len(missing_items),
        "all_images_exist": len(missing_items) == 0,
        "files": [asdict(item) for item in file_reports],
        "missing_items": missing_items,
    }

    summary_path = report_dir / f"rl_data_image_path_fix_report_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    text_path = report_dir / f"rl_data_image_path_fix_report_{timestamp}.txt"
    with text_path.open("w", encoding="utf-8") as f:
        f.write(f"root: {root}\n")
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"workers: {args.workers}\n")
        f.write(f"jsonl_files: {len(file_reports)}\n")
        f.write(f"records: {total_records}\n")
        f.write(f"changed_records: {total_changed_records}\n")
        f.write(f"image_refs: {total_image_refs}\n")
        f.write(f"rewritten_refs: {total_rewritten_refs}\n")
        f.write(f"existing_refs: {total_existing_refs}\n")
        f.write(f"missing_refs: {len(missing_items)}\n")
        f.write(f"all_images_exist: {len(missing_items) == 0}\n")
        f.write("\nPer-file summary:\n")
        for item in file_reports:
            f.write(
                f"- {item.path}: records={item.records}, changed_records={item.changed_records}, "
                f"rewritten_refs={item.rewritten_refs}, missing_refs={item.missing_refs}, "
                f"backup={item.backup_path}\n"
            )
        if missing_items:
            f.write("\nMissing items:\n")
            for item in missing_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON report: {summary_path}")
    print(f"Text report: {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
