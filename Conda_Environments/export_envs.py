"""Export repo Conda environment YAMLs while stripping local-only packages."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ENV_FILES = {
    "4dhoi": "4dhoi.yml",
    "gvhmr": "gvhmr.yml",
    "sam3": "sam3.yml",
    "sam3d-objects": "sam3d-objects.yml",
    "waft": "waft.yml",
}

DEFAULT_EXCLUDED_PIP_PACKAGES = {"sam-2", "sam2"}


def normalize_requirement_name(requirement: str) -> str:
    requirement = requirement.strip()
    if requirement.startswith("-e "):
        editable_target = requirement[3:].strip().rstrip("/")
        return Path(editable_target).name.lower().replace("_", "-").replace(".", "-")

    requirement = requirement.split(";", 1)[0].strip()
    if " @ " in requirement:
        requirement = requirement.split(" @ ", 1)[0].strip()
    else:
        requirement = re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip()

    return requirement.lower().replace("_", "-").replace(".", "-")


def should_exclude_requirement(requirement: str, excluded_packages: set[str]) -> bool:
    normalized_name = normalize_requirement_name(requirement)
    if normalized_name in excluded_packages:
        return True

    lowered = requirement.lower()
    return "/sam2" in lowered or lowered.rstrip("/").endswith("sam2")


def sanitize_exported_yaml(
    exported_yaml: str,
    excluded_packages: set[str],
    drop_prefix: bool,
) -> str:
    sanitized_lines: list[str] = []
    in_pip_block = False

    for line in exported_yaml.splitlines():
        if line.startswith("prefix: ") and drop_prefix:
            continue

        if line.startswith("  - pip:"):
            in_pip_block = True
            sanitized_lines.append(line)
            continue

        if in_pip_block:
            if line.startswith("      - "):
                requirement = line.strip()[2:].strip()
                if should_exclude_requirement(requirement, excluded_packages):
                    continue
                sanitized_lines.append(line)
                continue

            in_pip_block = False

        sanitized_lines.append(line)

    return "\n".join(sanitized_lines) + "\n"


def export_environment(
    env_name: str,
    output_path: Path,
    excluded_packages: set[str],
    drop_prefix: bool,
) -> None:
    result = subprocess.run(
        ["conda", "env", "export", "--name", env_name],
        check=True,
        capture_output=True,
        text=True,
    )
    sanitized_yaml = sanitize_exported_yaml(
        exported_yaml=result.stdout,
        excluded_packages=excluded_packages,
        drop_prefix=drop_prefix,
    )
    output_path.write_text(sanitized_yaml, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export repository Conda environment YAMLs while stripping local-only "
            "pip packages such as sam-2."
        )
    )
    parser.add_argument(
        "--env",
        action="append",
        dest="env_names",
        choices=sorted(DEFAULT_ENV_FILES),
        help="Export only the given environment. Repeat to export multiple.",
    )
    parser.add_argument(
        "--exclude-pip-package",
        action="append",
        default=[],
        help=(
            "Additional pip package name to remove from exported YAMLs. "
            "Repeat to exclude multiple packages."
        ),
    )
    parser.add_argument(
        "--drop-prefix",
        action="store_true",
        help="Remove the absolute prefix line from exported YAMLs.",
    )
    return parser.parse_args()


def main() -> int:
    if shutil.which("conda") is None:
        print("Error: `conda` was not found on PATH.", file=sys.stderr)
        return 1

    args = parse_args()
    env_names = args.env_names or list(DEFAULT_ENV_FILES)
    excluded_packages = {
        package.lower().replace("_", "-").replace(".", "-")
        for package in DEFAULT_EXCLUDED_PIP_PACKAGES
    }
    excluded_packages.update(
        package.lower().replace("_", "-").replace(".", "-")
        for package in args.exclude_pip_package
    )

    script_dir = Path(__file__).resolve().parent
    for env_name in env_names:
        output_path = script_dir / DEFAULT_ENV_FILES[env_name]
        export_environment(
            env_name=env_name,
            output_path=output_path,
            excluded_packages=excluded_packages,
            drop_prefix=args.drop_prefix,
        )
        print(f"Exported {env_name} -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
