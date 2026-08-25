import hashlib
import json
import shutil
import sys
from pathlib import Path

SOURCE_DIRS = ("analyses", "macros", "models", "seeds")
SOURCE_SUFFIXES = {".csv", ".py", ".sql", ".yaml", ".yml"}


def snapshot(project: Path, backup: Path | None = None) -> dict[str, str]:
    files = []
    for name in SOURCE_DIRS:
        root = project / name
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())

    project_file = project / "dbt_project.yml"
    if project_file.is_file():
        files.append(project_file)

    hashes = {}
    for path in files:
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(project)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if backup:
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return hashes


if __name__ == "__main__":
    project_path = Path(sys.argv[1])
    backup_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(json.dumps(snapshot(project_path, backup_path), sort_keys=True))
