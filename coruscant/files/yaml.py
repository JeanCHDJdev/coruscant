"""Small YAML file helpers shared across Coruscant consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _yaml_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _yaml_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe_value(item) for item in value]
    return value


def read_yaml_file(path: str | Path) -> dict[str, Any]:
    """Read one YAML file and return its root mapping.

    Args:
            path: Path to the YAML file to read.

    Returns:
            The YAML payload as a plain dictionary.

    Raises:
            TypeError: If the YAML document root is not a mapping.
    """
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping at the root of {yaml_path}, got {type(payload).__name__}.")
    return dict(payload)


def write_yaml_file(
    path: str | Path,
    payload: dict[str, Any],
    *,
    sort_keys: bool = False,
) -> Path:
    """Write one YAML mapping to disk.

    Args:
            path: Output path for the YAML file.
            payload: Mapping to serialize.
            sort_keys: Whether to sort mapping keys while dumping YAML.

    Returns:
            The resolved path written to disk.
    """
    yaml_path = Path(path).expanduser().resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(_yaml_safe_value(payload), sort_keys=sort_keys),
        encoding="utf-8",
    )
    return yaml_path
