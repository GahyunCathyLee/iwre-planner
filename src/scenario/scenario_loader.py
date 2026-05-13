from pathlib import Path
from typing import Optional

from scenario.random_scenario_generator import generate_random_scenario_dict
from scenario.scenario_config import ScenarioConfig, scenario_from_dict


def load_scenario(path: str, seed_override: Optional[int] = None) -> ScenarioConfig:
    scenario_path = Path(path)
    data = _load_yaml(scenario_path)

    if data.get("scenario_name") == "random_batch" or data.get("type") == "random_batch":
        data = generate_random_scenario_dict(data, seed_override)

    if seed_override is not None:
        data["seed"] = int(seed_override)

    return scenario_from_dict(data)


def _load_yaml(path: Path):
    try:
        import yaml

        with path.open("r") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        with path.open("r") as f:
            return _parse_simple_yaml(f.read())


def _parse_simple_yaml(text: str):
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    value, _ = _parse_block(lines, 0, 0)
    return value


def _parse_block(lines, index, indent):
    if index >= len(lines):
        return {}, index
    is_list = lines[index][1].startswith("- ")
    result = [] if is_list else {}

    while index < len(lines):
        line_indent, stripped = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            break

        if is_list:
            if not stripped.startswith("- "):
                break
            item_text = stripped[2:].strip()
            item = {}
            if item_text:
                key, value = _parse_key_value(item_text)
                if key is None:
                    item = _parse_value(item_text)
                elif value == "":
                    child, index = _parse_block(lines, index + 1, indent + 2)
                    item[key] = child
                    result.append(item)
                    continue
                else:
                    item[key] = _parse_value(value)
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_block(lines, index, indent + 2)
                if isinstance(item, dict) and isinstance(child, dict):
                    item.update(child)
                else:
                    item = child
            result.append(item)
        else:
            key, value = _parse_key_value(stripped)
            if key is None:
                index += 1
                continue
            if value == "":
                child, index = _parse_block(lines, index + 1, indent + 2)
                result[key] = child
            else:
                result[key] = _parse_value(value)
                index += 1

    return result, index


def _parse_key_value(text):
    if ":" not in text:
        return None, text
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_value(value):
    value = value.strip()
    if value == "":
        return ""
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None", "~"):
        return None
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1].strip()
        if not body:
            return {}
        parsed = {}
        for part in _split_inline(body):
            key, item_value = _parse_key_value(part)
            parsed[key] = _parse_value(item_value)
        return parsed
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_value(part) for part in _split_inline(body)]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_inline(body):
    parts = []
    current = []
    depth = 0
    quote = None
    for ch in body:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in ("{", "["):
            depth += 1
        elif ch in ("}", "]"):
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts
