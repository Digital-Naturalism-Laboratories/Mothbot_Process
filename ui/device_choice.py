"""Choosing the device type for the data being processed (Setup and Insert Metadata tabs).

Automated Insect Monitoring Ontology: Device, Device Type. Photos from current Mothbox
firmware say which device took them (the type chosen in the settings editor, with its
attractors, targets and sensors). By default that is what each detection file gets, and
photos that don't say get no device type. A person can instead choose a standard device
type, one found in the photos, or a custom device made here (like Classify's Devices
editor). Custom devices are remembered in ``~/.mothbot/device-types.ndjson``.

Everything here is plain Python (no Gradio), so it can be tested on its own.
"""

import json
import os
import re
from pathlib import Path

from core.device_info import device_from_exif, load_device_choices, load_device_types

PHOTOS_LABEL = "Use what the photos say"
IMAGE_EXTENSIONS = (".jpg", ".jpeg")
MAX_PHOTOS_TO_CHECK = 25
CUSTOM_TYPES_FILE = Path.home() / ".mothbot" / "device-types.ndjson"

ATTRACTOR_COLUMNS = ["Type", "Hardware version", "Location", "Switches on", "Power (W)"]
TARGET_COLUMNS = ["Type", "Material", "Color", "Width (cm)", "Height (cm)", "Distance (cm)"]
SENSOR_COLUMNS = ["Type", "Model", "Lens", "Resolution", "Notes"]
POWER_COLUMNS = ["Type", "Model", "Capacity (Wh)", "Nominal (V)", "At 80% full (V)", "At 20% full (V)"]
BATTERY_NUMBERS = ("capacityWh", "nominalVoltageV", "voltageAt80PercentV", "voltageAt20PercentV")


# ── the choices ────────────────────────────────────────────────────────────────

def standard_choices():
    """{label: definition} for the built-in device types."""
    return {t["deviceType"]: t for t in load_device_types()}


def saved_custom_types(path=None):
    """Custom devices made in Process before (newest wins for an id)."""
    path = Path(path or CUSTOM_TYPES_FILE)
    out = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue  # skip a damaged line rather than lose the rest
        if isinstance(item, dict) and item.get("deviceTypeID") and item.get("deviceType"):
            out[item["deviceTypeID"]] = item
    return list(out.values())


def save_custom_type(definition, path=None):
    """Adds or replaces (by id) a custom device in the remembered list."""
    path = Path(path or CUSTOM_TYPES_FILE)
    kept = [t for t in saved_custom_types(path) if t["deviceTypeID"] != definition["deviceTypeID"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".ndjson.tmp")
    tmp.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in [*kept, definition]), encoding="utf-8")
    os.replace(tmp, path)


def custom_label(definition):
    return f"{definition.get('deviceType') or 'Custom device'} (custom)"


def all_choices(found=None, custom_path=None, current=None):
    """{label: definition}: standard types, remembered custom devices, types found in the photos,
    and the current choice if it is none of those (e.g. found in folders no longer chosen)."""
    choices = dict(standard_choices())
    known_ids = {d["deviceTypeID"] for d in choices.values()}
    for t in saved_custom_types(custom_path):
        if t["deviceTypeID"] not in known_ids:
            choices[custom_label(t)] = t
            known_ids.add(t["deviceTypeID"])
    for label, definition in (found or {}).items():
        if definition and definition.get("deviceTypeID") not in known_ids:
            choices[f"{label} (from the photos)"] = definition
            known_ids.add(definition.get("deviceTypeID"))
    if current and current.get("deviceTypeID") not in known_ids:
        choices[f"{current.get('deviceType') or 'Device'} (chosen)"] = current
    return choices


def choice_labels(found=None, custom_path=None, current=None):
    return [PHOTOS_LABEL, *all_choices(found, custom_path, current)]


def chosen_definition(label, found=None, custom_path=None, current=None):
    """The device description for a dropdown label; None means use what the photos say."""
    if not label or label == PHOTOS_LABEL:
        return None
    return all_choices(found, custom_path, current).get(label)


def label_for(definition, found=None, custom_path=None):
    """The dropdown label showing this definition (PHOTOS_LABEL for none)."""
    if not definition:
        return PHOTOS_LABEL
    for label, d in all_choices(found, custom_path, definition).items():
        if d.get("deviceTypeID") == definition.get("deviceTypeID"):
            return label
    return PHOTOS_LABEL


# ── reading the photos ─────────────────────────────────────────────────────────

def _folder_of(entry):
    return entry["path"] if isinstance(entry, dict) else entry


def _sample_photos(folders, limit=MAX_PHOTOS_TO_CHECK):
    """A few full-size photos from the chosen folders (skips _processed and patches)."""
    found = []
    for entry in folders or []:
        folder = _folder_of(entry)
        if not folder or not os.path.isdir(folder):
            continue
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d not in ("_processed", "patches") and not d.startswith(".")]
            for name in sorted(files):
                if name.lower().endswith(IMAGE_EXTENSIONS) and "_nobg" not in name:
                    found.append(os.path.join(root, name))
                    break  # one per folder is plenty
            if len(found) >= limit:
                return found
    return found


def describe(definition):
    """One line about a device description: attractors, targets, sensors, power."""
    units = {"powerW": " W", "capacityWh": " Wh"}

    def items(key, parts):
        out = []
        for item in definition.get(key) or []:
            text = " ".join(f"{item.get(p)}{units.get(p, '')}" for p in parts if item.get(p) not in (None, ""))
            out.append(text or "details not filled in")
        return "; ".join(out) if out else "none"

    text = (f"attractors: {items('attractors', ('attractorType', 'hardwareVersion', 'powerW'))} · "
            f"targets: {items('targets', ('targetType', 'color', 'material'))} · "
            f"sensors: {items('sensors', ('sensorType', 'model', 'resolution'))}")
    if definition.get("powerSources"):
        text += f" · power: {items('powerSources', ('powerSourceType', 'model', 'capacityWh'))}"
    if definition.get("description"):
        text += f" · {definition['description']}"
    return text


def scan_photos(selected_folders):
    """Reads a few of the chosen folders' photos. Returns (markdown, {label: definition} found)."""
    photos = _sample_photos(selected_folders)
    if not photos:
        return ("No photos found in the chosen folders yet.", {})

    found = {}
    names, firmwares = set(), set()
    for photo in photos:
        info = device_from_exif(photo)
        if info.get("device"):
            names.add(info["device"])
        if info.get("firmware"):
            firmwares.add(info["firmware"])
        definition = info.get("deviceDefinition")
        label = (definition or {}).get("deviceType") or info.get("deviceType")
        if label:
            found.setdefault(label, definition)

    lines = [f"Checked {len(photos)} photo{'s' if len(photos) != 1 else ''}."]
    if names:
        lines.append(f"**Mothbox:** {', '.join(sorted(names))}")
    if firmwares:
        lines.append(f"**Firmware:** {', '.join(sorted(firmwares))}")
    if found:
        for label, definition in found.items():
            lines.append(f"**Device type in the photos:** {label}" + (f" — {describe(definition)}" if definition else ""))
        if len(found) > 1:
            lines.append("⚠️ The photos name more than one device type; choose the right one if they are all the same device.")
    else:
        lines.append("The photos don't say which device took them (older firmware, or another camera). "
                     "Unless you choose one, these detections get no device type.")
    return "\n\n".join(lines), {k: v for k, v in found.items() if v}


def summary(state, photos):
    """What the device section says: what the photos say, then what this run will use."""
    definition = (state or {}).get("definition")
    lines = [(photos or {}).get("text") or "Choose folders to see what their photos say."]
    if definition:
        lines.append(f"**Used for this run:** {definition.get('deviceType')} — {describe(definition)}")
    else:
        lines.append("**Used for this run:** what each photo says (nothing for photos that don't say).")
    return "\n\n".join(lines)


# ── the custom device editor ───────────────────────────────────────────────────

def editor_hint():
    """Suggested values, shown above the editor's tables."""
    c = load_device_choices()
    mothbeam = next(iter((c.get("attractorPowerOptions") or {}).values()), None)
    lights = ", ".join(o["configuration"] for o in (mothbeam or {}).get("options", []))
    parts = [
        f"**Attractor types:** {', '.join(c.get('attractorTypes', []))}",
        f"**Locations:** {', '.join(c.get('attractorLocations', []))}",
    ]
    if lights:
        parts.append(f"**Mothbeam “Switches on”** (the power fills in): {lights}")
    parts += [
        f"**Target materials:** {', '.join(c.get('targetMaterials', []))} · **colors:** {', '.join(c.get('targetColors', []))}",
        f"**Sensor types:** {', '.join(c.get('sensorTypes', []))}",
        f"**Power sources:** {', '.join(c.get('powerSourceTypes', []))} · **batteries:** "
        + ", ".join(f"{b['model']} ({b.get('capacityWh')} Wh, {b.get('nominalVoltageV')} V; "
                    f"{b.get('voltageAt80PercentV')} V at 80%, {b.get('voltageAt20PercentV')} V at 20%)"
                    for b in c.get("batteries", [])),
        "Any other value is fine too. Right-click a table to add or delete rows; empty rows are ignored.",
    ]
    return "\n\n".join(parts)


def _cell(value):
    return "" if value is None else value


def to_editor(definition):
    """The editor's fields and table rows for a device description."""
    d = definition or {}
    return {
        "name": d.get("deviceType") or "",
        "maker": d.get("maker") or "",
        "hardwareVersion": d.get("hardwareVersion") or "",
        "firmwareVersion": (d.get("firmware") or {}).get("version") or "",
        "computer": d.get("computer") or "",
        "description": d.get("description") or "",
        "attractors": [[_cell(a.get(k)) for k in ("attractorType", "hardwareVersion", "attractorLocation", "configuration", "powerW")]
                       for a in d.get("attractors") or []],
        "targets": [[_cell(t.get(k)) for k in ("targetType", "material", "color", "widthCm", "heightCm", "distanceCm")]
                    for t in d.get("targets") or []],
        "sensors": [[_cell(s.get(k)) for k in ("sensorType", "model", "lens", "resolution", "notes")]
                    for s in d.get("sensors") or []],
        "powerSources": [[_cell(p.get(k)) for k in ("powerSourceType", "model", *BATTERY_NUMBERS)]
                         for p in d.get("powerSources") or []],
    }


def _rows(table):
    """Rows from a Gradio table value (list of lists, or a pandas DataFrame), blank rows dropped."""
    if table is None:
        return []
    if hasattr(table, "values") and hasattr(table, "columns"):
        table = table.values.tolist()
    out = []
    for row in table:
        cells = ["" if v is None or (isinstance(v, float) and v != v) else v for v in row]
        if any(str(v).strip() for v in cells):
            out.append(cells)
    return out


def _text(value):
    text = str(value).strip() if value is not None else ""
    return text or None


def _number(value, what, problems):
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{what} must be a number (got “{value}”).")
        return None
    if number != number or number < 0:
        problems.append(f"{what} must be 0 or more.")
        return None
    return int(number) if number.is_integer() else number


def _power_options(attractor_type):
    wanted = (attractor_type or "").strip().lower()
    for key, value in (load_device_choices().get("attractorPowerOptions") or {}).items():
        if key.lower() == wanted:
            return value
    return None


def _same_setting(a, b):
    """“1 and 2”, “1 and 2 switches” and “1 And 2 Switch” are the same switch setting."""
    squash = lambda s: re.sub(r"\s*switch(es)?\s*$", "", str(s or "").strip().lower())
    return bool(b) and squash(a) == squash(b)


def custom_id_for(name):
    return "custom-" + (re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-") or "device")


def from_editor(base, name, maker, hardware_version, firmware_version, computer, description,
                attractors, targets, sensors, power_sources):
    """A custom device description from the editor, and the problems that stop it being used.

    ``base`` is the device it was customized from: its firmware family, channels and other
    fields not in the editor are kept.
    """
    base = base or {}
    problems = []
    name = (name or "").strip()
    if not name:
        problems.append("Give the device a name.")

    base_attractors = base.get("attractors") or []
    out_attractors = []
    for i, row in enumerate(_rows(attractors)):
        kind, hw, location, lights, power = (row + [""] * 5)[:5]
        item = {
            "channel": base_attractors[i].get("channel") if i < len(base_attractors) else None,
            "attractorType": _text(kind),
            "hardwareVersion": _text(hw),
            "attractorLocation": _text(location),
            "configuration": _text(lights),
            "powerW": _number(power, "Attractor power", problems),
        }
        options = _power_options(item["attractorType"])
        if options:
            if not item["hardwareVersion"]:
                item["hardwareVersion"] = options.get("hardwareVersion")
            match = next((o for o in options.get("options", []) if _same_setting(o["configuration"], item["configuration"])), None)
            if match:
                item["configuration"] = match["configuration"]  # "1 and 2" → "1 and 2 switches"
            if item["configuration"] and not match:
                problems.append(f"“{item['configuration']}” isn't one of the {item['attractorType']} light settings.")
            if match and item["powerW"] is None:
                item["powerW"] = match["powerW"]
        out_attractors.append(item)

    out_targets = []
    for row in _rows(targets):
        kind, material, color, width, height, distance = (row + [""] * 6)[:6]
        out_targets.append({
            "targetType": _text(kind), "material": _text(material), "color": _text(color),
            "widthCm": _number(width, "Target width", problems),
            "heightCm": _number(height, "Target height", problems),
            "distanceCm": _number(distance, "Target distance", problems),
        })

    out_sensors = []
    for row in _rows(sensors):
        kind, model, lens, resolution, notes = (row + [""] * 5)[:5]
        out_sensors.append({
            "sensorType": _text(kind) or "camera", "model": _text(model), "lens": _text(lens),
            "resolution": _text(resolution), "notes": _text(notes),
        })
    if not out_sensors:
        problems.append("A device needs at least one sensor.")

    batteries = {b["model"].lower(): b for b in load_device_choices().get("batteries", [])}
    out_power = []
    for row in _rows(power_sources):
        kind, model, capacity, nominal, at80, at20 = (row + [""] * 6)[:6]
        item = {"powerSourceType": (_text(kind) or "").lower() or None, "model": _text(model),
                "capacityWh": _number(capacity, "Battery capacity", problems),
                "nominalVoltageV": _number(nominal, "Battery nominal voltage", problems),
                "voltageAt80PercentV": _number(at80, "Battery voltage at 80%", problems),
                "voltageAt20PercentV": _number(at20, "Battery voltage at 20%", problems)}
        if not item["powerSourceType"]:
            problems.append("Every power source needs a type (battery, grid or solar).")
        # A known battery fills in its capacity and voltages (where they're empty).
        known = batteries.get((item["model"] or "").lower())
        for key in BATTERY_NUMBERS:
            if known and item[key] is None and known.get(key) is not None:
                item[key] = known[key]
        if item["powerSourceType"] != "battery":
            item = {k: v for k, v in item.items() if k not in BATTERY_NUMBERS or v is not None}
        elif (item["voltageAt80PercentV"] is not None and item["voltageAt20PercentV"] is not None
              and item["voltageAt80PercentV"] <= item["voltageAt20PercentV"]):
            problems.append("The battery's 80% voltage has to be higher than its 20% voltage.")
        out_power.append(item)

    standard_ids = {t["deviceTypeID"] for t in load_device_types()}
    definition = {
        **{k: v for k, v in base.items() if k not in ("deviceTypeID", "deviceType", "derivedFrom")},
        "deviceTypeID": custom_id_for(name),
        "deviceType": name,
        "derivedFrom": base.get("deviceTypeID") if base.get("deviceTypeID") in standard_ids else base.get("derivedFrom"),
        "maker": _text(maker),
        "hardwareVersion": _text(hardware_version),
        "firmware": {**(base.get("firmware") or {}), "version": _text(firmware_version)},
        "attractors": out_attractors,
        "targets": out_targets,
        "sensors": out_sensors,
        "powerSources": out_power,
        "computer": _text(computer),
        "description": _text(description),
    }
    if name and definition["deviceType"] in {t["deviceType"] for t in load_device_types()}:
        problems.append("That name belongs to a standard device; give your custom device its own name.")
    return definition, problems
