"""Devices and device types (Automated Insect Monitoring Ontology: Device, Device Type).

``assets/device-types.json`` is a copy of the ontology's built-in device types
(Automated-Insect-Monitoring-Ontology/device-types/device-types.json); update the copy
when that file changes.

The Mothbox firmware writes the device's name, firmware version and body model into
every photo's EXIF, so they are known even when no field sheet is given.
"""

import json
import os
from functools import lru_cache

_ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "device-types.json")

_EXIF_IFD = 0x8769
_TAG_MODEL = 0x0110
_TAG_USER_COMMENT = 0x9286


@lru_cache(maxsize=1)
def load_device_types():
    try:
        with open(_ASSET, "r", encoding="utf-8") as f:
            return json.load(f).get("deviceTypes", [])
    except (OSError, ValueError):
        return []


def device_type_id_for(name):
    """The ``deviceTypeID`` for a device-type name or id ("Mothbox Pro", "mothbox-pro", "Pro"), or None."""
    text = str(name or "").strip().lower()
    if not text:
        return None
    for device_type in load_device_types():
        type_id = str(device_type.get("deviceTypeID", "")).lower()
        type_name = str(device_type.get("deviceType", "")).lower()
        if text in (type_id, type_name):
            return device_type.get("deviceTypeID")
    # Firmware and field-sheet spellings: "Mothbox Pro", "MothboxPro", "pro", "DIY".
    squashed = text.replace(" ", "").replace("-", "").replace("_", "")
    if squashed in ("mothboxpro", "pro", "standardmothboxpro"):
        return "mothbox-pro"
    if squashed in ("mothboxdiy", "diy", "standardmothboxdiy"):
        return "mothbox-diy"
    if squashed in ("ami", "amidevice", "amitrap"):
        return "ami-device"
    return None


def device_from_exif(image_path):
    """``{"device": name, "firmware": version, "deviceType": body}`` from a Mothbox photo's EXIF.

    Missing values are left out; returns ``{}`` for photos without Mothbox EXIF.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            exif = img.getexif()
            model = exif.get(_TAG_MODEL)
            comment_raw = exif.get_ifd(_EXIF_IFD).get(_TAG_USER_COMMENT)
    except Exception:  # noqa: BLE001 — unreadable or non-image file
        return {}

    out = {}
    if isinstance(model, str) and model.strip():
        # "Mothbox Pro + Arducam 64MP OwlSight (OV64A40)" → "Mothbox Pro"
        body = model.split("+")[0].strip()
        if body and body.lower() != "mothbox":
            out["deviceType"] = body

    comment = _decode_user_comment(comment_raw)
    if comment:
        if comment.get("mothbox"):
            out["device"] = str(comment["mothbox"])
        if comment.get("firmware"):
            out["firmware"] = str(comment["firmware"])
        if comment.get("deviceTypeID") and "deviceType" not in out:
            out["deviceType"] = str(comment["deviceTypeID"])
    return out


def _decode_user_comment(raw):
    if raw is None:
        return None
    if isinstance(raw, bytes):
        # EXIF UserComment starts with an 8-byte character-code header ("ASCII\0\0\0").
        text = raw[8:] if raw[:5] in (b"ASCII", b"UNICO", b"JIS\x00\x00") or raw[:8] == b"\x00" * 8 else raw
        try:
            text = text.decode("utf-8", errors="ignore")
        except AttributeError:
            return None
    else:
        text = str(raw)
    text = text.strip("\x00 ").strip()
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
