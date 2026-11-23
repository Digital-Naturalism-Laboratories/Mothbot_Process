"""Box formats (Automated Insect Monitoring Ontology: Box Formats).

Every box Mothbot writes should say what format it is in. The Mothbox format for an
oriented box is four relative corner points, ``obb-corners-rel``:
``[x1, y1, x2, y2, x3, y3, x4, y4]``, each a fraction (0–1) of the image's width or
height, listed in order around the box. There is no angle, so there are no questions
about units or direction.

X-AnyLabeling files keep their own pixel format (``xanylabeling-rotation-px``: four
corner ``points`` in pixels plus a ``direction`` angle) so they stay editable there.
Keep the conversions in sync with Mothbot Classify's ``src/models/box-formats.ts``.
"""

import math

OBB_CORNERS_REL = "obb-corners-rel"
XANYLABELING_ROTATION_PX = "xanylabeling-rotation-px"


def corners_rel(points, image_width, image_height, digits=6):
    """Four pixel corners [[x, y] × 4] → flat relative corners, or None if unusable."""
    try:
        w = float(image_width)
        h = float(image_height)
        pts = [(float(x), float(y)) for x, y in points]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0 or len(pts) != 4:
        return None
    out = []
    for x, y in pts:
        out.append(round(x / w, digits))
        out.append(round(y / h, digits))
    return out


def labelled_box(points, image_width, image_height):
    """``{"format": "obb-corners-rel", "coordinates": [...]}`` for a shape, or None."""
    coords = corners_rel(points, image_width, image_height)
    return {"format": OBB_CORNERS_REL, "coordinates": coords} if coords else None


def direction_from_points(points):
    """X-AnyLabeling's ``direction``: the angle of the first edge (corner 0 → corner 1),
    in radians, from 0 up to 2π, measured in image coordinates (y pointing down).

    X-AnyLabeling starts a drawn rectangle at direction 0 with that edge pointing right,
    and adds each rotation to it, so this is the value it would have stored.
    """
    try:
        (x0, y0), (x1, y1) = points[0], points[1]
        angle = math.atan2(float(y1) - float(y0), float(x1) - float(x0))
    except (TypeError, ValueError, IndexError):
        return 0.0
    return angle % (2 * math.pi)
