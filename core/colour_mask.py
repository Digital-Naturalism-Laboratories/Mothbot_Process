"""
colour_mask.py — Quick insect mask from a patch's border colours (no AI model).

The background is every colour that covers at least 10% of the patch border
(near-identical shades merged), plus blends of two such colours (e.g. the soft
edge between a white sheet and the black night behind it). Pixels close to
those colours and connected to the border are background; the rest is insect.

About 1 ms for a small patch and 8 ms for a large one, versus ~6 s for
BiRefNet-lite on CPU. It is rougher: it misses thin legs and antennae and keeps
soft blur halos and shadows. On small or blurry patches that is about as good
as BiRefNet, which tends to break those up into specks.
"""

import cv2
import numpy as np

COLOUR_MASK_METHOD = "border-colour-mask/v1"

_RING = 3           # px of border sampled for background colours
_MIN_SHARE = 0.10   # a colour must cover this share of the border to count as background
_MERGE_DE = 10.0    # border colours closer than this (Lab) count as one
_MAX_COLOURS = 4    # colours looked for in the border
_FILL_MAX = 3       # Detect fills box areas that ran off the photo with black (<= this)


def _ring(a):
    w = _RING
    return np.concatenate([a[:w].reshape(-1, 3), a[-w:].reshape(-1, 3),
                           a[:, :w].reshape(-1, 3), a[:, -w:].reshape(-1, 3)])


def _border_colours(lab):
    """Lab colours that each cover >= _MIN_SHARE of the border ring."""
    ring = _ring(lab).astype(np.float32)
    k = min(_MAX_COLOURS, len(ring))
    cv2.setRNGSeed(0)  # k-means++ seeding: same patch, same mask
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centres = cv2.kmeans(ring, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    groups = []  # merge clusters whose centres are near-identical shades
    for i in np.argsort(-np.bincount(labels, minlength=k)):
        for group in groups:
            if np.linalg.norm(centres[i] - centres[group[0]]) < _MERGE_DE:
                group.append(i)
                break
        else:
            groups.append([i])

    colours = []
    for group in groups:
        members = np.isin(labels, group)
        if members.mean() >= _MIN_SHARE:
            colours.append(np.median(ring[members], axis=0))
    return colours or [np.median(ring, axis=0)]


def _distance_to_blend(lab, a, b):
    """Lab distance of each pixel to the nearest blend of colours a and b."""
    ab = b - a
    t = np.clip(((lab - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(lab - (a + t[..., None] * ab), axis=2)


def insect_mask(patch_bgr):
    """Boolean mask of the patch, True where the insect is."""
    h, w = patch_bgr.shape[:2]
    lab = cv2.cvtColor(patch_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    colours = _border_colours(lab)

    dists = [np.linalg.norm(lab - c, axis=2) for c in colours]
    for i in range(len(colours)):
        for j in range(i + 1, len(colours)):
            dists.append(_distance_to_blend(lab, colours[i], colours[j]))
    dist = cv2.GaussianBlur(np.min(dists, axis=0), (0, 0), 2.0)
    looks_like_background = (dist <= max(4.0, 0.3 * float(np.percentile(dist, 99)))) | (patch_bgr.max(axis=2) <= _FILL_MAX)

    # background = the background-looking regions that touch the border
    _, regions = cv2.connectedComponents(looks_like_background.astype(np.uint8), connectivity=8)
    edge = np.zeros((h, w), bool)
    edge[:_RING] = edge[-_RING:] = True
    edge[:, :_RING] = edge[:, -_RING:] = True
    touching = np.unique(regions[edge & looks_like_background])
    return ~np.isin(regions, touching[touching > 0])
