"""Trace each layer's alpha silhouette into simplified polygon boundaries.

Mirrors the OpenCV recipe used for tracing PNG cutouts directly against a
PSD layer's already-decoded RGBA pixels (no round-trip through a PNG file
needed, since psd_reader already gives us the cropped alpha channel in
memory):

    alpha channel -> binary threshold -> cv2.findContours -> cv2.approxPolyDP

`findContours` uses `RETR_EXTERNAL`, so only outer silhouette boundaries
are traced -- a layer with a literal hole in it (e.g. a ring) will come
back as a solid shape, matching the scope of the traced-outline approach
this replaces the plain bounding box with. Multiple disconnected opaque
regions in one layer (e.g. two separate rocks) come back as separate
shells; `mesh_builder` turns each into its own mesh.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .scene_model import SourceLayer

Point = Tuple[float, float]


def _drop_collinear_points(points: List[Point], eps: float = 1e-6) -> List[Point]:
    """Remove boundary points that sit on the line between their neighbors.

    `approxPolyDP` simplifies against the whole arc it's recursively
    splitting, so it can still leave a point that's collinear with its
    immediate neighbors (common along a long, nearly-straight edge with a
    lot of jittery micro-detail nearby). A "corner" with zero turning angle
    isn't a real corner, and feeding it to ear-clipping produces a
    zero-area sliver triangle -- pruning it here is both cheaper and more
    correct than special-casing degenerate ears downstream.
    """
    if len(points) <= 3:
        return points
    pts = list(points)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        n = len(pts)
        kept = []
        for i in range(n):
            a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(cross) <= eps:
                changed = True
                continue
            kept.append(b)
        pts = kept if len(kept) >= 3 else pts
    return pts


def trace_layer_contours(
    layer: SourceLayer,
    alpha_threshold: int = 10,
    detail_level: float = 2.0,
    min_contour_area: float = 50.0,
) -> List[List[Point]]:
    """Return one closed polygon (list of (px, py) pixel points) per opaque shell.

    `alpha_threshold` is the raw 0-255 alpha cutoff below which a pixel is
    treated as background. `detail_level` is the Douglas-Peucker epsilon in
    pixels passed to `cv2.approxPolyDP` -- lower hugs the silhouette more
    tightly at the cost of more vertices, higher simplifies more
    aggressively (same knob and semantics as manually tracing a PNG).
    """
    alpha = np.asarray(layer.pixels)[:, :, 3]
    _, binary = cv2.threshold(alpha, alpha_threshold, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shells: List[List[Point]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        approx = cv2.approxPolyDP(contour, detail_level, True)
        points = [(float(p[0][0]), float(p[0][1])) for p in approx]
        points = _drop_collinear_points(points)
        if len(points) >= 3:
            shells.append(points)

    return shells
