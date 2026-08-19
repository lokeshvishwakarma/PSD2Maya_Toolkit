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

import logging
from typing import List, Tuple

import cv2
import numpy as np

from .scene_model import SourceLayer

logger = logging.getLogger(__name__)

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


def _polygon_area(points: List[Point]) -> float:
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _orient(a: Point, b: Point, c: Point) -> int:
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(v) <= 1e-12:
        return 0
    return 1 if v > 0 else -1


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
    )


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    if d1 and d2 and d3 and d4 and (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0):
        return True
    return (
        (d1 == 0 and _on_segment(p3, p4, p1))
        or (d2 == 0 and _on_segment(p3, p4, p2))
        or (d3 == 0 and _on_segment(p1, p2, p3))
        or (d4 == 0 and _on_segment(p1, p2, p4))
    )


def _intersection_point(p1: Point, p2: Point, p3: Point, p4: Point, fallback: Point) -> Point:
    """Where segments p1p2 and p3p4 meet, or `fallback` if they're parallel/collinear."""
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (p4[0] - p3[0], p4[1] - p3[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) <= 1e-12:
        return fallback
    t = ((p3[0] - p1[0]) * s[1] - (p3[1] - p1[1]) * s[0]) / denom
    return (p1[0] + t * r[0], p1[1] + t * r[1])


def _find_self_intersection(points: List[Point]):
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue  # adjacent edges legitimately share an endpoint
            c, d = points[j], points[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return i, j, _intersection_point(a, b, c, d, points[j])
    return None


def _dedupe_consecutive(points: List[Point], eps: float = 1e-9) -> List[Point]:
    out: List[Point] = []
    for p in points:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    while len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def _remove_self_intersections(points: List[Point], max_passes: int = 64) -> List[Point]:
    """Split off and discard self-crossing sliver loops until the polygon is simple.

    `cv2.findContours` traces the pixel boundary, and where the artwork has a
    roughly one-pixel-wide neck or spur the traced path runs out and back
    within a pixel of itself, so the contour touches or crosses itself. This
    is *not* an artifact of `approxPolyDP` -- it's present in the raw contour
    too, so lowering `detail_level` doesn't help (verified: a spur still
    crosses at epsilon 0).

    Ear-clipping a self-crossing boundary emits faces that overlap, which
    makes the surface locally non-orientable: two same-winding faces end up
    sharing a directed edge. Maya reports that as nonmanifold geometry and
    `polyRetopo` refuses to run on it ("does not work on polygonal object
    with nonmanifold geometry").

    At each crossing the boundary is cut into two closed loops; the larger
    one is kept and the sliver is dropped. Crossings observed in practice are
    between near-adjacent edges, so the discarded loop is a handful of
    vertices enclosing near-zero area -- the visible silhouette is unchanged.
    """
    pts = _dedupe_consecutive(points)
    for _ in range(max_passes):
        if len(pts) < 4:
            return pts
        hit = _find_self_intersection(pts)
        if hit is None:
            return pts
        i, j, cut = hit
        # The crossing splits the ring into pts[i+1..j] and pts[j+1..i], each
        # closed through the intersection point. Keep whichever encloses more
        # area; the other is the degenerate sliver.
        loop_a = _dedupe_consecutive([cut] + pts[i + 1 : j + 1])
        loop_b = _dedupe_consecutive(pts[: i + 1] + [cut] + pts[j + 1 :])
        candidates = [p for p in (loop_a, loop_b) if len(p) >= 3]
        if not candidates:
            return pts
        best = max(candidates, key=_polygon_area)
        if len(best) >= len(pts):
            return pts  # no progress; bail out rather than loop forever
        pts = best
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
        before = len(points)
        points = _remove_self_intersections(points)
        if len(points) != before:
            logger.info(
                "Layer %r: removed %d vertex/vertices forming self-crossing sliver loop(s) "
                "from the traced boundary",
                layer.name,
                before - len(points),
            )
        points = _drop_collinear_points(points)
        if len(points) >= 3:
            shells.append(points)

    return shells
