"""Turn a traced silhouette polygon into an all-quad mesh, unconditionally.

Maya's own `polyTriangulate` + `polyQuad` combo (the commented-out
suggestion in most contour-to-mesh snippets) is a best-effort heuristic:
`polyQuad` merges adjacent triangle pairs where it can, but for an
arbitrary simple polygon there is no guarantee every triangle finds a
partner, so it can leave stray triangles or n-gons behind. Since the
requirement here is *only* quads, this module instead uses a
triangle-to-quad split that is unconditional for any triangle:

    ear-clip triangulate the polygon
    for each triangle (A, B, C):
        G  = centroid(A, B, C)
        Mab, Mbc, Mca = edge midpoints
        emit quads (A, Mab, G, Mca), (B, Mbc, G, Mab), (C, Mca, G, Mbc)

Every triangle always splits into exactly 3 quads this way -- no merge
heuristic, no leftover faces -- so the result is 100% quads regardless of
the input polygon's shape, and it works identically whether the caller is
building a live Maya scene or writing a plain-text .ma file with no Maya
involved at all. The boundary vertices are untouched, so the mesh's outer
silhouette still exactly matches the traced polygon; only the interior
gets extra vertices.

The tradeoff: face count triples relative to a plain triangulation, and
every triangle contributes an extra centroid + up to 3 new edge-midpoint
vertices. For a very high-vertex-count traced boundary (low
`--detail-level`), this can produce a lot of geometry -- raise
`--detail-level` to simplify the boundary first if that matters.
"""

from __future__ import annotations

from typing import List, Tuple

Point = Tuple[float, float]


def _signed_area(points: List[Point]) -> float:
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _cross(a: Point, b: Point, c: Point) -> float:
    """Z-component of (b-a) x (c-b); >0 means a left turn (CCW) at b."""
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def _point_in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def ear_clip_triangulate(points: List[Point]) -> List[Tuple[int, int, int]]:
    """Triangulate a simple polygon (no self-intersections, no holes).

    Returns triangles as index triples into `points`. Orientation of
    `points` doesn't matter -- normalized to CCW internally before clipping.
    """
    n = len(points)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    indices = list(range(n))
    if _signed_area(points) < 0:
        indices.reverse()

    triangles: List[Tuple[int, int, int]] = []
    guard, max_guard = 0, n * n + 16

    while len(indices) > 3 and guard < max_guard:
        guard += 1
        m = len(indices)
        ear_found = False
        # Two passes: first require a strictly convex tip (the normal,
        # correct case), then relax to "not reflex" (cross >= 0) so a
        # locally flat/collinear run of points -- which can survive
        # simplification along a noisy near-straight edge -- still yields
        # forward progress instead of falling through to the fan-triangulate
        # bailout below.
        for strict in (True, False):
            for k in range(m):
                i_prev, i_curr, i_next = indices[(k - 1) % m], indices[k], indices[(k + 1) % m]
                a, b, c = points[i_prev], points[i_curr], points[i_next]
                cross = _cross(a, b, c)
                if strict:
                    if cross <= 0:
                        continue  # reflex or flat, not a valid ear tip yet
                elif cross < -1e-9:
                    continue  # still clearly reflex even under the relaxed pass
                if any(
                    other not in (i_prev, i_curr, i_next) and _point_in_triangle(points[other], a, b, c)
                    for other in indices
                ):
                    continue
                triangles.append((i_prev, i_curr, i_next))
                del indices[k]
                ear_found = True
                break
            if ear_found:
                break
        if not ear_found:
            # Numerically pathological remainder (shouldn't happen for a
            # simple polygon traced from real image contours). Fan-
            # triangulate what's left from a single vertex rather than
            # dropping vertices, which would silently lose boundary
            # coverage instead of just producing a few degenerate triangles.
            for k in range(1, len(indices) - 1):
                triangles.append((indices[0], indices[k], indices[k + 1]))
            indices = []
            break

    if len(indices) == 3:
        triangles.append(tuple(indices))
    return triangles


def quadrangulate_polygon(points: List[Point]) -> Tuple[List[Point], List[Tuple[int, int, int, int]]]:
    """Ear-clip `points` then split every triangle into 3 quads. See module docstring."""
    triangles = ear_clip_triangulate(points)

    vertices: List[Point] = list(points)
    midpoint_cache = {}

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        idx = midpoint_cache.get(key)
        if idx is None:
            pi, pj = vertices[i], vertices[j]
            idx = len(vertices)
            vertices.append(((pi[0] + pj[0]) / 2.0, (pi[1] + pj[1]) / 2.0))
            midpoint_cache[key] = idx
        return idx

    faces: List[Tuple[int, int, int, int]] = []
    for i, j, k in triangles:
        a, b, c = vertices[i], vertices[j], vertices[k]
        centroid_idx = len(vertices)
        vertices.append(((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0))

        m_ij, m_jk, m_ki = midpoint(i, j), midpoint(j, k), midpoint(k, i)
        faces.append((i, m_ij, centroid_idx, m_ki))
        faces.append((j, m_jk, centroid_idx, m_ij))
        faces.append((k, m_ki, centroid_idx, m_jk))

    return vertices, faces
