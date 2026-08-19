"""Stage 3: turn (SourceLayer list + AtlasResult) into backend-agnostic SceneData.

Coordinate conventions
-----------------------
* PSD pixel space: X right, Y down, origin at canvas top-left.
* Maya world space: X right, Y up, origin at canvas center. Y is flipped
  and the canvas is re-centered so the recreated art sits symmetrically
  around the Maya origin instead of hanging off in +X/+Y like the PSD's
  top-left-origin space would.
* Depth: stack_index 0 (bottom-most PSD layer, see psd_reader) sits at
  Z=0. Each layer above it in the PSD steps further along +Z by
  `depth_step`, so the topmost PSD layer ends up nearest a camera looking
  down -Z from the +Z side -- i.e. PSD paint order becomes camera-facing
  parallax order with no extra flags needed. Pass a negative depth_step to
  reverse that if your rig's camera looks the other way.

Each layer's silhouette is traced by `contour_tracer` and quadrangulated by
`quadrangulate`, both of which work purely in layer-local pixel space; this
module is responsible for converting those pixel-space polygons into
world-placed, UV-mapped 3D geometry, picking unique Maya-safe node names,
and fixing up face winding (the Y-flip above can leave the traced
triangulation's winding backwards relative to the +Z-facing normal we
want).

A layer with more than one disconnected opaque region (e.g. two separate
rocks) traces to more than one shell; each shell becomes its own
LayerMesh, name-suffixed, so it still shows up as distinct geometry rather
than silently only keeping one piece.
"""

from __future__ import annotations

import logging
import re
from typing import List

from .contour_tracer import trace_layer_contours
from .quadrangulate import quadrangulate_polygon
from .scene_model import AtlasResult, LayerMesh, PackedLayer, SceneData, SourceLayer

logger = logging.getLogger(__name__)


def _maya_safe_name(raw: str, used: set) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", raw.strip()) or "layer"
    if not re.match(r"[A-Za-z_]", name[0]):
        name = f"_{name}"
    candidate = name
    n = 1
    while candidate in used:
        n += 1
        candidate = f"{name}_{n}"
    used.add(candidate)
    return candidate


def _pixel_to_local_and_uv(
    px: float,
    py: float,
    layer: SourceLayer,
    placement: PackedLayer,
    page_w: int,
    page_h: int,
    pixels_per_unit: float,
):
    local_x = (px - layer.width / 2.0) / pixels_per_unit
    local_y = -(py - layer.height / 2.0) / pixels_per_unit

    fx, fy = px / layer.width, py / layer.height
    atlas_px = placement.atlas_x + fx * placement.atlas_w
    atlas_py = placement.atlas_y + fy * placement.atlas_h
    u = atlas_px / page_w
    v = 1.0 - atlas_py / page_h

    return (local_x, local_y, 0.0), (u, v)


def _uv_affine(
    layer: SourceLayer,
    placement: PackedLayer,
    page_w: int,
    page_h: int,
    pixels_per_unit: float,
):
    """Collapse `_pixel_to_local_and_uv`'s mapping into (u0, su, v0, sv).

    Substituting the local->pixel inverse (px = local_x*ppu + W/2,
    py = -local_y*ppu + H/2) into the UV formula makes both coordinates
    affine in local space:

        u = (atlas_x + atlas_w/2)/page_w     + (atlas_w*ppu)/(W*page_w) * local_x
        v = 1 - (atlas_y + atlas_h/2)/page_h + (atlas_h*ppu)/(H*page_h) * local_y

    The layer's bbox center maps to the center of its atlas rect, and the
    scale terms convert one Maya unit into a fraction of the atlas page.
    """
    u0 = (placement.atlas_x + placement.atlas_w / 2.0) / page_w
    su = (placement.atlas_w * pixels_per_unit) / (layer.width * page_w)
    v0 = 1.0 - (placement.atlas_y + placement.atlas_h / 2.0) / page_h
    sv = (placement.atlas_h * pixels_per_unit) / (layer.height * page_h)
    return (u0, su, v0, sv)


def _signed_z(vertices, a, b, c) -> float:
    pa, pb, pc = vertices[a], vertices[b], vertices[c]
    return (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])


def _ensure_ccw_facing_positive_z(vertices, face):
    nz = _signed_z(vertices, face[0], face[1], face[2])
    return face if nz > 0 else tuple(reversed(face))


_DEGENERATE_AREA_EPS = 1e-9


def _is_degenerate(vertices, face) -> bool:
    """True if `face` has (numerically) zero area.

    Ear-clipping a boundary with locally flat/near-collinear runs (common
    in noisy micro-detail, e.g. anti-aliasing jitter along an otherwise
    straight edge) can still emit a flat sliver triangle as a last resort
    rather than losing polygon coverage -- see quadrangulate.py. Its 3
    derived quads are correct in structure but contribute zero visible
    area, so they're dropped here rather than exported as literal
    zero-area geometry.
    """
    area = abs(_signed_z(vertices, face[0], face[1], face[2])) + abs(_signed_z(vertices, face[0], face[2], face[3]))
    return area <= _DEGENERATE_AREA_EPS


def build_scene(
    layers: List[SourceLayer],
    atlas: AtlasResult,
    canvas_width: int,
    canvas_height: int,
    pixels_per_unit: float = 100.0,
    depth_step: float = 5.0,
    detail_level: float = 2.0,
    alpha_threshold: int = 10,
    min_contour_area: float = 50.0,
    source_psd: str = "",
) -> SceneData:
    placement_by_index = {p.source.stack_index: p for p in atlas.placements}
    page_dims = {p.index: (p.width, p.height) for p in atlas.pages}

    used_names: set = set()
    meshes: List[LayerMesh] = []

    for layer in sorted(layers, key=lambda l: l.stack_index):
        placement = placement_by_index[layer.stack_index]
        page_w, page_h = page_dims[placement.page]

        cx_px = (layer.left + layer.right) / 2.0
        cy_px = (layer.top + layer.bottom) / 2.0
        x = (cx_px - canvas_width / 2.0) / pixels_per_unit
        y = -(cy_px - canvas_height / 2.0) / pixels_per_unit
        z = layer.stack_index * depth_step

        shells = trace_layer_contours(
            layer,
            alpha_threshold=alpha_threshold,
            detail_level=detail_level,
            min_contour_area=min_contour_area,
        )
        if not shells:
            logger.warning("Layer %r produced no traceable silhouette; skipping", layer.name)
            continue

        for shell_index, boundary_px in enumerate(shells):
            verts_px, faces = quadrangulate_polygon(boundary_px)

            vertices, uvs = [], []
            for px, py in verts_px:
                local, uv = _pixel_to_local_and_uv(px, py, layer, placement, page_w, page_h, pixels_per_unit)
                vertices.append(local)
                uvs.append(uv)

            faces = [_ensure_ccw_facing_positive_z(vertices, f) for f in faces]
            kept_faces = [f for f in faces if not _is_degenerate(vertices, f)]
            if len(kept_faces) != len(faces):
                logger.info(
                    "Layer %r shell %d: dropped %d zero-area face(s) from a locally flat traced boundary",
                    layer.name,
                    shell_index,
                    len(faces) - len(kept_faces),
                )
            faces = kept_faces

            name = layer.name if len(shells) == 1 else f"{layer.name}_{shell_index + 1}"
            meshes.append(
                LayerMesh(
                    maya_name=_maya_safe_name(name, used_names),
                    source_name=layer.name,
                    stack_index=layer.stack_index,
                    translate=(x, y, z),
                    atlas_page=placement.page,
                    vertices=vertices,
                    uvs=uvs,
                    faces=faces,
                    uv_affine=_uv_affine(layer, placement, page_w, page_h, pixels_per_unit),
                )
            )

    return SceneData(
        meshes=meshes,
        atlas_pages=atlas.pages,
        pixels_per_unit=pixels_per_unit,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        depth_step=depth_step,
        source_psd=source_psd,
    )
