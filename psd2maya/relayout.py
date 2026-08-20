"""Rebake atlas textures to match a manual Maya "Layout UV" pass.

Workflow this supports: build the rig, then in Maya's UV Editor select some
of its meshes and run the native Layout UV tool (`polyLayoutUV` /
`polyMultiLayoutUV`, whichever the Maya version's "Layout" button calls) to
repack their UV shells more efficiently than this package's rectangular
per-layer atlas packing does. That only moves UV *coordinates* -- the atlas
PNG already on disk still has each layer's pixels at their *original*
packed position, so after a manual layout the mesh and its texture disagree
about where each layer's art actually is.
`rebuild_textures_from_uv_layout` resolves that by rebaking a fresh atlas
PNG per UV set (`atlasPage0`, `atlasPage1`, ... -- see
`maya_backend._sort_into_uv_set`) that puts each layer's pixels wherever its
mesh's UVs now say they belong, and repoints that UV set's file texture
node at the new image.

This is a *manual* step (call it, or click "Rebuild Texture from UV Layout"
in the UI, after running Layout UV yourself), not a callback that fires
automatically whenever Layout UV runs -- Maya's Layout UV command has no
built-in "after" hook to attach to across versions, and an explicit step is
more predictable than a background command-callback watching for
polyLayoutUV/polyMultiLayoutUV by name.

Rebuild scope: per UV set, independently. If a Layout UV pass mixed shells
from more than one UV set into the same 0..1 tile, each set's texture is
still rebaked separately from just that set's own meshes -- shells from a
different set that happen to visually overlap in UV space are not merged
into one shared image.

Pixel source: the original .psd file (`scene.source_psd`), recomposited
fresh per layer, not the already-packed atlas PNG -- avoids a second lossy
resample of pixels that were already resampled once into the old atlas.
This assumes the PSD's layers haven't moved or been resized since the rig
was built: the local-position <-> source-pixel mapping baked into each
mesh's geometry at build time is not re-derived here, so a changed layer
bbox in the PSD would desync it.

Geometry: Maya's Layout UV repositions/rescales/rotates each UV shell as a
whole without distorting it internally (it does not run Unfold), so for a
given mesh the mapping from its current local vertex positions to its
current UV coordinates is *exactly* one 2D affine transform, recoverable by
fitting `(u, v) = (a*x + b*y + c, d*x + e*y + f)` against every vertex via
least squares (numpy). Composed with the (unchanged) local-position-to-
source-pixel mapping, that gives an exact affine from "where a pixel is in
the layer's own recomposited image" to "where it belongs in the new atlas",
which PIL's affine image transform then uses to warp each layer's image
directly into its new spot. Relies on this package's own invariant that UV
index == vertex index (no UV seams -- every LayerMesh is a seamless planar
grid; see maya_backend._create_mesh_transform/reproject_uvs), which Layout
UV does not change.

After a successful rebake, each mesh's locked `psdUv*` attributes (see
maya_backend._tag_uv_affine) are overwritten with the newly fitted affine,
so a later `reproject_uvs`/`retopologize` call reprojects UVs consistent
with the *new* layout instead of silently reverting to the pre-layout one.

Multiple UV sets: before that overwrite, whatever the mapping looked like
*before* this rebake is written into a second, preserved UV set on the same
mesh (`_write_uv_set`, named `<uvSet>_original`, e.g.
`atlasPage0_original`) -- so nothing about the pre-layout state is lost,
even though the live/textured UV set (`atlasPage0`) keeps using its own
data (the post-Layout-UV values you already produced by hand) unchanged.
This happens automatically, every rebake, for every mesh -- there's no
toggle to skip it. It's a rolling one-generation snapshot rather than an
unbounded history: rebaking the same mesh again overwrites `_original`
with whatever the mapping was one rebake ago, it doesn't pile up
`_original_2`, `_original_3`, etc. New UV sets are appended after existing
ones without disturbing an existing set's index or which set is
"current" (verified against Maya directly), so this never interferes with
the `uvSet[0]` this module and `maya_backend._sort_into_uv_set` rely on
for the live, texture-linked set.

Known limitation: unlike `atlas_packer.py`'s original packing, rebaked
shells get no bleed-padding border, so filtering/mipmapping right at a
shell's edge can pick up a neighboring shell's pixels (or transparency)
slightly more readily than the original atlas did.

Superimposing the atlas texture PNGs themselves (`merge_textures`)
----------------------------------------------------------------------
A literal thing, not a scene-geometry thing: every atlas page still in
use has its own PNG file on disk (the original packed atlas, or a
Layout-UV-rebaked one). `merge_textures` superimposes those PNG files
directly on top of each other -- the same operation as selecting every
layer in Photoshop and choosing Merge Layers/Flatten Image, just applied
to this package's own generated PNGs instead of PSD layers -- into one
merged PNG, and repoints the existing meshes at it. It creates no new
mesh and no backdrop card; every mesh that already existed keeps its own
shape and position, just with its UVs and shading updated to point at the
one merged texture instead of its own page's.

Each page's PNG is composited at its native resolution, pasted at the
same (0, 0) origin as every other page (not resized to match one
another, and not repositioned/packed side by side -- this is a stack, not
a pack), in ascending page-index order, onto a canvas sized to the
largest page's dimensions. A page smaller than that canvas simply doesn't
cover the rest of it (left transparent), rather than being stretched to
fill it.

Because pages aren't resized, a mesh's UV mapping only needs a *scale* (no
translation, since every page shares the same (0, 0) origin in the merged
canvas) to stay correct: `new_u = (page.width/merged_w) * u`, `new_v` the
analogous per-page-height scale composed with the existing v-flip
convention -- composed directly with the mesh's existing `psdUv*` affine,
with no per-vertex fitting needed. Same rolling-snapshot behavior as the
Layout UV rebake: each mesh's pre-merge mapping is preserved as another UV
set (`<uvSet>_premerge`) before being overwritten. Every merged mesh's UV
set is renamed to one shared name (`mergedAtlas`) and reassigned to one
new shading group; the old per-page `atlasFileN`/`atlasPlace2dN`/
`atlasShaderN`/`atlasSGN` nodes are deleted once nothing references them,
but their PNG files are left on disk untouched.

Known limitation: since different atlas pages are independently packed
sprite sheets with no relationship to each other's pixel layout, directly
superimposing them can coincidentally overlap unrelated content (page 0's
sky at some pixel landing on top of page 1's rock at that same pixel) --
this is an accepted consequence of merging pages literally as whole
images rather than recomputing a new non-overlapping packing.

No-ops (returns `None`, changes nothing) if fewer than two atlas pages
currently have both an existing mesh and a resolvable texture on disk.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .scene_model import SceneData

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def _fit_affine(sources: List[Point], targets: List[Point]) -> Tuple[float, float, float, float, float, float]:
    """Least-squares fit of (u, v) = (a*x + b*y + c, d*x + e*y + f) mapping `sources` to `targets`."""
    import numpy as np  # noqa: PLC0415

    a_matrix = np.array([[x, y, 1.0] for x, y in sources])
    u = np.array([t[0] for t in targets])
    v = np.array([t[1] for t in targets])
    coeff_u, *_ = np.linalg.lstsq(a_matrix, u, rcond=None)
    coeff_v, *_ = np.linalg.lstsq(a_matrix, v, rcond=None)
    return (*coeff_u.tolist(), *coeff_v.tolist())


def _read_current_local_xy_and_uvs(shape: str) -> Tuple[List[Point], List[Point]]:
    """Read `shape`'s current object-space (x, y) and current-UV-set (u, v), aligned by index."""
    import maya.api.OpenMaya as om2  # noqa: PLC0415
    import maya.cmds as cmds  # noqa: PLC0415

    sel = om2.MSelectionList()
    sel.add(shape)
    dag = sel.getDagPath(0)
    fn_mesh = om2.MFnMesh(dag)

    points = fn_mesh.getPoints(om2.MSpace.kObject)
    uv_set = cmds.polyUVSet(shape, query=True, currentUVSet=True)[0]
    u_array, v_array = fn_mesh.getUVs(uvSet=uv_set)

    xy = [(points[i].x, points[i].y) for i in range(len(points))]
    uv = list(zip(u_array, v_array))
    return xy, uv


def _write_uv_set(shape: str, local_xy: List[Point], affine, uv_set_name: str) -> None:
    """Write `affine` evaluated at `local_xy` into `uv_set_name` on `shape`, creating it if needed.

    Two uses in this module: writing a fresh live mapping into whatever UV
    set is currently "current" (`merge_textures`), and preserving a
    snapshot of a mapping that's about to be overwritten into a separate,
    newly created set (both `rebuild_textures_from_uv_layout`'s
    `<uvSet>_original` and `merge_textures`'s `<uvSet>_premerge`) -- so a
    mesh naturally accumulates multiple UV sets as it goes through more of
    this package's steps. New UV sets are appended after existing ones
    (verified: creating a second UV set does not disturb the first one's
    index or which set is "current"), so writing a *new* named set here
    never disturbs the `uvSet[0]` the rest of this module and
    `maya_backend._sort_into_uv_set` rely on for the live/textured set.

    Each snapshot name is a rolling one-generation history, not an
    unbounded log: if `uv_set_name` already exists, its data is simply
    overwritten rather than accumulating `_2`, `_3`, ... suffixes.
    """
    import maya.api.OpenMaya as om2  # noqa: PLC0415
    import maya.cmds as cmds  # noqa: PLC0415

    if uv_set_name not in (cmds.polyUVSet(shape, query=True, allUVSets=True) or []):
        cmds.polyUVSet(shape, create=True, uvSet=uv_set_name)

    a, b, c, d, e, f = affine
    u_array = om2.MFloatArray([a * x + b * y + c for x, y in local_xy])
    v_array = om2.MFloatArray([d * x + e * y + f for x, y in local_xy])

    sel = om2.MSelectionList()
    sel.add(shape)
    dag = sel.getDagPath(0)
    fn_mesh = om2.MFnMesh(dag)

    face_counts, face_verts = [], []
    for face_index in range(fn_mesh.numPolygons):
        verts = fn_mesh.getPolygonVertices(face_index)
        face_counts.append(len(verts))
        face_verts.extend(verts)

    fn_mesh.clearUVs(uv_set_name)
    fn_mesh.setUVs(u_array, v_array, uv_set_name)
    fn_mesh.assignUVs(om2.MIntArray(face_counts), om2.MIntArray(face_verts), uv_set_name)


def _pad_edge_replicate(image, pad: int):
    """Return `image` grown by `pad` px on every side, replicating its edge pixels.

    The affine fit in `_rebake_shell` is exact only up to floating-point
    residual (~1e-5 px in practice), but PIL's affine transform does not
    clamp an out-of-range source sample to the nearest valid pixel -- it
    fills with transparent black. Right at a shell's true boundary (e.g. a
    vertex whose exact source position is x=0), that residual can push the
    computed source coordinate a hair negative, sampling into the void
    instead of the actual edge pixel and puncturing an otherwise-opaque
    edge. Padding with replicated edge pixels first means a few-pixel
    numerical miss still lands on the correct content instead of a hole.
    """
    from PIL import Image  # noqa: PLC0415

    w, h = image.size
    padded = Image.new("RGBA", (w + 2 * pad, h + 2 * pad))
    padded.paste(image, (pad, pad))
    if pad <= 0:
        return padded

    left, right = image.crop((0, 0, 1, h)), image.crop((w - 1, 0, w, h))
    top, bottom = image.crop((0, 0, w, 1)), image.crop((0, h - 1, w, h))
    padded.paste(left.resize((pad, h)), (0, pad))
    padded.paste(right.resize((pad, h)), (w + pad, pad))
    padded.paste(top.resize((w, pad)), (pad, 0))
    padded.paste(bottom.resize((w, pad)), (pad, h + pad))
    padded.paste(image.crop((0, 0, 1, 1)).resize((pad, pad)), (0, 0))
    padded.paste(image.crop((w - 1, 0, w, 1)).resize((pad, pad)), (w + pad, 0))
    padded.paste(image.crop((0, h - 1, 1, h)).resize((pad, pad)), (0, h + pad))
    padded.paste(image.crop((w - 1, h - 1, w, h)).resize((pad, pad)), (w + pad, h + pad))
    return padded


def _rebake_shell(
    layer_image,
    pixels_per_unit: float,
    local_xy: List[Point],
    new_uv: List[Point],
    page_w: int,
    page_h: int,
    canvas,
) -> None:
    """Warp `layer_image` into `canvas` wherever `new_uv` now places it."""
    from PIL import Image  # noqa: PLC0415

    src_pad = 3
    padded_source = _pad_edge_replicate(layer_image.convert("RGBA"), src_pad)

    # Source coordinates shift by +src_pad in both axes now that the source
    # image itself has a src_pad-wide border glued on.
    old_px = [
        (
            x * pixels_per_unit + layer_image.width / 2.0 + src_pad,
            layer_image.height / 2.0 - y * pixels_per_unit + src_pad,
        )
        for x, y in local_xy
    ]
    new_atlas_px = [(u * page_w, (1.0 - v) * page_h) for u, v in new_uv]

    # Fit the inverse (new atlas pixel -> old layer pixel) directly, rather
    # than fitting forward and inverting the matrix -- for an exact affine
    # relationship (see module docstring) both give the same answer, and
    # fitting directly needs no matrix inversion.
    a, b, c, d, e, f = _fit_affine(new_atlas_px, old_px)

    dest_xs = [p[0] for p in new_atlas_px]
    dest_ys = [p[1] for p in new_atlas_px]
    dst_pad = 2
    dst_x0 = max(math.floor(min(dest_xs)) - dst_pad, 0)
    dst_y0 = max(math.floor(min(dest_ys)) - dst_pad, 0)
    dst_x1 = min(math.ceil(max(dest_xs)) + dst_pad, page_w)
    dst_y1 = min(math.ceil(max(dest_ys)) + dst_pad, page_h)
    dst_w, dst_h = dst_x1 - dst_x0, dst_y1 - dst_y0
    if dst_w <= 0 or dst_h <= 0:
        return

    # PIL's affine transform maps a destination-local pixel (i, j) back to a
    # source pixel via `coeffs`; offset by (dst_x0, dst_y0) since (i, j) is
    # local to the small destination crop, not the full atlas page. BILINEAR
    # rather than BICUBIC: cubic resampling can overshoot past the local
    # min/max near a hard alpha edge (a traced silhouette boundary is
    # exactly that), visibly tinting/darkening pixels just inside the edge.
    coeffs = (a, b, a * dst_x0 + b * dst_y0 + c, d, e, d * dst_x0 + e * dst_y0 + f)
    warped = padded_source.transform((dst_w, dst_h), Image.AFFINE, coeffs, resample=Image.BILINEAR)
    canvas.alpha_composite(warped, (dst_x0, dst_y0))


def rebuild_textures_from_uv_layout(
    scene: SceneData,
    psd_path: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> Dict[int, str]:
    """Rebake each UV set's atlas texture to match its meshes' current UV positions.

    Call this after manually running Maya's Layout UV on some of the rig's
    meshes. Returns {atlas_page_index: new_texture_path} for every page that
    had at least one still-existing, still-matched mesh to rebake.
    """
    import maya.cmds as cmds  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from .maya_backend import _tag_uv_affine, read_uv_affine  # noqa: PLC0415 -- local import to avoid a cycle
    from .psd_reader import extract_layers  # noqa: PLC0415

    psd_path = psd_path or scene.source_psd
    if not psd_path or not os.path.isfile(psd_path):
        raise ValueError(f"Can't rebake: source PSD not found at {psd_path!r}")
    out_dir = out_dir or os.path.dirname(psd_path)

    # include_hidden=True: a superset of whatever filter the original build
    # used, so every layer that made it into `scene.meshes` the first time
    # is guaranteed to be found again here regardless of that setting.
    layers, _, _ = extract_layers(psd_path, include_hidden=True)
    layers_by_name = {layer.name: layer for layer in layers}

    meshes_by_page: Dict[int, list] = defaultdict(list)
    for mesh in scene.meshes:
        meshes_by_page[mesh.atlas_page].append(mesh)

    new_paths: Dict[int, str] = {}
    for page in scene.atlas_pages:
        group = [m for m in meshes_by_page.get(page.index, []) if cmds.objExists(m.maya_name)]
        if not group:
            logger.warning("No existing mesh(es) for atlas page %d; skipping", page.index)
            continue

        canvas = Image.new("RGBA", (page.width, page.height), (0, 0, 0, 0))
        rebaked = 0
        for mesh in group:
            layer = layers_by_name.get(mesh.source_name)
            if layer is None:
                logger.warning(
                    "Mesh %r's source layer %r not found in %r; leaving it out of the rebake",
                    mesh.maya_name,
                    mesh.source_name,
                    psd_path,
                )
                continue

            shape = cmds.listRelatives(mesh.maya_name, shapes=True, fullPath=True)[0]
            local_xy, new_uv = _read_current_local_xy_and_uvs(shape)
            _rebake_shell(layer.pixels, scene.pixels_per_unit, local_xy, new_uv, page.width, page.height, canvas)

            old_affine = read_uv_affine(mesh.maya_name)
            if old_affine is not None:
                current_uv_set = cmds.polyUVSet(shape, query=True, currentUVSet=True)[0]
                _write_uv_set(shape, local_xy, old_affine, f"{current_uv_set}_original")

            new_affine = _fit_affine(local_xy, new_uv)
            _tag_uv_affine(mesh.maya_name, new_affine)
            rebaked += 1

        if rebaked == 0:
            continue

        new_path = os.path.join(out_dir, f"atlas_{page.index}_relayout.png")
        canvas.save(new_path)

        first_shape = cmds.listRelatives(group[0].maya_name, shapes=True, fullPath=True)[0]
        linked = cmds.uvLink(query=True, uvSet=f"{first_shape}.uvSet[0].uvSetName")
        if not linked:
            raise RuntimeError(
                f"{first_shape!r} has no uvLink'd texture; it wasn't sorted into a UV set by "
                "maya_backend.build_in_maya, so there's no file node to repoint."
            )
        cmds.setAttr(f"{linked[0]}.fileTextureName", new_path, type="string")
        new_paths[page.index] = new_path
        logger.info("Rebaked atlas page %d -> %s (%d mesh(es))", page.index, new_path, rebaked)

    return new_paths


def merge_textures(scene: SceneData, out_dir: Optional[str] = None) -> Optional[str]:
    """Superimpose every atlas page still in use into one merged PNG, and repoint
    the existing meshes at it. Creates no new geometry.

    See the module docstring's "Superimposing the atlas texture PNGs
    themselves" section for the full design rationale. Returns the merged
    texture's path, or None if fewer than two atlas pages currently have
    both an existing mesh and a resolvable texture on disk (nothing to
    merge).
    """
    import maya.cmds as cmds  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from .maya_backend import _tag_uv_affine, read_uv_affine  # noqa: PLC0415 -- local import to avoid a cycle

    meshes_by_page: Dict[int, list] = defaultdict(list)
    for mesh in scene.meshes:
        if cmds.objExists(mesh.maya_name):
            meshes_by_page[mesh.atlas_page].append(mesh)

    page_file_nodes: Dict[int, str] = {}
    page_images: Dict[int, object] = {}
    page_paths: Dict[int, str] = {}
    for page in scene.atlas_pages:
        if page.index not in meshes_by_page:
            continue
        file_node = f"atlasFile{page.index}"
        if not cmds.objExists(file_node):
            logger.warning("File node %r missing for atlas page %d; leaving it out of the merge", file_node, page.index)
            continue
        tex_path = cmds.getAttr(f"{file_node}.fileTextureName")
        if not tex_path or not os.path.isfile(tex_path):
            logger.warning(
                "Texture %r for atlas page %d not found on disk; leaving it out of the merge", tex_path, page.index
            )
            continue
        page_file_nodes[page.index] = file_node
        page_paths[page.index] = tex_path
        page_images[page.index] = Image.open(tex_path).convert("RGBA")

    if len(page_images) < 2:
        logger.info("Fewer than two atlas pages have both an existing mesh and a texture; nothing to merge.")
        return None

    out_dir = out_dir or os.path.dirname(next(iter(page_paths.values())))

    # Stack, don't pack: every page is pasted at its own native resolution at
    # the same (0, 0) origin -- like Photoshop's Merge Layers/Flatten Image --
    # onto a canvas sized to the largest page, in ascending page-index order.
    merged_w = max(img.width for img in page_images.values())
    merged_h = max(img.height for img in page_images.values())
    canvas = Image.new("RGBA", (merged_w, merged_h), (0, 0, 0, 0))
    for page_index in sorted(page_images):
        canvas.alpha_composite(page_images[page_index], (0, 0))

    merged_path = os.path.join(out_dir, "merged_atlas.png")
    canvas.save(merged_path)

    # One shared shading network for everything, matching maya_backend.build_in_maya's
    # per-page naming pattern but singular, since there's only one texture now.
    file_node = cmds.shadingNode("file", asTexture=True, isColorManaged=True, name="mergedAtlasFile")
    cmds.setAttr(f"{file_node}.fileTextureName", merged_path, type="string")
    cmds.setAttr(f"{file_node}.alphaIsLuminance", False)
    place2d = cmds.shadingNode("place2dTexture", asUtility=True, name="mergedAtlasPlace2d")
    cmds.connectAttr(f"{place2d}.outUV", f"{file_node}.uvCoord")
    cmds.connectAttr(f"{place2d}.outUvFilterSize", f"{file_node}.uvFilterSize")
    shader = cmds.shadingNode("lambert", asShader=True, name="mergedAtlasShader")
    cmds.connectAttr(f"{file_node}.outColor", f"{shader}.color")
    cmds.connectAttr(f"{file_node}.outTransparency", f"{shader}.transparency")
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="mergedAtlasSG")
    cmds.connectAttr(f"{shader}.outColor", f"{sg}.surfaceShader", force=True)

    merged_uv_set_name = "mergedAtlas"
    merged_count = 0
    for page_index, meshes in meshes_by_page.items():
        if page_index not in page_images:
            continue
        page = next(p for p in scene.atlas_pages if p.index == page_index)

        # Closed-form remap, no fitting needed: every page shares the same
        # (0, 0) origin in the merged canvas (a stack, not a pack), so only a
        # scale is needed, never a translation -- new_u = q*u, new_v = (1-s) + s*v
        # (the v-flip convention means a page's own top edge, v=1, must still
        # land at the merged canvas's v=1 only when the page is as tall as the
        # canvas; shorter pages' v=1 lands at (1-s) + s*1 = 1, always correct,
        # while v=0 lands at 1-s, i.e. flush with the canvas's own bottom
        # margin above y=merged_h).
        q = page.width / merged_w
        s = page.height / merged_h

        for mesh in meshes:
            old_affine = read_uv_affine(mesh.maya_name)
            if old_affine is None:
                logger.warning("Mesh %r has no psdUv* attributes; leaving it out of the merge", mesh.maya_name)
                continue
            a, b, c, d, e, f = old_affine
            new_affine = (q * a, q * b, q * c, s * d, s * e, (1.0 - s) + s * f)

            shape = cmds.listRelatives(mesh.maya_name, shapes=True, fullPath=True)[0]
            local_xy, _ = _read_current_local_xy_and_uvs(shape)
            current_uv_set = cmds.polyUVSet(shape, query=True, currentUVSet=True)[0]

            _write_uv_set(shape, local_xy, old_affine, f"{current_uv_set}_premerge")
            _write_uv_set(shape, local_xy, new_affine, current_uv_set)
            _tag_uv_affine(mesh.maya_name, new_affine)

            if current_uv_set != merged_uv_set_name:
                cmds.polyUVSet(shape, rename=True, uvSet=current_uv_set, newUVSet=merged_uv_set_name)

            cmds.sets(shape, edit=True, forceElement=sg)
            cmds.uvLink(make=True, uvSet=f"{shape}.uvSet[0].uvSetName", texture=file_node)
            merged_count += 1

    for page_index, old_file_node in page_file_nodes.items():
        for node in (old_file_node, f"atlasPlace2d{page_index}", f"atlasShader{page_index}", f"atlasSG{page_index}"):
            if cmds.objExists(node):
                cmds.delete(node)

    logger.info("Merged %d atlas page(s) (%d mesh(es)) -> %s", len(page_images), merged_count, merged_path)
    return merged_path
