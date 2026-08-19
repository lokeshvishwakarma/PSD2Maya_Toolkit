"""Backend B: build the same SceneData live inside Maya via the Maya Python API.

This module only imports `maya.cmds`/`maya.api.OpenMaya` lazily, inside
functions, so the rest of the package (and this file itself) can still be
imported for introspection/testing outside Maya. Actually calling
`build_in_maya` requires either running inside Maya (Script Editor / a
plugin) or under `mayapy`.

Each LayerMesh is now a quad grid (potentially hundreds of faces with
shared edges/vertices), not a single independent quad, so it's built with
one `MFnMesh.create()` call per layer via the Maya Python API (`maya.api.
OpenMaya`) rather than by chaining many `cmds.polyCreateFacet` calls and
merging them -- `create()` takes the full vertex array plus a flat
polygon-connectivity array directly, which is both the correct way to
build arbitrary shared-vertex topology in one shot and far cheaper than
building N independent quads and welding their borders afterward.

Optional Maya-native retopology
--------------------------------
`build_in_maya(..., retopo=True)` runs `cmds.polyRetopo` on each mesh after
it's built, trading the guaranteed-quad ear-clip topology for Maya's more
uniform quad flow. That normally destroys UVs -- a remesher replaces every
vertex, so there's nothing left for the original UVs to attach to, and the
usual remedy is `cmds.transferAttributes` from a saved copy of the source
mesh (a world-space raycast projection, which introduces error and jagged
UV borders wherever the new topology crosses a seam).

This module doesn't need that. Each LayerMesh carries `uv_affine`, the
closed-form `(u, v) = (u0 + su*x, v0 + sv*y)` mapping from local position to
atlas UV (exact because the mesh is planar and the pixel->atlas mapping is
affine -- see mesh_builder._uv_affine). `reproject_uvs` evaluates it for
whatever vertices exist *now*, so UVs come back exact at any topology with
no projection error, no seam cleanup, and no source mesh to keep around.
The mapping is stored on the transform as locked `psdUv*` attributes, so
reprojection also works later, after any further remeshing the user does by
hand.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .scene_model import LayerMesh, SceneData

logger = logging.getLogger(__name__)

_UV_ATTRS = ("psdUvU0", "psdUvScaleU", "psdUvV0", "psdUvScaleV")


def _create_mesh_transform(mesh: LayerMesh):
    """Build one LayerMesh's geometry+UVs via the API. Returns the new transform's MObject.

    `MFnMesh.create()` called with no explicit parent creates a new
    transform + shape pair and returns the *transform*'s MObject (its
    apiType is kTransform, not kMesh) -- despite the name suggesting
    otherwise, there is no separate shape MObject to hand back here.
    """
    import maya.api.OpenMaya as om2  # noqa: PLC0415 -- only valid inside Maya/mayapy

    points = om2.MPointArray([om2.MPoint(x, y, z) for x, y, z in mesh.vertices])
    poly_counts = om2.MIntArray([4] * len(mesh.faces))
    poly_connects = om2.MIntArray([idx for face in mesh.faces for idx in face])

    fn_mesh = om2.MFnMesh()
    transform_obj = fn_mesh.create(points, poly_counts, poly_connects)

    u_array = om2.MFloatArray([u for u, _ in mesh.uvs])
    v_array = om2.MFloatArray([v for _, v in mesh.uvs])
    fn_mesh.setUVs(u_array, v_array)
    uv_counts = om2.MIntArray([4] * len(mesh.faces))
    uv_ids = poly_connects  # uv index == vertex index; there are no UV seams in a planar grid
    fn_mesh.assignUVs(uv_counts, uv_ids)

    return transform_obj


def _tag_uv_affine(transform: str, uv_affine) -> None:
    """Store the local->UV affine mapping on `transform` as locked scalar attrs."""
    import maya.cmds as cmds  # noqa: PLC0415

    for attr, value in zip(_UV_ATTRS, uv_affine):
        if not cmds.attributeQuery(attr, node=transform, exists=True):
            cmds.addAttr(transform, longName=attr, attributeType="double", keyable=False)
        cmds.setAttr(f"{transform}.{attr}", value, lock=True)


def read_uv_affine(transform: str) -> Optional[tuple]:
    """Read back the `psdUv*` mapping from a transform, or None if it isn't tagged."""
    import maya.cmds as cmds  # noqa: PLC0415

    if not all(cmds.attributeQuery(a, node=transform, exists=True) for a in _UV_ATTRS):
        return None
    return tuple(cmds.getAttr(f"{transform}.{a}") for a in _UV_ATTRS)


def reproject_uvs(transform: str, uv_affine=None) -> int:
    """Recompute `transform`'s UVs from its current vertex positions. Returns vertex count.

    Evaluates the stored affine mapping per vertex, so it's exact for any
    topology -- unlike `transferAttributes`, there's no source mesh, no
    raycast, and no seam repair. One UV per vertex (a planar mesh needs no
    UV seams), so the UV indices are just the vertex indices.
    """
    import maya.api.OpenMaya as om2  # noqa: PLC0415

    if uv_affine is None:
        uv_affine = read_uv_affine(transform)
        if uv_affine is None:
            raise ValueError(
                f"{transform!r} has no psdUv* attributes; it wasn't built by psd2maya "
                "(or was rebuilt without them), so its UVs can't be reprojected."
            )
    u0, su, v0, sv = uv_affine

    sel = om2.MSelectionList()
    sel.add(transform)
    dag = sel.getDagPath(0)
    dag.extendToShape()
    fn_mesh = om2.MFnMesh(dag)

    # Local (object) space: uv_affine is defined against the mesh's own local
    # coordinates, which is what makes it survive the transform being moved.
    points = fn_mesh.getPoints(om2.MSpace.kObject)
    u_array = om2.MFloatArray([u0 + su * points[i].x for i in range(len(points))])
    v_array = om2.MFloatArray([v0 + sv * points[i].y for i in range(len(points))])

    face_counts, face_verts = [], []
    for face_index in range(fn_mesh.numPolygons):
        verts = fn_mesh.getPolygonVertices(face_index)
        face_counts.append(len(verts))
        face_verts.extend(verts)

    fn_mesh.clearUVs()
    fn_mesh.setUVs(u_array, v_array)
    fn_mesh.assignUVs(om2.MIntArray(face_counts), om2.MIntArray(face_verts))
    return len(points)


def retopologize(transform: str, target_face_count: int = 200, uv_affine=None) -> None:
    """Remesh `transform` with Maya's polyRetopo, then restore exact UVs.

    History is deleted before reprojecting: polyRetopo leaves a live
    construction-history node that would otherwise re-evaluate and discard
    the UVs written afterward.
    """
    import maya.cmds as cmds  # noqa: PLC0415

    if uv_affine is None:
        uv_affine = read_uv_affine(transform)

    cmds.polyRetopo(transform, targetFaceCount=target_face_count)
    cmds.delete(transform, constructionHistory=True)
    reproject_uvs(transform, uv_affine=uv_affine)


def build_in_maya(
    scene: SceneData,
    atlas_texture_paths: Dict[int, str],
    root_name: str = "psd2maya_root",
    retopo: bool = False,
    target_face_count: int = 200,
    retopo_failures: Optional[list] = None,
) -> str:
    """Build `scene` in the currently open Maya session. Returns the root transform's name.

    Pass a list as `retopo_failures` to find out which meshes fell back to
    their original traced topology. Worth checking: the fallback is also
    all-quads, so a face/quad audit alone can't distinguish "retopologized"
    from "polyRetopo refused and we kept the original".
    """
    import maya.api.OpenMaya as om2  # noqa: PLC0415
    import maya.cmds as cmds  # noqa: PLC0415

    root = cmds.group(empty=True, name=root_name)

    shading_groups = {}
    for page in scene.atlas_pages:
        tex_path = atlas_texture_paths[page.index]
        file_node = cmds.shadingNode("file", asTexture=True, isColorManaged=True, name=f"atlasFile{page.index}")
        cmds.setAttr(f"{file_node}.fileTextureName", tex_path, type="string")
        cmds.setAttr(f"{file_node}.alphaIsLuminance", False)

        place2d = cmds.shadingNode("place2dTexture", asUtility=True, name=f"atlasPlace2d{page.index}")
        cmds.connectAttr(f"{place2d}.outUV", f"{file_node}.uvCoord")
        cmds.connectAttr(f"{place2d}.outUvFilterSize", f"{file_node}.uvFilterSize")

        shader = cmds.shadingNode("lambert", asShader=True, name=f"atlasShader{page.index}")
        cmds.connectAttr(f"{file_node}.outColor", f"{shader}.color")
        cmds.connectAttr(f"{file_node}.outTransparency", f"{shader}.transparency")

        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"atlasSG{page.index}")
        cmds.connectAttr(f"{shader}.outColor", f"{sg}.surfaceShader", force=True)
        shading_groups[page.index] = sg

    for mesh in scene.meshes:
        transform_obj = _create_mesh_transform(mesh)

        transform_dag = om2.MFnDagNode(transform_obj)
        transform = transform_dag.name()

        cmds.xform(transform, worldSpace=True, translation=mesh.translate)
        cmds.rename(transform, mesh.maya_name)
        cmds.parent(mesh.maya_name, root)
        _tag_uv_affine(mesh.maya_name, mesh.uv_affine)

        if retopo:
            try:
                retopologize(mesh.maya_name, target_face_count, uv_affine=mesh.uv_affine)
            except Exception:
                # A silhouette polyRetopo can't handle (too few faces to work
                # with, or a sliver it collapses entirely) shouldn't abort the
                # whole rig -- keep the original guaranteed-quad mesh instead.
                # Maya also echoes its own error to the Script Editor before
                # raising, so the user sees it even though it's handled here.
                logger.warning(
                    "polyRetopo failed on %r; keeping the original traced topology",
                    mesh.maya_name,
                    exc_info=True,
                )
                if retopo_failures is not None:
                    retopo_failures.append(mesh.maya_name)

        shape = cmds.listRelatives(mesh.maya_name, shapes=True, fullPath=True)[0]
        cmds.sets(shape, edit=True, forceElement=shading_groups[mesh.atlas_page])

    return root


def run_headless(
    psd_path: str,
    out_dir: str,
    retopo: bool = False,
    target_face_count: int = 200,
    **pipeline_kwargs,
) -> str:
    """Convenience entry point for `mayapy -m psd2maya.maya_backend <psd> <out_dir>`.

    Runs the full standalone pipeline (parse -> pack -> build SceneData),
    saves the atlas PNG(s) to `out_dir`, builds the scene via `build_in_maya`,
    and saves a .ma next to the atlas. Requires mayapy; a plain `python3`
    interpreter has no `maya.standalone` module.
    """
    import os

    import maya.standalone  # noqa: PLC0415

    maya.standalone.initialize(name="python")
    import maya.cmds as cmds  # noqa: PLC0415

    from .pipeline import run_pipeline  # noqa: PLC0415 -- local import to avoid a cycle

    scene, atlas_paths = run_pipeline(psd_path, out_dir, **pipeline_kwargs)
    build_in_maya(scene, atlas_paths, retopo=retopo, target_face_count=target_face_count)

    out_ma = os.path.join(out_dir, os.path.splitext(os.path.basename(psd_path))[0] + ".ma")
    cmds.file(rename=out_ma)
    cmds.file(save=True, type="mayaAscii", force=True)
    return out_ma


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: mayapy -m psd2maya.maya_backend <input.psd> <out_dir>")
        raise SystemExit(1)
    saved = run_headless(sys.argv[1], sys.argv[2])
    print(f"Saved {saved}")
