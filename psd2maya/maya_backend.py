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
"""

from __future__ import annotations

from typing import Dict

from .scene_model import LayerMesh, SceneData


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


def build_in_maya(
    scene: SceneData,
    atlas_texture_paths: Dict[int, str],
    root_name: str = "psd2maya_root",
) -> str:
    """Build `scene` in the currently open Maya session. Returns the root transform's name."""
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

        shape = cmds.listRelatives(mesh.maya_name, shapes=True, fullPath=True)[0]
        cmds.sets(shape, edit=True, forceElement=shading_groups[mesh.atlas_page])

    return root


def run_headless(psd_path: str, out_dir: str, **pipeline_kwargs) -> str:
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
    build_in_maya(scene, atlas_paths)

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
