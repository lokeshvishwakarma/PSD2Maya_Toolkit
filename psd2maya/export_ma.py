"""Backend A: emit a Maya ASCII (.ma) scene directly, no Maya installation required.

Only the minimal attribute set needed for Maya to reconstruct a static,
history-free poly mesh is written (`.vt`, `.ed`, `.fc`, and the `map1` UV
set). Maya-saved files usually also include component-index caches
(`.pd`/`.cd`/`.cvd`/`.hfd`) used to speed up construction-history
evaluation; those are safe to omit for a mesh with no history, since Maya
derives them on load.

Edge sharing
------------
Each `LayerMesh` is now a quad grid with edges shared between adjacent
faces (not one independent quad like the original bbox-card version), so
edges have to be deduplicated: Maya's `.ed` array stores each edge once as
a directed (start, end) vertex pair, and a face's `f` line in `.fc`
references edges *by index*, using Maya's one's-complement convention
(`-(edgeIndex + 1)`) when the face traverses a shared edge in the opposite
direction from how it was first stored. `_EdgeTable` below builds that
array and resolves each face's edge references accordingly.

One shader/file/place2dTexture/shadingEngine chain is created per atlas
page (there's normally exactly one page), and every mesh sourced from that
page is wired into that page's shadingEngine via instObjGroups ->
dagSetMembers -- the same mechanism Maya itself uses to store shading
assignment in a scene file (this is the file-format equivalent of running
`sets -e -forceElement` interactively).

Group hierarchy
----------------
Each mesh's `group_path` (see mesh_builder._resolve_group_path) is a chain
of already-unique, already Maya-safe names -- `_ensure_group_chain` just
needs to emit one `createNode transform` per name the first time it's
seen and remember that it has, so a folder shared by many meshes gets a
single group node reused by all of them rather than being recreated (and
Maya erroring on a duplicate node name) for every mesh underneath it.
Because every name in the whole scene is globally unique (see
mesh_builder), checking "have I created this name yet" needs only a flat
set, not a full path-tuple key.

This has been written to match Maya's documented ASCII format but has not
been round-tripped through a real Maya install in this environment (none is
available here) -- open the result in Maya and check the Outliner/UVs
before relying on it in production.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .scene_model import LayerMesh, SceneData


def _f(value: float) -> str:
    return f"{value:.6f}"


class _EdgeTable:
    """Dedups a quad mesh's edges and resolves each face to signed edge indices."""

    def __init__(self):
        self._edges: List[Tuple[int, int]] = []
        self._lookup: Dict[Tuple[int, int], int] = {}  # sorted (a,b) -> index into self._edges

    def edge_ref(self, a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        idx = self._lookup.get(key)
        if idx is None:
            idx = len(self._edges)
            self._edges.append((a, b))
            self._lookup[key] = idx
            return idx
        stored_a, _ = self._edges[idx]
        return idx if stored_a == a else -(idx + 1)

    def face_edge_refs(self, verts: Tuple[int, int, int, int]) -> List[int]:
        n = len(verts)
        return [self.edge_ref(verts[i], verts[(i + 1) % n]) for i in range(n)]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return self._edges


def _ensure_group_chain(w, group_path, created: set, root_name: str) -> str:
    """Emit any not-yet-created group transforms in `group_path`; return the innermost name."""
    parent = root_name
    for name in group_path:
        if name not in created:
            w(f'createNode transform -n "{name}" -p "{parent}";')
            created.add(name)
        parent = name
    return parent


def _write_mesh(w, mesh: LayerMesh, parent: str) -> None:
    x, y, z = mesh.translate
    transform_name = mesh.maya_name
    shape_name = f"{mesh.maya_name}Shape"

    w(f'createNode transform -n "{transform_name}" -p "{parent}";')
    w(f'	setAttr ".t" -type "double3" {_f(x)} {_f(y)} {_f(z)} ;')
    w(f'createNode mesh -n "{shape_name}" -p "{transform_name}";')
    w('	setAttr -k off ".v";')
    w('	setAttr ".vir" yes;')
    w('	setAttr ".vif" yes;')
    w('	setAttr ".uvst[0].uvsn" -type "string" "map1";')

    n_verts = len(mesh.vertices)
    uv_flat = " ".join(f"{_f(u)} {_f(v)}" for u, v in mesh.uvs)
    w(f'	setAttr ".uvst[0].uvsp[0:{n_verts - 1}]" -type "float2" {uv_flat};')
    w('	setAttr ".cuvs" -type "string" "map1";')
    w('	setAttr ".dcc" -type "string" "Ambient+Diffuse";')
    w('	setAttr ".covm[0]" 0 1 1;')
    w('	setAttr ".cdvm[0]" 0 1 1;')

    vt_flat = " ".join(f"{_f(vx)} {_f(vy)} {_f(vz)}" for vx, vy, vz in mesh.vertices)
    w(f'	setAttr ".vt[0:{n_verts - 1}]"  {vt_flat};')

    edge_table = _EdgeTable()
    face_edge_lists = [edge_table.face_edge_refs(face) for face in mesh.faces]

    n_edges = len(edge_table.edges)
    ed_flat = "  ".join(f"{a} {b} 0" for a, b in edge_table.edges)
    w(f'	setAttr ".ed[0:{n_edges - 1}]"  {ed_flat};')

    n_faces = len(mesh.faces)
    w(f'	setAttr -s {n_faces} ".fc[0:{n_faces - 1}]" -type "polyFaces" ')
    for face, edge_refs in zip(mesh.faces, face_edge_lists):
        w("		f 4 " + " ".join(str(e) for e in edge_refs))
        w("		mu 0 4 " + " ".join(str(v) for v in face))
    w("		;")


def write_ma(scene: SceneData, out_path: str, atlas_texture_paths: Dict[int, str]) -> None:
    lines = []
    w = lines.append

    w('//Maya ASCII scene generated by psd2maya')
    if scene.source_psd:
        w(f'//Source PSD: {scene.source_psd}')
    w('requires maya "2018";')
    w('currentUnit -l centimeter -a degree -t film;')
    w('createNode transform -n "psd2maya_root";')
    w('	setAttr ".rp" -type "double3" 0 0 0 ;')

    for page in scene.atlas_pages:
        tex_path = atlas_texture_paths[page.index].replace("\\", "/")
        shader, file_node, place2d, sg = (
            f"atlasShader{page.index}",
            f"atlasFile{page.index}",
            f"atlasPlace2d{page.index}",
            f"atlasSG{page.index}",
        )
        w(f'createNode file -n "{file_node}";')
        w(f'	setAttr ".fileTextureName" -type "string" "{tex_path}";')
        w('	setAttr ".alphaIsLuminance" no;')
        w(f'createNode place2dTexture -n "{place2d}";')
        w(f'connectAttr "{place2d}.outUV" "{file_node}.uvCoord";')
        w(f'connectAttr "{place2d}.outUvFilterSize" "{file_node}.uvFilterSize";')
        w(f'createNode lambert -n "{shader}";')
        w(f'connectAttr "{file_node}.outColor" "{shader}.color";')
        w(f'connectAttr "{file_node}.outTransparency" "{shader}.transparency";')
        w(f'createNode shadingEngine -n "{sg}" -s;')
        w('	setAttr ".ihi" 0;')
        w('	setAttr ".ro" yes;')
        w(f'connectAttr "{shader}.outColor" "{sg}.surfaceShader";')

    sg_member_counters = {page.index: 0 for page in scene.atlas_pages}
    sg_connections = []
    created_groups: set = set()

    for mesh in scene.meshes:
        parent = _ensure_group_chain(w, mesh.group_path, created_groups, "psd2maya_root")
        _write_mesh(w, mesh, parent)

        sg = f"atlasSG{mesh.atlas_page}"
        idx = sg_member_counters[mesh.atlas_page]
        sg_member_counters[mesh.atlas_page] += 1
        sg_connections.append(
            f'connectAttr "{mesh.maya_name}Shape.instObjGroups[0]" "{sg}.dagSetMembers[{idx}]";'
        )

    lines.extend(sg_connections)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
