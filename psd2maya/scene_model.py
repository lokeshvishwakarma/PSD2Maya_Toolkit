"""Plain-data structures shared by every stage of the pipeline.

Nothing in this module imports psd-tools, Pillow, or maya.cmds. Keeping it
dependency-free is what lets `mesh_builder` output feed either the
standalone `.ma` writer or the in-Maya backend without duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceLayer:
    """One rasterized PSD layer, already cropped to its own bounding box."""

    name: str
    stack_index: int  # 0 = bottom-most layer in the PSD (see psd_reader docstring)
    left: int
    top: int
    right: int
    bottom: int
    opacity: float  # 0..1
    pixels: object  # PIL.Image.Image (RGBA), cropped to (left, top, right, bottom)
    # "High"/"Mid"/"Low", assigned per-layer in the UI's LOD column (ui.py) and
    # threaded through by psd_reader.extract_layers' lod_by_name lookup; drives
    # maya_backend.build_in_maya's per-mesh polyRetopo target face count when
    # retopo=True. Defaults to "Mid" for anything not explicitly tagged (the
    # CLI/mayapy entry points have no per-layer UI, so everything they build
    # falls back to this).
    lod: str = "Mid"

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass
class PackedLayer:
    """Where a SourceLayer's pixels landed inside an atlas page."""

    source: SourceLayer
    page: int
    atlas_x: int
    atlas_y: int
    atlas_w: int
    atlas_h: int


@dataclass
class AtlasPage:
    index: int
    width: int
    height: int
    image: object  # PIL.Image.Image (RGBA)


@dataclass
class AtlasResult:
    pages: list  # list[AtlasPage]
    placements: list  # list[PackedLayer], one per source layer


@dataclass
class LayerMesh:
    """A quad-only polygon mesh trimmed to a layer's alpha silhouette.

    Geometry is stored local to `translate` (the layer's bbox center in
    Maya world space), so every vertex tuple is a small offset around the
    origin rather than an absolute world position.
    """

    maya_name: str
    source_name: str
    stack_index: int
    translate: tuple  # (x, y, z) in Maya world units -- the mesh's local origin
    atlas_page: int
    vertices: list  # list[(x, y, z)], local space, z is always 0.0 (planar)
    uvs: list  # list[(u, v)], parallel to `vertices`, in the atlas page's 0..1 space
    faces: list  # list[(v0, v1, v2, v3)], CCW winding, indices into vertices/uvs
    # Closed-form (u, v) = (a*local_x + b*local_y + c, d*local_x + e*local_y + f).
    # Because the mesh is planar and the pixel->atlas mapping is affine, this
    # reproduces `uvs` exactly from a vertex's local position alone -- which means
    # UVs can be recomputed for *any* topology (e.g. after Maya's polyRetopo
    # replaces every vertex, or after a manual Layout UV pass re-transforms the
    # shell) without projecting or transferring from a source mesh. The full
    # 6-parameter form (rather than a simpler axis-aligned u=f(x), v=f(y) form)
    # is what lets it also represent a UV shell that's been rotated, not just
    # translated/scaled -- see maya_backend.reproject_uvs and
    # relayout.rebuild_textures_from_uv_layout.
    uv_affine: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # (a, b, c, d, e, f)
    lod: str = "Mid"  # copied from the source SourceLayer.lod; see that field's docstring


@dataclass
class SceneData:
    meshes: list  # list[LayerMesh], in stack order (index 0 = bottom-most)
    atlas_pages: list  # list[AtlasPage]
    pixels_per_unit: float
    canvas_width: int
    canvas_height: int
    depth_step: float
    source_psd: str = ""
