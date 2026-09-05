"""Cheap PSD layer-tree introspection for UI display.

Unlike `psd_reader.extract_layers` (which composites every rasterizable
layer to build meshes from), this only reads layer metadata and walks the
PSD's native group hierarchy -- no compositing, so it's cheap enough to
re-run every time the user picks a file in the UI just to populate a tree
widget.

Iterating a `PSDImage` or `Group` with `for layer in container` yields its
*direct* children in on-disk order, which is bottom-to-top (same convention
as `psd_reader.extract_layers`'s docstring) -- the opposite of how
Photoshop's Layers panel lists them (topmost layer at the top of the list).
`read_layer_tree` reverses each level so the returned tree matches what the
user actually sees in Photoshop.

Each `LayerNode` also carries `source`, the underlying psd-tools
Layer/Group it was built from. That's the one deliberate crack in this
module's "metadata only, no compositing" rule: stashing the handle costs
nothing by itself (`.composite()` is never called here), but it's what
lets a caller -- namely `ui.py`'s preview panel -- rasterize on demand
just the *one* layer or group the user actually clicked, instead of either
compositing all of them upfront (defeats the point of this module existing
at all) or re-opening and re-walking the whole PSD from scratch for every
click. `PSDImage.open` keeps the file handle around for lazy pixel access,
so `source` stays valid as long as the caller keeps this tree (or the
`PSDImage` it came from) alive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from psd_tools import PSDImage


@dataclass
class LayerNode:
    name: str
    kind: str  # 'group', 'pixel', 'type', 'shape', 'smartobject', ...
    is_group: bool
    visible: bool
    opacity: float  # 0..1
    width: int
    height: int
    children: List["LayerNode"] = field(default_factory=list)
    source: Optional[object] = None  # the psd-tools Layer/Group itself; see module docstring
    # User-assigned Level of Detail hint ("High"/"Mid"/"Low"), purely a UI-side
    # tag right now -- see ui.py's LOD column -- not read by anything in the
    # build pipeline (psd_reader/mesh_builder/pipeline) yet.
    lod: str = "Mid"


def read_layer_tree(psd_path: str) -> Tuple[List[LayerNode], int, int]:
    """Return (top_level_nodes, canvas_width, canvas_height) in Photoshop panel order."""
    psd = PSDImage.open(psd_path)

    def walk(container) -> List[LayerNode]:
        nodes = [
            LayerNode(
                name=layer.name or "Layer",
                kind=layer.kind,
                is_group=layer.is_group(),
                visible=bool(layer.visible),
                opacity=getattr(layer, "opacity", 255) / 255.0,
                width=getattr(layer, "width", 0) or 0,
                height=getattr(layer, "height", 0) or 0,
                children=walk(layer) if layer.is_group() else [],
                source=layer,
            )
            for layer in container
        ]
        nodes.reverse()
        return nodes

    return walk(psd), psd.width, psd.height
