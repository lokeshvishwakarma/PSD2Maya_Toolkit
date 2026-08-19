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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from psd_tools import PSDImage


@dataclass
class LayerNode:
    name: str
    kind: str  # 'group', 'pixel', 'type', 'shape', 'smartobject', ...
    is_group: bool
    visible: bool
    opacity: float  # 0..1
    children: List["LayerNode"] = field(default_factory=list)


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
                children=walk(layer) if layer.is_group() else [],
            )
            for layer in container
        ]
        nodes.reverse()
        return nodes

    return walk(psd), psd.width, psd.height
