"""Stage 1: turn a .psd file into a flat list of SourceLayer objects.

Ordering guarantee
-------------------
The PSD file format stores layer records bottom-to-top (the first record in
the file is the layer furthest back; the last is the one nearest the
viewer) -- see the Adobe Photoshop File Format spec, "Layer records" section.
psd-tools' `PSDImage.descendants()` walks the parsed tree in that same
on-disk order without reversing it, so the first layer this module yields is
the bottom-most one in the Photoshop layers panel and the last is the
topmost. `mesh_builder` relies on that to place background layers further
from camera by default.

Groups themselves are transparent: only rasterizable layers (pixel, text,
shape, smart object, adjustment-with-pixels, ...) are returned. A group's
own visibility/opacity already gates its children in psd-tools' compositing,
so we don't need to special-case folders here beyond skipping the group
node itself.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from psd_tools import PSDImage

from .scene_model import SourceLayer

logger = logging.getLogger(__name__)


def extract_layers(
    psd_path: str,
    include_hidden: bool = False,
    lod_by_name: Optional[Dict[str, str]] = None,
) -> Tuple[List[SourceLayer], int, int]:
    """Read `psd_path` and return (layers, canvas_width, canvas_height).

    `layers` is ordered bottom-to-top (see module docstring) and contains
    only layers that (a) are visible (unless include_hidden), (b) have a
    non-empty bounding box, and (c) composite to at least one non-transparent
    pixel. Empty/fully-transparent layers are skipped with a debug log since
    they'd produce a degenerate, invisible plane in Maya.

    `lod_by_name` maps a PSD layer's own name to a "High"/"Mid"/"Low" tag
    (see `ui.py`'s LOD column); a layer not present in it -- including every
    layer, when this is None -- gets `SourceLayer`'s own default ("Mid").
    Keyed by plain name rather than any more precise identity, same caveat
    as `relayout.py`'s `layers_by_name`: two layers sharing a name are
    indistinguishable to this lookup and would get the same tag.
    """
    psd = PSDImage.open(psd_path)
    canvas_w, canvas_h = psd.width, psd.height

    layers: List[SourceLayer] = []
    stack_index = 0
    skipped = []

    for layer in psd.descendants():
        if layer.is_group():
            continue
        if not include_hidden and not layer.visible:
            skipped.append((layer.name, "hidden"))
            continue

        bbox = layer.bbox
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            skipped.append((layer.name, "empty bbox"))
            continue

        image = layer.composite()
        if image is None:
            skipped.append((layer.name, "failed to composite (unsupported layer kind)"))
            continue

        image = image.convert("RGBA")
        if image.getbbox() is None:
            skipped.append((layer.name, "fully transparent"))
            continue

        opacity = getattr(layer, "opacity", 255) / 255.0
        layers.append(
            SourceLayer(
                name=layer.name or f"layer_{stack_index}",
                stack_index=stack_index,
                left=bbox[0],
                top=bbox[1],
                right=bbox[2],
                bottom=bbox[3],
                opacity=opacity,
                pixels=image,
                lod=(lod_by_name or {}).get(layer.name, "Mid"),
            )
        )
        stack_index += 1

    for name, reason in skipped:
        logger.info("Skipped layer %r: %s", name, reason)

    if not layers:
        raise ValueError(f"No rasterizable layers found in {psd_path!r}")

    return layers, canvas_w, canvas_h
