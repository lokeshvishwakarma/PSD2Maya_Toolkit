"""Fast approximate thumbnail compositing for the UI's layer preview panel.

A single leaf layer's own `.composite()` is already fast -- it's bbox-limited,
so it only ever decodes/blends that one layer. A *group's* `.composite()` is a
different story: it alpha/blend-mode-composites every descendant leaf at full
PSD resolution, even though the preview panel immediately scales the result
down to ~180px. On a real 41-leaf production background group that took
6-7 seconds to produce an image this UI was always going to shrink by ~15x.

`render_group_thumbnail` fixes the actual mismatch (full-resolution work for a
tiny on-screen result) rather than trying to make the expensive path less
expensive: it decodes each leaf itself (still bbox-limited, so still cheap
per leaf), shrinks that leaf immediately, and only then pastes the small
result onto a small shared canvas -- so the "combine everything" step, which
is what scales with layer count and canvas size, operates on thumbnail-sized
images throughout instead of full-resolution ones.

Each leaf's decode is also independent of every other leaf's, so they're
farmed out to a thread pool: `leaf.composite()` is CPU-bound Python/Pillow
work, but enough of it (image decompression, resampling) happens in C with
the GIL released that real overlap is achievable -- measured 3x on top of the
downscale-first change alone, ~13-16x faster overall than the naive
`group.composite()` then shrink.

Approximation, not a substitute for `layer.composite()`: every leaf is
composited as a plain alpha-over with only its own effective opacity applied
(a nested group's opacity multiplies into every descendant leaf rather than
being applied once to that group's own pre-composited result, which is what
Photoshop actually does) -- blend modes (multiply, screen, ...), layer masks,
clipping, and adjustment layers are all ignored. Verified side by side
against the real production PSD's two top-level groups: visually
indistinguishable at thumbnail size, because this background art's layers
all happen to use Normal blend mode -- a PSD leaning on multiply/screen
layers or masks for its look would show more daylight between this and the
exact composite. Fine for "which layer is this" in a tree view; not a
stand-in for an accurate render, and not used for anything that ends up in
the actual Maya scene (only `ui.py`'s preview panel calls this; the real
build pipeline never does).

`render_group_thumbnail` (one group's own descendants) and
`render_nodes_thumbnail` (an arbitrary, possibly unrelated, multi-selection
from the UI's tree -- e.g. three ctrl-clicked layers scattered across
different folders) both reduce to the same problem once they've each built
their own flat (leaf, effective_opacity) list: decode+shrink+position every
leaf against one shared bounding box. `_composite_entries` is that shared
core. For a multi-selection, "one shared bounding box" is the union of the
selected leaves' own bboxes rather than one group's already-known bbox, so
each selected layer still renders at its true position *relative to the
others* -- not just stacked arbitrarily -- which is what makes it a
meaningful preview of "what am I about to bulk-tag" rather than a random
collage.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

_MAX_WORKERS = 8


def _iter_visible_leaves(container, opacity: float = 1.0) -> Iterator[Tuple[object, float]]:
    """Yield (leaf_layer, effective_opacity) for every visible leaf under `container`.

    A hidden layer's entire subtree is skipped -- matching Photoshop, where a
    hidden group's children never render regardless of their own visibility
    flag -- and order follows `container`'s own (bottom-to-top, see
    psd_reader.py's docstring) child order, so later alpha_composite calls
    naturally paint later leaves on top.
    """
    for layer in container:
        if not layer.visible:
            continue
        layer_opacity = opacity * (getattr(layer, "opacity", 255) / 255.0)
        if layer.is_group():
            yield from _iter_visible_leaves(layer, layer_opacity)
        else:
            yield layer, layer_opacity


def _decode_and_shrink(entry: Tuple[object, float], scale: float, origin: Tuple[int, int]):
    """Runs in a worker thread: composite+shrink one leaf, or None if it can't contribute."""
    leaf, opacity = entry
    try:
        image = leaf.composite()
    except Exception:
        logger.warning("Thumbnail: failed to composite leaf %r; skipping it", getattr(leaf, "name", leaf), exc_info=True)
        return None
    if image is None:
        return None
    image = image.convert("RGBA")

    tw, th = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    thumb = image.resize((tw, th), Image.BILINEAR)

    if opacity < 1.0:
        r, g, b, a = thumb.split()
        a = a.point(lambda v: int(v * opacity))
        thumb = Image.merge("RGBA", (r, g, b, a))

    left, top = origin
    lb = leaf.bbox
    px, py = round((lb[0] - left) * scale), round((lb[1] - top) * scale)
    return thumb, px, py


def _has_area(bbox) -> bool:
    return bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _union_bbox(entries: List[Tuple[object, float]]) -> Tuple[int, int, int, int]:
    lefts, tops, rights, bottoms = zip(*(leaf.bbox for leaf, _ in entries))
    return min(lefts), min(tops), max(rights), max(bottoms)


def _composite_entries(
    entries: List[Tuple[object, float]], max_size: int, max_workers: int
) -> Optional[Image.Image]:
    """Shared core: decode+shrink+position every (leaf, opacity) entry against their union bbox."""
    entries = [(leaf, opacity) for leaf, opacity in entries if _has_area(leaf.bbox)]
    if not entries:
        return None

    left, top, right, bottom = _union_bbox(entries)
    full_w, full_h = right - left, bottom - top
    scale = min(1.0, max_size / max(full_w, full_h))
    canvas = Image.new("RGBA", (max(1, round(full_w * scale)), max(1, round(full_h * scale))), (0, 0, 0, 0))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda e: _decode_and_shrink(e, scale, (left, top)), entries))

    for result in results:
        if result is None:
            continue
        thumb, px, py = result
        canvas.alpha_composite(thumb, (px, py))

    return canvas


def render_group_thumbnail(group, max_size: int = 256, max_workers: int = _MAX_WORKERS) -> Optional[Image.Image]:
    """Approximate composite of `group`'s visible leaves, downscaled to `max_size` on its long side.

    Returns None if `group` has no visible content (empty/zero-area bbox --
    e.g. every child is hidden), matching `leaf.composite()`'s own None
    convention for "nothing to show" so callers can handle both the same way.
    """
    return _composite_entries(list(_iter_visible_leaves(group)), max_size, max_workers)


def render_nodes_thumbnail(nodes, max_size: int = 256, max_workers: int = _MAX_WORKERS) -> Optional[Image.Image]:
    """Approximate composite of an arbitrary, possibly unrelated multi-selection.

    `nodes` is any iterable of objects with `.is_group` (bool) and `.source`
    (the psd-tools Layer/Group) -- duck-typed against `layer_tree.LayerNode`
    rather than importing it, since this module otherwise has no UI/tree
    dependency. A selected group contributes its visible descendant leaves
    (`_iter_visible_leaves`, so a hidden child inside it is still skipped);
    a selected leaf contributes itself directly using its *own* opacity,
    regardless of its own visibility flag -- consistent with this UI's
    existing single-leaf preview behavior of showing exactly what you
    explicitly clicked on, hidden or not. Returns None if nothing in the
    selection has any visible content to show.
    """
    entries: List[Tuple[object, float]] = []
    for node in nodes:
        if node.source is None:
            continue
        if node.is_group:
            entries.extend(_iter_visible_leaves(node.source))
        else:
            entries.append((node.source, getattr(node.source, "opacity", 255) / 255.0))

    return _composite_entries(entries, max_size, max_workers)
