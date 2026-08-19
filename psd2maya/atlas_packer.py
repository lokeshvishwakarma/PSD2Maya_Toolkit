"""Stage 2: pack every layer's cropped pixels into one or more atlas pages.

Background-art layers are typically each close to full-canvas size, so this
is not a bin-packing problem in the "save space" sense -- there's little
overlap to exploit. The packer's job is instead just to lay images out
without overlapping, split across multiple pages if they don't fit within
`max_page_size`, and leave a padded, bleed-extended border around each so
texture filtering/mipmapping doesn't smear one layer's edge into its
neighbor's.

Algorithm: a simple shelf packer (sort tallest-first, fill left-to-right
rows, start a new row when one wouldn't fit, start a new page when no row
has vertical room left). This is O(n log n) and easy to reason about; a
tighter packer (MaxRects, guillotine) would only pay off if layers were
numerous and small, which isn't the expected input for this tool.
"""

from __future__ import annotations

from typing import List

from PIL import Image

from .scene_model import AtlasPage, AtlasResult, PackedLayer, SourceLayer


class _Shelf:
    __slots__ = ("y", "height", "cursor_x")

    def __init__(self, y: int):
        self.y = y
        self.height = 0
        self.cursor_x = 0


class _Page:
    def __init__(self, index: int, size: int):
        self.index = index
        self.size = size
        self.shelves: List[_Shelf] = []
        self.placements: List[PackedLayer] = []

    def try_place(self, layer: SourceLayer, padded_w: int, padded_h: int) -> bool:
        for shelf in self.shelves:
            if shelf.cursor_x + padded_w <= self.size and shelf.height >= padded_h:
                self._commit(layer, shelf, padded_w, padded_h)
                return True
        # start a new shelf below the last one
        next_y = self.shelves[-1].y + self.shelves[-1].height if self.shelves else 0
        if next_y + padded_h > self.size:
            return False
        shelf = _Shelf(next_y)
        self.shelves.append(shelf)
        self._commit(layer, shelf, padded_w, padded_h)
        return True

    def _commit(self, layer: SourceLayer, shelf: _Shelf, padded_w: int, padded_h: int):
        x, y = shelf.cursor_x, shelf.y
        shelf.cursor_x += padded_w
        shelf.height = max(shelf.height, padded_h)
        self.placements.append(
            PackedLayer(
                source=layer,
                page=self.index,
                atlas_x=x,
                atlas_y=y,
                atlas_w=layer.width,
                atlas_h=layer.height,
            )
        )


def _bleed_extend(image: Image.Image, padding: int) -> Image.Image:
    """Pad `image` on all sides by `padding` px, replicating edge pixels.

    Without this, bilinear filtering at a shell's border samples the
    transparent padding and produces a visible dark/light fringe.
    """
    if padding == 0:
        return image
    w, h = image.size
    out = Image.new("RGBA", (w + 2 * padding, h + 2 * padding), (0, 0, 0, 0))
    out.paste(image, (padding, padding))
    # edges
    out.paste(image.crop((0, 0, w, 1)).resize((w, padding)), (padding, 0))
    out.paste(image.crop((0, h - 1, w, h)).resize((w, padding)), (padding, padding + h))
    out.paste(image.crop((0, 0, 1, h)).resize((padding, h)), (0, padding))
    out.paste(image.crop((w - 1, 0, w, h)).resize((padding, h)), (padding + w, padding))
    # corners
    out.paste(image.crop((0, 0, 1, 1)).resize((padding, padding)), (0, 0))
    out.paste(image.crop((w - 1, 0, w, 1)).resize((padding, padding)), (padding + w, 0))
    out.paste(image.crop((0, h - 1, 1, h)).resize((padding, padding)), (0, padding + h))
    out.paste(image.crop((w - 1, h - 1, w, h)).resize((padding, padding)), (padding + w, padding + h))
    return out


def pack_layers(
    layers: List[SourceLayer],
    max_page_size: int = 4096,
    padding: int = 4,
) -> AtlasResult:
    """Pack `layers` into square-ish atlas pages no larger than `max_page_size`.

    Raises ValueError if a single layer's (padded) footprint exceeds
    `max_page_size` on either axis -- the caller should raise max_page_size
    or pre-downsample source art in that case.
    """
    ordered = sorted(layers, key=lambda l: l.height, reverse=True)

    for l in ordered:
        if l.width + 2 * padding > max_page_size or l.height + 2 * padding > max_page_size:
            raise ValueError(
                f"Layer {l.name!r} ({l.width}x{l.height}) does not fit within "
                f"max_page_size={max_page_size} even alone; increase --max-page-size."
            )

    pages: List[_Page] = [_Page(0, max_page_size)]
    for layer in ordered:
        padded_w, padded_h = layer.width + 2 * padding, layer.height + 2 * padding
        placed = False
        for page in pages:
            if page.try_place(layer, padded_w, padded_h):
                placed = True
                break
        if not placed:
            page = _Page(len(pages), max_page_size)
            pages.append(page)
            if not page.try_place(layer, padded_w, padded_h):
                raise AssertionError("layer rejected by a freshly created empty page")

    atlas_pages: List[AtlasPage] = []
    all_placements: List[PackedLayer] = []
    for page in pages:
        used_h = page.shelves[-1].y + page.shelves[-1].height if page.shelves else 0
        used_w = max((s.cursor_x for s in page.shelves), default=0)
        page_h = _next_pow2_or_exact(used_h, cap=max_page_size)
        page_w = _next_pow2_or_exact(used_w, cap=max_page_size)
        canvas = Image.new("RGBA", (page_w, page_h), (0, 0, 0, 0))
        for placement in page.placements:
            bled = _bleed_extend(placement.source.pixels, padding)
            canvas.paste(bled, (placement.atlas_x, placement.atlas_y), bled)
            # the placement's stored atlas_x/y should point at the *unpadded*
            # image origin, which sits `padding` px into the bled block.
            placement.atlas_x += padding
            placement.atlas_y += padding
            all_placements.append(placement)
        atlas_pages.append(AtlasPage(index=page.index, width=page_w, height=page_h, image=canvas))

    # restore original layer order (pack_layers sorted tallest-first for packing efficiency)
    order_lookup = {id(l): i for i, l in enumerate(layers)}
    all_placements.sort(key=lambda p: order_lookup[id(p.source)])

    return AtlasResult(pages=atlas_pages, placements=all_placements)


def _next_pow2_or_exact(value: int, cap: int) -> int:
    """Round up to a power-of-two <= cap for GPU-friendly atlas dimensions.

    `value` is guaranteed by the shelf packer to already be <= cap; if
    rounding up to the next power of two would exceed cap, we keep the
    exact (non-power-of-two) value instead of overflowing the page.
    """
    if value <= 0:
        return 1
    p = 1
    while p < value:
        p *= 2
    return p if p <= cap else value
