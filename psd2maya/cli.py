"""Standalone command-line entry point: `python3 -m psd2maya <input.psd> [options]`.

Runs the full parse -> pack -> build -> write-.ma pipeline with no Maya
installation required. For running inside Maya/mayapy instead, see
maya_backend.py.
"""

from __future__ import annotations

import argparse
import os
import sys

from .export_ma import write_ma
from .pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="psd2maya",
        description="Recreate a layered PSD's background art as a Maya parallax card rig: "
        "one plane per layer, a packed texture atlas, and a .ma scene file.",
    )
    p.add_argument("psd_path", help="Path to the source .psd file")
    p.add_argument("-o", "--out-dir", default="./psd2maya_out", help="Output directory (default: %(default)s)")
    p.add_argument(
        "--pixels-per-unit",
        type=float,
        default=100.0,
        help="PSD pixels per 1 Maya world unit (default: %(default)s)",
    )
    p.add_argument(
        "--depth-step",
        type=float,
        default=5.0,
        help="Maya units between successive layers along Z; negative reverses the "
        "front/back direction (default: %(default)s)",
    )
    p.add_argument(
        "--max-page-size",
        type=int,
        default=4096,
        help="Max atlas page width/height in pixels before spilling to a new page (default: %(default)s)",
    )
    p.add_argument(
        "--padding",
        type=int,
        default=4,
        help="Bleed-extended border in pixels around each packed layer, to avoid filtering seams (default: %(default)s)",
    )
    p.add_argument(
        "--include-hidden",
        action="store_true",
        help="Also rasterize layers that are hidden in Photoshop (default: skip them)",
    )
    p.add_argument(
        "--detail-level",
        type=float,
        default=2.0,
        help="Douglas-Peucker epsilon in pixels for simplifying the traced silhouette: "
        "lower hugs the outline more tightly (more vertices), higher simplifies more "
        "aggressively (fewer vertices) (default: %(default)s)",
    )
    p.add_argument(
        "--alpha-threshold",
        type=int,
        default=10,
        help="Raw 0-255 alpha cutoff below which a pixel counts as background when "
        "tracing a layer's silhouette (default: %(default)s)",
    )
    p.add_argument(
        "--min-contour-area",
        type=float,
        default=50.0,
        help="Ignore traced silhouette regions smaller than this many pixels^2 "
        "(filters out compression/anti-aliasing dust) (default: %(default)s)",
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        scene, atlas_paths = run_pipeline(
            args.psd_path,
            args.out_dir,
            pixels_per_unit=args.pixels_per_unit,
            depth_step=args.depth_step,
            max_page_size=args.max_page_size,
            padding=args.padding,
            include_hidden=args.include_hidden,
            detail_level=args.detail_level,
            alpha_threshold=args.alpha_threshold,
            min_contour_area=args.min_contour_area,
        )
    except Exception as exc:  # surface a clean CLI error instead of a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ma_path = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.psd_path))[0] + ".ma")
    write_ma(scene, ma_path, atlas_paths)

    print(f"Layers recreated : {len(scene.meshes)}")
    for mesh in scene.meshes:
        print(f"  {mesh.source_name!r}: {len(mesh.faces)} quads, {len(mesh.vertices)} verts")
    print(f"Atlas page(s)    : {len(scene.atlas_pages)}")
    for page_index, path in atlas_paths.items():
        page = next(p for p in scene.atlas_pages if p.index == page_index)
        print(f"  page {page_index}: {path} ({page.width}x{page.height})")
    print(f"Maya scene       : {ma_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
