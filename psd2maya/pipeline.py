"""Orchestrates stages 1-3 (parse -> pack -> build) and writes atlas PNGs to disk.

Shared by cli.py (standalone .ma output) and maya_backend.run_headless
(live Maya scene via mayapy) so both entry points parse/pack/build a PSD
identically and only differ in how they materialize the result.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from .atlas_packer import pack_layers
from .mesh_builder import build_scene
from .psd_reader import extract_layers
from .scene_model import SceneData


def run_pipeline(
    psd_path: str,
    out_dir: str,
    pixels_per_unit: float = 100.0,
    depth_step: float = 5.0,
    max_page_size: int = 4096,
    padding: int = 4,
    include_hidden: bool = False,
    detail_level: float = 2.0,
    alpha_threshold: int = 10,
    min_contour_area: float = 50.0,
    atlas_basename: str = "atlas",
    lod_by_name: Optional[Dict[str, str]] = None,
) -> Tuple[SceneData, Dict[int, str]]:
    os.makedirs(out_dir, exist_ok=True)

    layers, canvas_w, canvas_h = extract_layers(psd_path, include_hidden=include_hidden, lod_by_name=lod_by_name)
    atlas = pack_layers(layers, max_page_size=max_page_size, padding=padding)
    scene = build_scene(
        layers,
        atlas,
        canvas_w,
        canvas_h,
        pixels_per_unit=pixels_per_unit,
        depth_step=depth_step,
        detail_level=detail_level,
        alpha_threshold=alpha_threshold,
        min_contour_area=min_contour_area,
        source_psd=psd_path,
    )

    atlas_paths: Dict[int, str] = {}
    for page in atlas.pages:
        suffix = "" if len(atlas.pages) == 1 else f"_{page.index}"
        filename = f"{atlas_basename}{suffix}.png"
        full_path = os.path.abspath(os.path.join(out_dir, filename))
        page.image.save(full_path)
        # Absolute path: Maya resolves relative fileTextureName values against
        # the current project's sourceimages directory, not the .ma's own
        # folder, so a relative path here would silently break unless the
        # caller's Maya project happens to match `out_dir`.
        atlas_paths[page.index] = full_path

    return scene, atlas_paths
