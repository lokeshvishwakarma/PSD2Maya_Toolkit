# psd2maya

Recreates a layered Photoshop background as a Maya parallax rig: one
quad-only polygon mesh per PSD layer, traced from that layer's actual
alpha silhouette (not just its bounding box), staggered in depth by paint
order, all textured from a single packed atlas PNG with matching UVs.

## Architecture

```
 .psd file
     |
     v
 psd_reader.py     -- psd-tools         --> list[SourceLayer]
     |                                       (name, bbox, cropped RGBA pixels,
     |                                        opacity, stack order)
     v
 atlas_packer.py   -- Pillow only       --> AtlasResult
     |                                       (1+ atlas pages, per-layer
     |                                        placement rect, bleed padding)
     v
 contour_tracer.py -- OpenCV            --> polygon(s) per layer
     |                                       (alpha threshold -> findContours
     |                                        -> approxPolyDP simplify)
     v
 quadrangulate.py  -- pure Python       --> quad-only mesh per polygon
     |                                       (ear-clip triangulate, then split
     |                                        every triangle into 3 quads)
     v
 mesh_builder.py   -- pure Python       --> SceneData
     |                                       (world-placed, UV-mapped,
     |                                        winding-corrected quad meshes)
     v
   +-------------------------+-------------------------------+
   |                         |
   v                         v
 export_ma.py              maya_backend.py
 (pure Python,              (Maya Python API, needs
  no Maya needed)            Maya or mayapy)
   |                         |
   v                         v
 scene.ma file           live Maya scene
 + atlas PNG(s)           + atlas PNG(s)
```

`pipeline.py` wires stages 1-5 together and saves the atlas PNG(s) to disk;
`cli.py` is the standalone entry point (ending at `export_ma`);
`maya_backend.py`'s `run_headless()` is the mayapy entry point (ending at
`build_in_maya`). Neither backend needs the other -- `scene_model.py`'s
dataclasses are the only thing they share.

### Key design decisions

- **Traced silhouette, not a grid or a bounding box.** Each layer's alpha
  channel is thresholded and traced with OpenCV (`findContours` +
  `approxPolyDP`, the same recipe as tracing a PNG cutout by hand) to get
  a simplified polygon that hugs the actual artwork -- `--detail-level`
  controls the Douglas-Peucker epsilon (lower hugs tighter, more
  vertices). A layer with more than one disconnected opaque region (two
  separate rocks, say) traces to more than one shell, and each shell
  becomes its own mesh rather than silently merging or dropping one.
  Tracing uses `RETR_EXTERNAL`, so a literal hole in a layer's art (a
  ring shape) will come back filled solid -- there's no interior-hole
  support.
- **Self-crossing traced boundaries are repaired before triangulation.**
  Where the art has a roughly one-pixel-wide neck or spur, `findContours`
  traces out and back within a pixel of itself, so the contour touches or
  crosses itself. This is *not* an `approxPolyDP` artifact -- it's in the
  raw contour too, so lowering `--detail-level` doesn't help (verified: a
  spur still crosses at epsilon 0). Ear-clipping a self-crossing boundary
  emits overlapping faces, which makes the surface locally
  non-orientable: two same-winding faces share a directed edge. Maya
  reports that as nonmanifold and `polyRetopo` refuses to run
  ("PolyRetopo does not work on polygonal object with nonmanifold
  geometry"). `contour_tracer._remove_self_intersections` cuts the ring at
  each crossing and keeps the larger loop. On the 50-layer production PSD
  this touched 2 of 68 shells, removing 3 and 2 vertices for an area loss
  of 0.012% and 0.0007% -- the discarded loops are degenerate slivers, so
  the visible silhouette is unchanged. Afterwards all 68 shells are simple,
  no mesh is nonmanifold, and `polyRetopo` succeeds on every one.
- **Guaranteed-quad remeshing, not `polyTriangulate`+`polyQuad`.** Maya's
  own quadrangulate command is a best-effort merge of adjacent triangle
  pairs and does not guarantee it can pair off every triangle -- for an
  arbitrary traced silhouette it can leave stray triangles or n-gons
  behind, which would violate "only quads". Instead, `quadrangulate.py`
  ear-clip triangulates the traced polygon and then splits *every*
  triangle into exactly 3 quads via its centroid and edge midpoints. This
  is unconditional for any triangle, so the result is 100% quads by
  construction regardless of the input shape, and it works identically
  whether the output goes through the standalone `.ma` writer or the live
  Maya backend -- no Maya needs to be running to get an all-quad mesh.
  The boundary vertices are untouched by this step, so the mesh's outer
  edge still matches the traced silhouette exactly; only the interior
  gets extra vertices. Tradeoff: face count is 3x the triangle count, so
  a very tightly-traced (low `--detail-level`) boundary can produce a lot
  of geometry -- raise `--detail-level` to simplify the boundary first if
  that matters.
- **Optional Maya-native retopology, with exact UVs (not Transfer
  Attributes).** `build_in_maya(..., retopo=True)` (the "Use Maya
  polyRetopo" checkbox in the UI) replaces the ear-clip topology with
  `cmds.polyRetopo`'s more uniform quad flow. Remeshing normally destroys
  UVs, and the standard fix is `cmds.transferAttributes` from a saved copy
  of the original -- a world-space raycast projection that introduces error
  and leaves jagged UV borders where the new topology crosses a seam. This
  package doesn't need that: because each mesh is planar and the
  pixel->atlas mapping is affine, the whole mapping collapses to four
  numbers, `(u, v) = (u0 + su*x, v0 + sv*y)` (see
  `mesh_builder._uv_affine`). `maya_backend.reproject_uvs` evaluates that
  per vertex, so UVs come back **exact at any topology** with no source
  mesh, no projection, and no seam repair. The mapping is also stored on
  each transform as locked `psdUv*` attributes, so you can call
  `reproject_uvs("layerName")` yourself after any further hand remeshing.
  Verified on a real 3840x2139, 50-layer production PSD: 100% quads before
  and after, UV error at float32 precision (~3e-8), every UV inside 0..1,
  and the traced silhouette held to under 1% of each mesh's span.
  Tradeoffs: `polyRetopo` costs roughly 0.5s per mesh (~33s for 68
  shells), and `targetFaceCount` is applied per mesh, so simple layers get
  *padded up* to the target -- on that production file a target of 200 took
  the scene from 9,876 faces to 16,254. Lower the target, or leave the
  checkbox off, if face count matters more than quad flow. If `polyRetopo`
  fails on a given shell it's logged and that mesh keeps its original
  traced topology rather than aborting the build.
- **One mesh per layer (or per shell), not one shared mesh.** Each
  traced shell becomes its own transform + mesh, parented under a common
  `psd2maya_root` group. This is what makes it a *parallax rig* rather
  than a flat cutout: each piece can be independently offset, hidden, or
  animated later.
- **Depth from PSD stack order.** The PSD format stores layer records
  bottom-to-top (see `psd_reader.py` docstring); layer 0 in that order is
  placed at `Z=0` and each subsequent layer steps forward by
  `--depth-step` Maya units. No manual depth entry needed. Pass a negative
  `--depth-step` if your camera looks the other way down Z.
- **Shared atlas, not per-layer textures.** All layers pack into one (or
  a few, if they don't fit) atlas PNG via a shelf packer, so the whole
  background rig uses a single shading network / one or two draw calls
  instead of one material per layer. Each layer gets a bleed-extended
  padding border so bilinear filtering doesn't smear a neighboring
  layer's pixels across the seam.
- **Hidden layers, empty layers, and groups are skipped**, not
  recreated as empty meshes. Text/shape/smart-object layers are
  rasterized via `layer.composite()` like everything else -- there's no
  special-casing by layer kind.
- **Absolute texture paths.** `fileTextureName` is written as an absolute
  path rather than relative, because Maya resolves relative texture paths
  against the current project's `sourceimages` directory, not against the
  `.ma` file's own folder -- a relative path would silently break unless
  the user's Maya project happens to match the output directory.

### Known limitation

`export_ma.py` was written to match Maya's documented ASCII mesh/shading
format -- including the edge-dedup/sign convention `.fc` needs once faces
share edges (see that module's docstring) -- but there is no Maya
installation in this environment to actually round-trip a file through
it. Quad-only-ness and face winding *have* been verified independently in
Python (every face has 4 distinct vertex indices, every face's normal
points the same way), but not the ASCII syntax itself against a real
Maya parser. **Open the first generated `.ma` in Maya and check the
Outliner, UVs, and shading before relying on it in production.** If
anything doesn't load cleanly, `maya_backend.py`'s `build_in_maya()`
builds the identical scene via the Maya Python API instead and sidesteps
hand-written ASCII entirely -- see Usage below.

## Install

### Standalone (CLI only)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### For use inside Maya

The `.venv` above is a separate interpreter that Maya never sees -- the
dependencies have to go into **Maya's own Python** (`mayapy`) instead:

```bash
"/Applications/Autodesk/maya2026/Maya.app/Contents/bin/mayapy" -m pip install --user psd-tools Pillow "attrs==25.3.0"
```

(Adjust the path for your Maya version. `numpy` and `opencv-python-headless`
already ship with Maya 2026.)

`attrs` **must** be pinned to 25.3.0. Maya's bundled `mayaflow` addin ships
its own `attr` package at that version and wins the import, so a newer
`attrs` from pip fails at import time with
`cannot import name 'ClassProps' from 'attr._make'`.

Then make the package importable from Maya. In the Script Editor:

```python
import sys
sys.path.append("/path/to/PSD_Maya_Toolkit")
```

Add that line to `userSetup.py` (in
`~/Library/Preferences/Autodesk/maya/<version>/scripts/` on macOS) to make it
stick across restarts, since `sys.path` resets every session.

## Usage

### Standalone (no Maya installed)

```bash
python3 -m psd2maya path/to/background.psd -o out/
```

Produces `out/background.ma` and `out/atlas.png` (or `atlas_0.png`,
`atlas_1.png`, ... if the layers don't fit one page). Open the `.ma` in
Maya.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--pixels-per-unit` | 100 | PSD pixels per 1 Maya world unit |
| `--depth-step` | 5.0 | Maya units between layers along Z (negative reverses front/back) |
| `--detail-level` | 2.0 | Douglas-Peucker epsilon in px for silhouette tracing (lower = tighter, more verts) |
| `--alpha-threshold` | 10 | Raw 0-255 alpha cutoff for what counts as opaque when tracing |
| `--min-contour-area` | 50.0 | Ignore traced regions smaller than this many px^2 (filters AA dust) |
| `--max-page-size` | 4096 | Max atlas page width/height in px before spilling to a new page |
| `--padding` | 4 | Bleed-extended border in px around each packed layer |
| `--include-hidden` | off | Also rasterize layers hidden in Photoshop |

### Inside Maya / mayapy

```bash
mayapy -m psd2maya.maya_backend path/to/background.psd out/
```

Builds the rig live via the Maya Python API (`maya.api.OpenMaya`'s
`MFnMesh.create()`, one call per traced shell -- no hand-written ASCII
involved) and saves `out/background.ma`. To build into an already-open
Maya session instead of a fresh headless one, call the pieces directly
from the Script Editor:

```python
from psd2maya.pipeline import run_pipeline
from psd2maya.maya_backend import build_in_maya

scene, atlas_paths = run_pipeline("path/to/background.psd", "out/")
build_in_maya(scene, atlas_paths)
```

### PySide6 UI (inside a running Maya session)

Requires Maya 2025+ (ships PySide6/shiboken6). From the Script Editor:

```python
from psd2maya.ui import show
show()
```

Drag a `.psd`/`.psb` file onto the path field (or use Browse...) to
populate the layer tree -- groups nest, hidden layers show greyed out.
Adjust the build options if needed, then click **Build Mesh** to run the
full parse -> pack -> trace -> quadrangulate -> build pipeline and create
the rig, textured meshes, and atlas PNG(s) directly in the open scene.
Output (the atlas PNG(s)) is written to `<psd_dir>/<psd_name>_maya/` next
to the source file.

Ticking **Use Maya polyRetopo** remeshes each layer with Maya's
retopologizer instead of the built-in ear-clip topology and then restores
exact UVs (see the design note above for the tradeoffs -- it's slower and
usually *raises* face count). **Target faces per mesh** enables only when
that box is ticked.

To re-derive UVs yourself after remeshing a layer by hand:

```python
from psd2maya.maya_backend import reproject_uvs
reproject_uvs("Sky")   # reads the locked psdUv* attrs on the transform
```

## Testing

There's no real background-art PSD available in this environment, so
`tests/make_test_psd.py` generates a small synthetic one (sky/hills/rock
layers plus a hidden layer that should be skipped) to exercise the
standalone pipeline end-to-end:

```bash
python3 tests/make_test_psd.py tests/fixtures/test_scene.psd
python3 -m psd2maya tests/fixtures/test_scene.psd -o tests/out
```

Quad-only-ness, face winding, and how tightly the traced boundary hugs
each layer's silhouette were all checked directly against this fixture's
output (see the design-decisions notes above) since there's no Maya
install here to verify visually.
