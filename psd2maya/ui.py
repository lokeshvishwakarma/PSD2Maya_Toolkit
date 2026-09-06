"""PySide6 UI for building a psd2maya parallax rig inside a running Maya session.

Run from Maya's Script Editor (Python tab):

    from psd2maya.ui import show
    show()

Only imports `maya.cmds`/`maya.OpenMayaUI`/`shiboken6` at call time isn't
possible here the way `maya_backend.py` does it for its build functions,
because the widgets themselves need `maya.OpenMayaUI` to parent to the main
window -- so, unlike the rest of the package, this module requires Maya
(PySide6 + shiboken6, both shipped with Maya 2025+) to even import.

Selecting a layer or group in the tree renders a live thumbnail beside it
(over a transparency checkerboard) by calling `.composite()` on the
`LayerNode.source` handle `layer_tree.read_layer_tree` stashed on it -- a
single leaf goes straight through `.composite()` (already fast, bbox-
limited), while a group instead goes through `layer_preview.
render_group_thumbnail`'s downscale-first approximation, since a group's
own `.composite()` blends every descendant at full PSD resolution just to
be immediately shrunk to ~180px (see that module's docstring: 6-7s vs
under half a second on a real 41-layer production group). Single-item
results are cached per node for the lifetime of the loaded file, so
revisiting an already-viewed item is instant.

The tree supports multi-selection (ctrl/shift-click, same convention as
any file browser). Selecting more than one item previews all of them
composited together at their true relative position via `layer_preview.
render_nodes_thumbnail` -- a deliberate choice over a blank "N selected"
placeholder, since seeing what you're about to bulk-tag is more useful
than seeing nothing, and it reuses the exact same fast per-leaf machinery
as the single-group case. Right-clicking the tree opens a context menu to
bulk-assign the LOD column (see below) to every selected row at once.

The LOD column embeds a real `QComboBox` per row (`High`/`Mid`/`Low`) via
`setItemWidget` rather than Qt's built-in click-to-edit item mechanism, so
the current value is always visible without an extra click, and is purely
a UI-side tag on `LayerNode.lod` right now -- nothing in the build
pipeline reads it (yet).
"""

from __future__ import annotations

import os
import traceback
from typing import Dict

import maya.cmds as cmds
import maya.OpenMayaUI as omui
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import wrapInstance

from .layer_preview import render_group_thumbnail, render_nodes_thumbnail
from .layer_tree import LayerNode, read_layer_tree
from .maya_backend import build_in_maya
from .pipeline import run_pipeline
from .relayout import merge_textures, rebuild_textures_from_uv_layout

WINDOW_OBJECT_NAME = "psd2mayaBuildWindow"

_KIND_LABELS = {
    "group": "Group",
    "pixel": "Pixel",
    "type": "Text",
    "shape": "Shape",
    "smartobject": "Smart Object",
}

_PREVIEW_SIZE = 180
_LAYER_NODE_ROLE = QtCore.Qt.UserRole

_LOD_COLUMN = 3
_LOD_LEVELS = ("High", "Mid", "Low")
# Subtle background tints so a column of many rows scans at a glance without
# fighting Maya's own dark theme -- not saturated enough to read as an alert.
_LOD_COLORS = {"High": "#3d5a3d", "Mid": "#454545", "Low": "#5a3d3d"}


def _maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    if ptr is None:
        return None
    return wrapInstance(int(ptr), QtWidgets.QWidget)


def _checkerboard_pixmap(size: int, cell: int = 9) -> QtGui.QPixmap:
    """A light/dark checkerboard, the usual "this is transparent" convention.

    Built once per preview update (cheap at this size) rather than cached,
    since it's only ever the backdrop a layer thumbnail gets painted over.
    """
    pixmap = QtGui.QPixmap(size, size)
    light, dark = QtGui.QColor(90, 90, 90), QtGui.QColor(70, 70, 70)
    painter = QtGui.QPainter(pixmap)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            even = ((x // cell) + (y // cell)) % 2 == 0
            painter.fillRect(x, y, cell, cell, light if even else dark)
    painter.end()
    return pixmap


def _pil_to_qpixmap(image) -> QtGui.QPixmap:
    """Convert a PIL RGBA image to a QPixmap.

    `.copy()` on the QImage forces Qt to own its own copy of the pixel
    buffer immediately -- without it, the QImage would keep referencing the
    `bytes` object underneath, which Python is free to garbage-collect as
    soon as this function returns, corrupting or crashing on whatever
    tries to paint the pixmap afterward.
    """
    rgba = image.convert("RGBA")
    raw = rgba.tobytes("raw", "RGBA")
    qimage = QtGui.QImage(raw, rgba.width, rgba.height, QtGui.QImage.Format_RGBA8888).copy()
    return QtGui.QPixmap.fromImage(qimage)


class PsdDropLineEdit(QtWidgets.QLineEdit):
    """A QLineEdit that also accepts a dragged-and-dropped .psd/.psb file."""

    fileDropped = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drag a .psd file here, or Browse...")

    @staticmethod
    def _psd_path_from_event(event) -> str:
        mime = event.mimeData()
        if not mime.hasUrls():
            return ""
        for url in mime.urls():
            path = url.toLocalFile()
            if path.lower().endswith((".psd", ".psb")):
                return path
        return ""

    def dragEnterEvent(self, event):
        if self._psd_path_from_event(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._psd_path_from_event(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._psd_path_from_event(event)
        if not path:
            event.ignore()
            return
        self.setText(path)
        event.acceptProposedAction()
        self.fileDropped.emit(path)


class Psd2MayaWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent if parent is not None else _maya_main_window())
        self.setObjectName(WINDOW_OBJECT_NAME)
        self.setWindowTitle("PSD to Maya - Parallax Rig Builder")
        self.setMinimumSize(520, 560)
        self.setWindowFlags(
            (self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint) | QtCore.Qt.WindowStaysOnTopHint
        )

        self._canvas_size = None  # (width, height) of the currently loaded PSD
        self._last_scene = None  # SceneData from the most recent successful Build Mesh
        # id(LayerNode) -> QPixmap, cleared per file load (see _load_psd). Keyed by
        # id() rather than the node itself: LayerNode is a plain @dataclass, so its
        # auto-generated __eq__ makes it unhashable (__hash__ is None) -- and each
        # node is only ever built once per tree anyway, so identity is exactly the
        # right notion of "same node" here.
        self._preview_cache = {}

        self._build_ui()
        self._connect_signals()

    # -- construction ---------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = PsdDropLineEdit(self)
        self.browse_btn = QtWidgets.QPushButton("Browse...", self)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_btn)
        layout.addLayout(path_row)

        tree_row = QtWidgets.QHBoxLayout()

        self.layer_tree = QtWidgets.QTreeWidget(self)
        self.layer_tree.setHeaderLabels(["Layer", "Kind", "Opacity", "LOD"])
        self.layer_tree.setColumnWidth(0, 220)
        self.layer_tree.setColumnWidth(_LOD_COLUMN, 90)
        self.layer_tree.setAlternatingRowColors(True)
        self.layer_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.layer_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        tree_row.addWidget(self.layer_tree, 1)

        preview_col = QtWidgets.QVBoxLayout()
        self.preview_label = QtWidgets.QLabel(self)
        self.preview_label.setFixedSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setFrameShape(QtWidgets.QFrame.Box)
        self.preview_label.setStyleSheet("color: #888;")
        self.preview_label.setWordWrap(True)
        self.preview_label.setText("Select a layer\nto preview")
        preview_col.addWidget(self.preview_label)

        self.preview_info_label = QtWidgets.QLabel("", self)
        self.preview_info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_info_label.setFixedWidth(_PREVIEW_SIZE)
        self.preview_info_label.setWordWrap(True)
        preview_col.addWidget(self.preview_info_label)
        preview_col.addStretch(1)

        tree_row.addLayout(preview_col)
        layout.addLayout(tree_row, 1)

        self.canvas_label = QtWidgets.QLabel("No file loaded.", self)
        layout.addWidget(self.canvas_label)

        options_group = QtWidgets.QGroupBox("Options", self)
        opt_form = QtWidgets.QFormLayout(options_group)

        self.include_hidden_chk = QtWidgets.QCheckBox(
            "Include hidden layers (shown in grey above)", options_group
        )

        self.pixels_per_unit_spin = QtWidgets.QDoubleSpinBox(options_group)
        self.pixels_per_unit_spin.setRange(1.0, 100000.0)
        self.pixels_per_unit_spin.setValue(100.0)

        self.depth_step_spin = QtWidgets.QDoubleSpinBox(options_group)
        self.depth_step_spin.setRange(-10000.0, 10000.0)
        self.depth_step_spin.setValue(5.0)

        self.detail_level_spin = QtWidgets.QDoubleSpinBox(options_group)
        self.detail_level_spin.setRange(0.1, 50.0)
        self.detail_level_spin.setSingleStep(0.5)
        self.detail_level_spin.setValue(2.0)

        self.retopo_chk = QtWidgets.QCheckBox("Use Maya polyRetopo", options_group)
        self.retopo_chk.setToolTip(
            "Remesh each layer with Maya's polyRetopo for more uniform quad flow "
            "instead of the built-in ear-clip topology.\n"
            "UVs are recomputed exactly afterward, so the texture is unaffected."
        )

        # One target-face-count spinner per LOD tier rather than a single
        # value applied to every mesh alike: a layer explicitly tagged "Low"
        # in the tree's LOD column (see _attach_lod_combo) can collapse to a
        # handful of faces while one tagged "High" keeps considerably more,
        # independent of how big either happens to be on screen -- looked up
        # per mesh by maya_backend.build_in_maya via each LayerMesh.lod.
        self.lod_polycount_spins: Dict[str, QtWidgets.QSpinBox] = {}
        lod_polycount_row = QtWidgets.QHBoxLayout()
        for level, default_value in zip(_LOD_LEVELS, (25, 15, 5)):
            spin = QtWidgets.QSpinBox(options_group)
            spin.setRange(3, 100000)  # polyRetopo needs at least a triangle's worth of faces
            spin.setSingleStep(5)
            spin.setValue(default_value)
            spin.setEnabled(False)
            spin.setToolTip(f"Approximate quad count polyRetopo aims for on a '{level}' LOD mesh.")
            self.lod_polycount_spins[level] = spin
            lod_polycount_row.addWidget(QtWidgets.QLabel(f"{level}:", options_group))
            lod_polycount_row.addWidget(spin)

        opt_form.addRow(self.include_hidden_chk)
        opt_form.addRow("Pixels per Maya unit:", self.pixels_per_unit_spin)
        opt_form.addRow("Depth step (Z per layer):", self.depth_step_spin)
        opt_form.addRow("Trace detail (lower = tighter):", self.detail_level_spin)
        opt_form.addRow(self.retopo_chk)
        opt_form.addRow("Target faces per LOD:", lod_polycount_row)
        layout.addWidget(options_group)

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.build_btn = QtWidgets.QPushButton("Build Mesh", self)
        self.build_btn.setEnabled(False)
        self.build_btn.setMinimumHeight(32)
        layout.addWidget(self.build_btn)

        self.rebuild_texture_btn = QtWidgets.QPushButton("Rebuild Texture from UV Layout", self)
        self.rebuild_texture_btn.setEnabled(False)
        self.rebuild_texture_btn.setToolTip(
            "After manually running Maya's Layout UV on some of this rig's meshes, "
            "click this to rebake each affected UV set's atlas texture to match the "
            "new UV positions and repoint its file texture node at the result."
        )
        layout.addWidget(self.rebuild_texture_btn)

        self.merge_textures_btn = QtWidgets.QPushButton("Merge Textures", self)
        self.merge_textures_btn.setEnabled(False)
        self.merge_textures_btn.setToolTip(
            "Superimpose every atlas page still in use into one merged PNG "
            "(like merging layers in Photoshop) and repoint the existing "
            "meshes at it. Creates no new geometry."
        )
        layout.addWidget(self.merge_textures_btn)

    def _connect_signals(self):
        self.browse_btn.clicked.connect(self._on_browse)
        for spin in self.lod_polycount_spins.values():
            self.retopo_chk.toggled.connect(spin.setEnabled)
        self.path_edit.fileDropped.connect(self._load_psd)
        self.path_edit.returnPressed.connect(lambda: self._load_psd(self.path_edit.text().strip()))
        self.build_btn.clicked.connect(self._on_build_mesh)
        self.rebuild_texture_btn.clicked.connect(self._on_rebuild_texture)
        self.merge_textures_btn.clicked.connect(self._on_merge_textures)
        self.layer_tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.layer_tree.customContextMenuRequested.connect(self._on_tree_context_menu)

    # -- PSD loading / tree population -----------------------------------

    def _on_browse(self):
        start_dir = os.path.dirname(self.path_edit.text().strip()) or ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Photoshop File", start_dir, "Photoshop Files (*.psd *.psb)"
        )
        if path:
            self.path_edit.setText(path)
            self._load_psd(path)

    def _load_psd(self, path: str):
        if not path:
            return

        # Invalidate before validating: a failed load must not leave the
        # previous file's layer tree on screen with Build still live, or the
        # user would be looking at one PSD's layers while the path field
        # points at another.
        self.layer_tree.clear()
        self.build_btn.setEnabled(False)
        self._canvas_size = None
        self.canvas_label.setText("No file loaded.")
        self._show_preview_placeholder("Select a layer\nto preview")
        self._preview_cache = {}

        if not os.path.isfile(path):
            self._set_status(f"File not found: {path}", error=True)
            return

        try:
            nodes, canvas_w, canvas_h = read_layer_tree(path)
        except Exception as exc:
            self._set_status(f"Failed to read {os.path.basename(path)}: {exc}", error=True)
            return

        self._canvas_size = (canvas_w, canvas_h)
        self.canvas_label.setText(f"Canvas: {canvas_w} x {canvas_h} px")

        for node in nodes:
            self.layer_tree.addTopLevelItem(self._make_tree_item(node))
        self.layer_tree.expandAll()

        # A second pass, after every item is actually in the tree: setItemWidget
        # needs an item that already belongs to the widget to reliably attach,
        # which isn't yet true of a child item at the point _make_tree_item
        # builds it (it's only attached to its *parent* item so far, and that
        # parent isn't in the tree yet either until this loop above runs).
        iterator = QtWidgets.QTreeWidgetItemIterator(self.layer_tree)
        while iterator.value():
            item = iterator.value()
            node = item.data(0, _LAYER_NODE_ROLE)
            if node is not None:
                self._attach_lod_combo(item, node)
            iterator += 1

        self.build_btn.setEnabled(True)
        self._set_status(f"Loaded {os.path.basename(path)}.")

    def _make_tree_item(self, node: LayerNode) -> QtWidgets.QTreeWidgetItem:
        kind_label = "Group" if node.is_group else _KIND_LABELS.get(node.kind, node.kind)
        opacity_label = "" if node.is_group else f"{round(node.opacity * 100)}%"
        item = QtWidgets.QTreeWidgetItem([node.name, kind_label, opacity_label])
        item.setData(0, _LAYER_NODE_ROLE, node)

        if not node.visible:
            grey = QtGui.QBrush(QtGui.QColor(120, 120, 120))
            for col in range(3):
                item.setForeground(col, grey)
            item.setToolTip(0, "Hidden in Photoshop")

        for child in node.children:
            item.addChild(self._make_tree_item(child))
        return item

    # -- LOD column ---------------------------------------------------------

    def _attach_lod_combo(self, item: QtWidgets.QTreeWidgetItem, node: LayerNode):
        combo = QtWidgets.QComboBox(self.layer_tree)
        combo.addItems(_LOD_LEVELS)
        combo.setCurrentText(node.lod)
        self._style_lod_combo(combo, node.lod)
        # node=node, combo=combo: default-arg capture, since a plain closure
        # over the loop variables in _load_psd's iterator would otherwise have
        # every combo's callback see whatever `node`/`item` last ended up as.
        combo.currentTextChanged.connect(lambda text, node=node, combo=combo: self._on_lod_changed(node, combo, text))
        self.layer_tree.setItemWidget(item, _LOD_COLUMN, combo)

    def _on_lod_changed(self, node: LayerNode, combo: QtWidgets.QComboBox, level: str):
        node.lod = level
        self._style_lod_combo(combo, level)

    def _style_lod_combo(self, combo: QtWidgets.QComboBox, level: str):
        color = _LOD_COLORS.get(level)
        combo.setStyleSheet(f"QComboBox {{ background-color: {color}; }}" if color else "")

    def _on_tree_context_menu(self, pos: QtCore.QPoint):
        item = self.layer_tree.itemAt(pos)
        if item is None:
            return
        # Right-clicking an item outside the current selection replaces the
        # selection with just that one, matching most file browsers; right-
        # clicking a row that's already part of a multi-selection leaves the
        # whole selection intact so the menu below applies to all of it.
        if item not in self.layer_tree.selectedItems():
            self.layer_tree.setCurrentItem(item)

        selected = self.layer_tree.selectedItems()
        menu = QtWidgets.QMenu(self)
        lod_menu = menu.addMenu(f"Set LOD ({len(selected)} selected)")
        for level in _LOD_LEVELS:
            action = lod_menu.addAction(level)
            action.triggered.connect(lambda checked=False, level=level: self._assign_lod_to_selection(level))
        menu.exec(self.layer_tree.viewport().mapToGlobal(pos))

    def _assign_lod_to_selection(self, level: str):
        for item in self.layer_tree.selectedItems():
            combo = self.layer_tree.itemWidget(item, _LOD_COLUMN)
            if combo is not None:
                combo.setCurrentText(level)  # _on_lod_changed does the rest (node.lod + restyle)

    def _collect_lod_by_name(self) -> Dict[str, str]:
        """{layer name: LOD} for every non-group row, read fresh from the live tree.

        Passed to `pipeline.run_pipeline` as `lod_by_name` so `psd_reader.
        extract_layers` can tag each `SourceLayer` with whatever LOD the user
        last set in the tree -- reading the tree directly (rather than some
        snapshot taken at load time) means edits made right up until Build
        Mesh is clicked are picked up. Groups are skipped: they never become
        their own mesh, so a group's own LOD combo has nothing to apply to.
        """
        lod_by_name: Dict[str, str] = {}
        iterator = QtWidgets.QTreeWidgetItemIterator(self.layer_tree)
        while iterator.value():
            node = iterator.value().data(0, _LAYER_NODE_ROLE)
            if node is not None and not node.is_group:
                lod_by_name[node.name] = node.lod
            iterator += 1
        return lod_by_name

    # -- layer preview ------------------------------------------------------

    def _show_preview_placeholder(self, text: str):
        self.preview_label.clear()  # drop any pixmap so setText below actually shows
        self.preview_label.setText(text)
        self.preview_info_label.setText("")

    def _on_selection_changed(self):
        items = self.layer_tree.selectedItems()
        if not items:
            self._show_preview_placeholder("Select a layer\nto preview")
            return

        nodes = [item.data(0, _LAYER_NODE_ROLE) for item in items]
        nodes = [node for node in nodes if node is not None and node.source is not None]
        if not nodes:
            self._show_preview_placeholder("No preview\navailable")
            return

        # Only a single selected node has a stable, reusable cache key -- any
        # of the astronomically many possible multi-selections could be picked
        # next, so caching those isn't worth the memory; a fresh composite
        # for that case is still fast (same thread-pooled fast path).
        cache_key = id(nodes[0]) if len(nodes) == 1 else None
        self._render_preview(nodes, cache_key)

    def _render_preview(self, nodes: list, cache_key):
        if cache_key is not None:
            cached = self._preview_cache.get(cache_key)
            if cached is not None:
                self.preview_label.setPixmap(cached)
                self._set_preview_info(nodes)
                return

        # A group's own .composite() blends every descendant at full PSD
        # resolution; layer_preview.render_group_thumbnail/render_nodes_thumbnail
        # decode each leaf and shrink it immediately instead, which is what
        # actually matters once the result only has to look right at ~180px
        # (measured 13-16x faster on a real 41-layer production group -- see
        # that module's docstring). A single leaf's own .composite() is
        # already bbox-limited and fast, so it's untouched. Either way this
        # still isn't instant on a big enough PSD, so the wait cursor matches
        # this window's existing convention for anything that might take a
        # beat (see _on_build_mesh/_on_rebuild_texture/_on_merge_textures).
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if len(nodes) > 1:
                image = render_nodes_thumbnail(nodes, max_size=256)
            elif nodes[0].is_group:
                image = render_group_thumbnail(nodes[0].source, max_size=256)
            else:
                image = nodes[0].source.composite()
        except Exception:
            traceback.print_exc()
            self._show_preview_placeholder("Preview failed\n(see Script Editor)")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if image is None or image.width == 0 or image.height == 0:
            self._show_preview_placeholder("No preview\navailable")
            return

        pixmap = _pil_to_qpixmap(image)
        scaled = pixmap.scaled(
            _PREVIEW_SIZE, _PREVIEW_SIZE, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        canvas = _checkerboard_pixmap(_PREVIEW_SIZE)
        painter = QtGui.QPainter(canvas)
        painter.drawPixmap((_PREVIEW_SIZE - scaled.width()) // 2, (_PREVIEW_SIZE - scaled.height()) // 2, scaled)
        painter.end()

        if cache_key is not None:
            self._preview_cache[cache_key] = canvas
        self.preview_label.setPixmap(canvas)
        self._set_preview_info(nodes)

    def _set_preview_info(self, nodes: list):
        if len(nodes) == 1:
            self.preview_info_label.setText(f"{nodes[0].width} x {nodes[0].height} px")
        else:
            self.preview_info_label.setText(f"{len(nodes)} layers selected")

    # -- build ------------------------------------------------------------

    def _on_build_mesh(self):
        psd_path = self.path_edit.text().strip()
        if not psd_path or not os.path.isfile(psd_path):
            self._set_status("Pick a valid .psd/.psb file first.", error=True)
            return

        out_dir = os.path.join(
            os.path.dirname(psd_path), f"{os.path.splitext(os.path.basename(psd_path))[0]}_maya"
        )

        self.build_btn.setEnabled(False)
        self._set_status("Building...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        cmds.waitCursor(state=True)
        try:
            scene, atlas_paths = run_pipeline(
                psd_path,
                out_dir,
                pixels_per_unit=self.pixels_per_unit_spin.value(),
                depth_step=self.depth_step_spin.value(),
                detail_level=self.detail_level_spin.value(),
                include_hidden=self.include_hidden_chk.isChecked(),
                lod_by_name=self._collect_lod_by_name(),
            )
            retopo_failures = []
            root = build_in_maya(
                scene,
                atlas_paths,
                retopo=self.retopo_chk.isChecked(),
                target_face_count_by_lod={level: spin.value() for level, spin in self.lod_polycount_spins.items()},
                retopo_failures=retopo_failures,
            )
            cmds.select(root, replace=True)
        except Exception as exc:
            traceback.print_exc()
            self._set_status(f"Build failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Build Mesh Failed", str(exc))
            return
        finally:
            cmds.waitCursor(state=False)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.build_btn.setEnabled(True)

        self._last_scene = scene
        self.rebuild_texture_btn.setEnabled(True)
        self.merge_textures_btn.setEnabled(True)

        total_faces = sum(cmds.polyEvaluate(m.maya_name, face=True) for m in scene.meshes)
        topo = "polyRetopo" if self.retopo_chk.isChecked() else "traced quads"
        note = ""
        if retopo_failures:
            # Never let this pass silently: the fallback topology is also
            # all-quads, so nothing else on screen would reveal it.
            note = (
                f" NOTE: polyRetopo could not process {len(retopo_failures)} mesh(es) "
                f"({', '.join(retopo_failures[:3])}{'...' if len(retopo_failures) > 3 else ''}); "
                "those kept their original traced topology."
            )
        self._set_status(
            f"Built {len(scene.meshes)} mesh(es), {total_faces} faces ({topo}) under '{root}'. "
            f"Atlas: {', '.join(atlas_paths.values())}.{note}"
        )

    def _on_rebuild_texture(self):
        if self._last_scene is None:
            self._set_status("Build Mesh first -- nothing to rebake yet.", error=True)
            return

        self.rebuild_texture_btn.setEnabled(False)
        self._set_status("Rebaking texture(s) from current UV layout...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        cmds.waitCursor(state=True)
        try:
            new_paths = rebuild_textures_from_uv_layout(self._last_scene)
        except Exception as exc:
            traceback.print_exc()
            self._set_status(f"Texture rebake failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Rebuild Texture Failed", str(exc))
            return
        finally:
            cmds.waitCursor(state=False)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.rebuild_texture_btn.setEnabled(True)

        if not new_paths:
            self._set_status("No UV sets had any rebakeable meshes -- nothing was changed.", error=True)
            return
        self._set_status(f"Rebaked {len(new_paths)} atlas page(s): {', '.join(new_paths.values())}")

    def _on_merge_textures(self):
        if self._last_scene is None:
            self._set_status("Build Mesh first -- nothing to merge yet.", error=True)
            return

        self.merge_textures_btn.setEnabled(False)
        self._set_status("Superimposing atlas textures into one merged PNG...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        cmds.waitCursor(state=True)
        try:
            merged_path = merge_textures(self._last_scene)
        except Exception as exc:
            traceback.print_exc()
            self._set_status(f"Texture merge failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Merge Textures Failed", str(exc))
            return
        finally:
            cmds.waitCursor(state=False)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.merge_textures_btn.setEnabled(True)

        if merged_path is None:
            self._set_status("Fewer than two atlas pages have both an existing mesh and a texture -- nothing to merge.")
            return
        self._set_status(f"Merged atlas pages into one material: {merged_path}")

    def _set_status(self, text: str, error: bool = False):
        color = "#ff6b6b" if error else "#9fd39f"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)


def show():
    """Close any existing instance and open a fresh PSD-to-Maya build window."""
    main_window = _maya_main_window()
    if main_window is not None:
        existing = main_window.findChild(QtWidgets.QWidget, WINDOW_OBJECT_NAME)
        if existing is not None:
            existing.close()
            existing.deleteLater()

    window = Psd2MayaWindow()
    window.show()
    return window
