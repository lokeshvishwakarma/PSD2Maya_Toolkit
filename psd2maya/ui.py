"""PySide6 UI for building a psd2maya parallax rig inside a running Maya session.

Run from Maya's Script Editor (Python tab):

    from psd2maya.ui import show
    show()

Only imports `maya.cmds`/`maya.OpenMayaUI`/`shiboken6` at call time isn't
possible here the way `maya_backend.py` does it for its build functions,
because the widgets themselves need `maya.OpenMayaUI` to parent to the main
window -- so, unlike the rest of the package, this module requires Maya
(PySide6 + shiboken6, both shipped with Maya 2025+) to even import.
"""

from __future__ import annotations

import os
import traceback

import maya.cmds as cmds
import maya.OpenMayaUI as omui
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import wrapInstance

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


def _maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    if ptr is None:
        return None
    return wrapInstance(int(ptr), QtWidgets.QWidget)


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

        self.layer_tree = QtWidgets.QTreeWidget(self)
        self.layer_tree.setHeaderLabels(["Layer", "Kind", "Opacity"])
        self.layer_tree.setColumnWidth(0, 260)
        self.layer_tree.setAlternatingRowColors(True)
        layout.addWidget(self.layer_tree, 1)

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

        self.target_face_count_spin = QtWidgets.QSpinBox(options_group)
        self.target_face_count_spin.setRange(4, 100000)
        self.target_face_count_spin.setSingleStep(50)
        self.target_face_count_spin.setValue(200)
        self.target_face_count_spin.setEnabled(False)
        self.target_face_count_spin.setToolTip(
            "Approximate quad count polyRetopo aims for, per mesh. Applied to every "
            "layer alike, so large background layers get the same budget as small props."
        )

        opt_form.addRow(self.include_hidden_chk)
        opt_form.addRow("Pixels per Maya unit:", self.pixels_per_unit_spin)
        opt_form.addRow("Depth step (Z per layer):", self.depth_step_spin)
        opt_form.addRow("Trace detail (lower = tighter):", self.detail_level_spin)
        opt_form.addRow(self.retopo_chk)
        opt_form.addRow("Target faces per mesh:", self.target_face_count_spin)
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
            "Superimpose every existing mesh's current texture appearance (in PSD "
            "paint order) into one flattened image of the whole canvas, and apply "
            "it to a new flat backdrop card -- reflects any Layout UV / Rebuild "
            "Texture changes you've already made."
        )
        layout.addWidget(self.merge_textures_btn)

    def _connect_signals(self):
        self.browse_btn.clicked.connect(self._on_browse)
        self.retopo_chk.toggled.connect(self.target_face_count_spin.setEnabled)
        self.path_edit.fileDropped.connect(self._load_psd)
        self.path_edit.returnPressed.connect(lambda: self._load_psd(self.path_edit.text().strip()))
        self.build_btn.clicked.connect(self._on_build_mesh)
        self.rebuild_texture_btn.clicked.connect(self._on_rebuild_texture)
        self.merge_textures_btn.clicked.connect(self._on_merge_textures)

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

        self.build_btn.setEnabled(True)
        self._set_status(f"Loaded {os.path.basename(path)}.")

    def _make_tree_item(self, node: LayerNode) -> QtWidgets.QTreeWidgetItem:
        kind_label = "Group" if node.is_group else _KIND_LABELS.get(node.kind, node.kind)
        opacity_label = "" if node.is_group else f"{round(node.opacity * 100)}%"
        item = QtWidgets.QTreeWidgetItem([node.name, kind_label, opacity_label])

        if not node.visible:
            grey = QtGui.QBrush(QtGui.QColor(120, 120, 120))
            for col in range(3):
                item.setForeground(col, grey)
            item.setToolTip(0, "Hidden in Photoshop")

        for child in node.children:
            item.addChild(self._make_tree_item(child))
        return item

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
            )
            retopo_failures = []
            root = build_in_maya(
                scene,
                atlas_paths,
                retopo=self.retopo_chk.isChecked(),
                target_face_count=self.target_face_count_spin.value(),
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
        self._set_status("Flattening current textures into one backdrop...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        cmds.waitCursor(state=True)
        try:
            flattened_path = merge_textures(self._last_scene)
        except Exception as exc:
            traceback.print_exc()
            self._set_status(f"Texture flatten failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Merge Textures Failed", str(exc))
            return
        finally:
            cmds.waitCursor(state=False)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.merge_textures_btn.setEnabled(True)

        if flattened_path is None:
            self._set_status("No existing mesh had a resolvable current texture -- nothing to flatten.")
            return
        self._set_status(f"Flattened backdrop applied to 'psd2mayaFlattenedBackdrop': {flattened_path}")

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
