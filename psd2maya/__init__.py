"""psd2maya: recreate layered PSD background art as a Maya parallax card rig.

Pipeline (see each module's docstring for details):
    psd_reader   -> SourceLayer list           (Pillow/psd-tools only)
    atlas_packer -> AtlasResult                (Pillow only)
    mesh_builder -> SceneData                  (pure Python)
    export_ma    -> .ma file                   (pure Python, no Maya needed)
    maya_backend -> live Maya scene            (needs maya.cmds / mayapy)
    ui           -> PySide6 build window       (needs Maya 2025+ / PySide6)
"""

__version__ = "0.1.0"
