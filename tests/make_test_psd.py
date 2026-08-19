"""Generate a small synthetic multi-layer PSD to exercise the pipeline end-to-end.

Not a unit test in itself -- just a fixture builder, since a real
background-art PSD isn't available in this environment. Produces three
layers of different sizes/positions/colors so packing, positioning, and
depth ordering are all visibly distinguishable in the result.
"""

from __future__ import annotations

import os

from PIL import Image
from psd_tools.api.psd_image import PSDImage


def make_test_psd(path: str, canvas=(800, 600)) -> None:
    psd = PSDImage.new(mode="RGBA", size=canvas, color=(0, 0, 0, 0))

    sky = Image.new("RGBA", canvas, (60, 110, 200, 255))
    psd.create_pixel_layer(sky, name="sky_bg", top=0, left=0)

    hills = Image.new("RGBA", (700, 220), (0, 0, 0, 0))
    for x in range(700):
        h = int(80 + 60 * abs((x % 350) - 175) / 175)
        for y in range(220 - h, 220):
            hills.putpixel((x, y), (40, 140, 70, 255))
    psd.create_pixel_layer(hills, name="mid_hills", top=300, left=50)

    rock = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
    for x in range(180):
        for y in range(140):
            if (x - 90) ** 2 / 90.0**2 + (y - 130) ** 2 / 60.0**2 <= 1.0 and y < 130:
                rock.putpixel((x, y), (90, 70, 60, 255))
    psd.create_pixel_layer(rock, name="fg_rock", top=420, left=500)

    hidden = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    unused = psd.create_pixel_layer(hidden, name="unused_note", top=10, left=10)
    unused.visible = False

    os.makedirs(os.path.dirname(path), exist_ok=True)
    psd.save(path)


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/test_scene.psd"
    make_test_psd(out)
    print(f"Wrote {out}")
