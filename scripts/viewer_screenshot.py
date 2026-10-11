"""Take a screenshot of the trajectory viewer for the documentation.

Usage::

    uv run --no-sync --with cairosvg python scripts/viewer_screenshot.py RESULTS [OUT_DIR]

The script starts the viewer without a terminal at 160 by 46 cells. It presses the
down arrow once to select the second episode and ``r`` once to expand the tool
results. It writes ``viewer.svg`` and a 1600-pixel-wide ``viewer.png`` to
``OUT_DIR`` (default ``assets``). The rasteriser has no Fira Code font, so the
script changes the SVG font to DejaVu Sans Mono first.
"""

import asyncio
import sys
from pathlib import Path

from stargazer.viewer import load, make_viewer

SIZE = (160, 46)
"""Terminal size of the screenshot in cells: columns, rows."""

PNG_WIDTH = 1600
"""Width of the PNG in pixels."""

FONTS = ("font-family: Fira Code, monospace", "font-family: DejaVu Sans Mono, monospace")
"""The font that Textual writes into the SVG and the font that replaces it."""


async def capture(results_path: Path, keys: tuple[str, ...] = ("down", "r")) -> str:
    """Start the viewer on a results file, press the keys, and return the screen as SVG."""
    results, truth = load(results_path)
    app = make_viewer(results, truth, title=str(results_path))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        for key in keys:
            await pilot.press(key)
            await pilot.pause()
        await pilot.pause(0.3)
        return app.export_screenshot()


def main(arguments: list[str]) -> int:
    """Write the SVG and the PNG screenshot. Return the exit code."""
    import cairosvg

    results_path = Path(arguments[0])
    out_dir = Path(arguments[1]) if len(arguments) > 1 else Path("assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = asyncio.run(capture(results_path)).replace(*FONTS)
    svg_path, png_path = out_dir / "viewer.svg", out_dir / "viewer.png"
    svg_path.write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), output_width=PNG_WIDTH)
    print(f"wrote {svg_path} and {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
