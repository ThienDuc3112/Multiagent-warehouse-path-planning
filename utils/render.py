from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont

# Map action ints to readable names
ACTION_NAMES: Dict[int, str] = {
    0: "WAIT",
    1: "NORTH",
    2: "SOUTH",
    3: "WEST",
    4: "EAST",
}

# ---------- Styling knobs ----------
CELL = 26   # tile size in px
CELL_PAD = 2    # inner padding for rounded tile
MARGIN = 16   # outer image margin
GRIDLINE = 1    # grid line width (0 to disable)
PANEL_MIN_W = 100  # minimum width for right-hand stats panel
FONT_TILE = 12   # on-tile labels (R1, A1, etc.)
FONT_PANEL = 16   # sidebar text
FPS_DEFAULT = 2    # default frames per second

# Colors
COL_BG = (250, 250, 252)
COL_FLOOR = (235, 238, 242)
COL_GRID = (210, 214, 220)
COL_WALL = (0, 0, 0)            # walls = solid black (no '#' text shown)

# Per-entity colors (adjust as you like)
COL_ROBOTS = {"R1": (64, 120, 242), "R2": (236, 70, 91), "R3": (54, 188, 155)}
COL_TARGETS = {"A": (252, 191, 73), "B": (163, 122, 255), "C": (255, 145, 77)}


def _pick_font(size: int) -> ImageFont.FreeTypeFont:
    """Try a few monospaced fonts and fall back to default."""
    candidates = [
        "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Microsoft/Consolas.ttf",
        "Consolas.ttf", "Menlo.ttc", "Courier New.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_TILE_OBJ = _pick_font(FONT_TILE)
FONT_PANEL_OBJ = _pick_font(FONT_PANEL)

# ---------- Parsing / token helpers ----------


def _split_grid_lines(render_ansi: str) -> List[List[str]]:
    """Drop the first 't=... (...)' line and split each grid line on ANY whitespace."""
    lines = render_ansi.splitlines()
    if not lines:
        return []
    # Heuristic: first line is a header ("t=... (render post-reset)")
    lines = lines[1:]
    grid: List[List[str]] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        toks = re.split(r"\s+", s)     # robust: any whitespace separates cells
        toks = [t for t in toks if t]
        grid.append(toks)
    return grid


def _is_wall(tok: str) -> bool:
    """Anything containing '#' is treated as a wall cell."""
    return "#" in tok


def _token_color(tok: str) -> Tuple[int, int, int]:
    if _is_wall(tok):
        return COL_WALL
    if tok in (".", "·"):
        return COL_FLOOR
    if tok in COL_ROBOTS:
        return COL_ROBOTS[tok]
    if tok and tok[0] in COL_TARGETS:
        return COL_TARGETS[tok[0]]
    return COL_FLOOR


def _token_label(tok: str) -> str:
    """No labels for walls/floor; show labels for robots/targets like 'R1','A2'."""
    if _is_wall(tok) or tok in (".", "·"):
        return ""
    return tok

# ---------- Stats panel ----------


def _measure_panel(text: str) -> Tuple[int, int]:
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    l, t, r, b = d.multiline_textbbox((0, 0), text, font=FONT_PANEL_OBJ, spacing=2)
    return r - l, b - t


def _compose_panel(step: Dict[str, Any], robots_in_meta: List[str], show_logps: bool = True) -> str:
    lines = []
    if "t_global" in step:
        lines.append(f"t_global={step['t_global']}")
    if "t_local" in step:
        lines.append(f"t_local={step['t_local']}")
    if "reward" in step:
        lines.append(f"reward={step['reward']:0.4f}")
    if "done" in step:
        lines.append(f"done={step['done']}")
    if "entropy_est" in step:
        lines.append(f"entropy={step['entropy_est']:.6f}")

    lines.append("")
    lines.append("Actions:")
    acts = step.get("action", {}) or {}
    logps = step.get("logps", {}) if show_logps else {}
    keys = robots_in_meta if robots_in_meta else sorted(acts.keys())

    for r in keys:
        a = acts.get(r)
        if a is None:
            continue
        name = ACTION_NAMES.get(a, str(a))
        if isinstance(logps, dict) and r in logps and show_logps:
            lines.append(f"  {r}: {a} {name:<5}  (logp={logps[r]:.3g})")
        else:
            lines.append(f"  {r}: {a} {name}")
    return "\n".join(lines)

# ---------- Main renderer ----------


def ansi_frames_to_gif(
    data: Dict[str, Any],
    out_path: str = "episode_color.gif",
    fps: int = FPS_DEFAULT,
) -> str:
    """
    Render an animated color GIF from the given ANSI-grid JSON.
    Returns the output path.
    """
    episodes: List[Dict[str, Any]] = data.get("episodes", []) or []
    robots_meta: List[str] = data.get("meta", {}).get("robots", []) or []

    frames_parsed: List[Tuple[List[List[str]], str]] = []
    max_rows = max_cols = 0
    max_panel_w = PANEL_MIN_W
    max_panel_h = 0

    # Parse everything first to lock dimensions
    for ep in episodes[:1]:
        for st in ep.get("steps", []) or []:
            ra = st.get("render_ansi", "")
            if not isinstance(ra, str) or not ra.strip():
                continue
            grid = _split_grid_lines(ra)
            if not grid:
                continue

            rows = len(grid)
            cols = max((len(r) for r in grid), default=0)
            max_rows = max(max_rows, rows)
            max_cols = max(max_cols, cols)

            panel_text = _compose_panel(st, robots_meta, show_logps=True)
            pw, ph = _measure_panel(panel_text)
            max_panel_w = max(max_panel_w, pw)
            max_panel_h = max(max_panel_h, ph)

            frames_parsed.append((grid, panel_text))

    if not frames_parsed:
        raise ValueError("No frames found in data (missing render_ansi?).")

    # Canvas geometry
    grid_w = max_cols * CELL + (max_cols - 1) * GRIDLINE
    grid_h = max_rows * CELL + (max_rows - 1) * GRIDLINE
    panel_w = max(PANEL_MIN_W, max_panel_w + 16)
    panel_h = max(grid_h, max_panel_h)

    W = MARGIN + grid_w + MARGIN + panel_w + MARGIN
    H = MARGIN + max(grid_h, panel_h) + MARGIN

    # Render frames
    frames: List[Image.Image] = []
    for grid, panel_text in frames_parsed:
        img = Image.new("RGB", (W, H), COL_BG)
        d = ImageDraw.Draw(img)

        gx, gy = MARGIN, MARGIN

        # grid tiles
        for r in range(max_rows):
            row = grid[r] if r < len(grid) else []
            for c in range(max_cols):
                tok = row[c] if c < len(row) else "."
                x0 = gx + c * (CELL + GRIDLINE)
                y0 = gy + r * (CELL + GRIDLINE)
                x1, y1 = x0 + CELL, y0 + CELL

                # base & inner rounded colored tile
                d.rectangle([x0, y0, x1, y1], fill=COL_FLOOR)
                col = _token_color(tok)
                d.rounded_rectangle([x0 + CELL_PAD, y0 + CELL_PAD, x1 - CELL_PAD, y1 - CELL_PAD],
                                    radius=6, fill=col)

                # label (never for walls/floor)
                lbl = _token_label(tok)
                if lbl:
                    l, t, r2, b2 = d.textbbox((0, 0), lbl, font=FONT_TILE_OBJ)
                    tw, th = (r2 - l), (b2 - t)
                    tx = x0 + (CELL - tw) // 2
                    ty = y0 + (CELL - th) // 2 - 1
                    # thin dark outline for contrast
                    for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                        d.text((tx + dx, ty + dy), lbl, font=FONT_TILE_OBJ, fill=(0, 0, 0))
                    d.text((tx, ty), lbl, font=FONT_TILE_OBJ, fill=(255, 255, 255))

                if GRIDLINE:
                    d.rectangle([x0, y0, x1, y1], outline=COL_GRID, width=GRIDLINE)

        # stats panel
        px = gx + grid_w + MARGIN
        py = gy
        d.rounded_rectangle([px, py, px + panel_w, py + panel_h],
                            radius=10, fill=(246, 247, 250), outline=COL_GRID, width=1)
        d.multiline_text((px + 10, py + 10), panel_text,
                         font=FONT_PANEL_OBJ, fill=(30, 33, 39), spacing=4)

        frames.append(img)

    # Save GIF
    duration_ms = max(1, int(1000 / max(1, fps)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        disposal=2,
        optimize=False,
    )
    return out_path


# ---------- Optional: quick CLI test ----------
if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 3:
        print("Usage: python render_color_gif.py <input.json> <output.gif> [fps]")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)
    fps = int(sys.argv[3]) if len(sys.argv) >= 4 else FPS_DEFAULT
    outp = ansi_frames_to_gif(payload, out_path=sys.argv[2], fps=fps)
    print(f"Wrote: {outp}")
