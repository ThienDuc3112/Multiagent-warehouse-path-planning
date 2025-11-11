from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple

ACTION_NAMES = {0: "WAIT", 1: "NORTH", 2: "SOUTH", 3: "WEST", 4: "EAST"}


def _pick_mono_font(font_size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",  # macOS
        "/Library/Fonts/Microsoft/Consolas.ttf",
        "Consolas.ttf", "Menlo.ttc", "Courier New.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def _fmt_action(robot: str, a_int: int, logp: float | None) -> str:
    name = ACTION_NAMES.get(a_int, str(a_int))
    return f"{robot}: {a_int:>1} {name:<5}" if logp is None else f"{robot}: {a_int:>1} {name:<5} (logp={logp:.3g})"


def ansi_frames_to_gif(
    data: Dict[str, Any],
    out_path: str = "episode_with_stats.gif",
    fps: int = 2,
    font_size: int = 18,
    margin: int = 16,
    line_spacing: int = 2,
    fg: Tuple[int,int,int] = (0,0,0),
    bg: Tuple[int,int,int] = (255,255,255),
    show_t_global: bool = True,
    show_t_local: bool = True,
    show_entropy: bool = True,
    show_actions: bool = True,
    show_logps: bool = True,
) -> str:
    """Renders ANSI grid + a stats panel per frame into a GIF."""
    font = _pick_mono_font(font_size)
    robots: List[str] = data.get("meta", {}).get("robots", [])
    frames_text: List[str] = []

    for ep in data.get("episodes", []):
        for st in ep.get("steps", []):
            grid_txt = st.get("render_ansi", "").rstrip("\n")
            if not grid_txt:
                continue

            # Header: steps, reward, done, entropy
            hdr = []
            if show_t_global and "t_global" in st:
                hdr.append(f"t_global={st['t_global']:03d}")
            if show_t_local and "t_local" in st:
                hdr.append(f"t_local={st['t_local']:03d}")
            if "reward" in st:
                hdr.append(f"reward={' ' if st['reward']>=0.0 else ''}{st['reward']:.3f}")
            if "done" in st:
                hdr.append(f"done={st['done']}")
            if show_entropy and "entropy_est" in st:
                hdr.append(f"entropy={st['entropy_est']:.6f}")
            header = " | ".join(hdr)

            # Actions: per robot with names + logps
            act_lines = []
            if show_actions and isinstance(st.get("action"), dict):
                logps = st.get("logps", {}) if show_logps else {}
                keys = robots if robots else sorted(st["action"].keys())
                for r in keys:
                    a = st["action"].get(r)
                    if a is None:
                        continue
                    lp = logps.get(r) if isinstance(logps, dict) else None
                    act_lines.append(_fmt_action(r, a, lp))
            action_block = "\n".join(act_lines)

            combined = f"{header}\n{action_block}\n\n{grid_txt}" if action_block else f"{header}\n\n{grid_txt}"
            frames_text.append(combined)

    if not frames_text:
        raise ValueError("No frames found.")

    # Measure to get a consistent canvas
    def _measure(txt: str) -> Tuple[int,int]:
        d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        l,t,r,b = d.multiline_textbbox((0,0), txt, font=font, spacing=line_spacing)
        return r - l, b - t

    sizes = [_measure(t) for t in frames_text]
    max_w = max(w for w,h in sizes) + 2 * margin
    max_h = max(h for w,h in sizes) + 2 * margin

    # Render frames
    frames = []
    for txt, (w,h) in zip(frames_text, sizes):
        img = Image.new("RGB", (max_w, max_h), color=bg)
        d = ImageDraw.Draw(img)
        x = (max_w - (w + 2 * margin)) // 2 + margin
        y = (max_h - (h + 2 * margin)) // 2 + margin
        d.multiline_text((x, y), txt, font=font, fill=fg, spacing=line_spacing, align="left")
        frames.append(img)

    # Save GIF
    duration_ms = max(1, int(1000 / max(1, fps)))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0,
                   duration=duration_ms, disposal=2, optimize=False)
    return out_path
