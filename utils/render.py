from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple


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


def ansi_frames_to_gif(
    data: Dict[str, Any],
    out_path: str = "episode.gif",
    fps: int = 2,
    font_size: int = 18,
    margin: int = 16,
    line_spacing: int = 2,
    fg=(0,0,0),
    bg=(255,255,255),
) -> str:
    """Collect episodes[*].steps[*].render_ansi and save as an animated GIF."""
    font = _pick_mono_font(font_size)

    # Collect text frames
    texts: List[str] = []
    for ep in data.get("episodes", []):
        for st in ep.get("steps", []):
            t = st.get("render_ansi")
            if isinstance(t, str) and t.strip():
                texts.append(t)
    if not texts:
        raise ValueError("No render_ansi frames found.")

    # Measure to get a consistent canvas across frames
    def _measure(txt: str) -> Tuple[int, int]:
        d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        l,t,r,b = d.multiline_textbbox((0,0), txt, font=font, spacing=line_spacing)
        return r - l, b - t

    sizes = [_measure(t) for t in texts]
    max_w = max(w for w,h in sizes) + 2 * margin
    max_h = max(h for w,h in sizes) + 2 * margin

    # Render each frame
    frames: List[Image.Image] = []
    for txt, (w,h) in zip(texts, sizes):
        img = Image.new("RGB", (max_w, max_h), color=bg)
        d = ImageDraw.Draw(img)
        x = (max_w - (w + 2 * margin)) // 2 + margin
        y = (max_h - (h + 2 * margin)) // 2 + margin
        d.multiline_text((x, y), txt, font=font, fill=fg, spacing=line_spacing, align="left")
        frames.append(img)

    # Save as GIF
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
