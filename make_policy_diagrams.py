"""Generate two PNG diagrams describing the policy architectures:
- MLP policy used with flat observations (SB3 MlpPolicy defaults)
- CNN policy used with conv observations (SmallObsCNN -> features -> MLP heads)

The script tries to use PIL; if Pillow is not available it falls back to matplotlib.
"""
from __future__ import annotations

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visualizations')
os.makedirs(OUT_DIR, exist_ok=True)

WIDTH = 1000
HEIGHT = 420
BG = (255, 255, 255)
BOX_FILL = (230, 240, 255)
BOX_EDGE = (30, 60, 120)
TEXT_COLOR = (10, 10, 10)
ARROW_COLOR = (80, 80, 80)

try:
    from PIL import Image, ImageDraw, ImageFont
    has_pil = True
except Exception:
    has_pil = False

if has_pil:
    font = ImageFont.load_default()

    def draw_box(draw, xy, text, fill=BOX_FILL, edge=BOX_EDGE):
        draw.rectangle(xy, fill=fill, outline=edge, width=2)
        # center text
        x0, y0, x1, y1 = xy
        w = x1 - x0
        h = y1 - y0
        lines = text.split('\n')
        # compute text size with best available API
        if hasattr(font, 'getsize'):
            def text_size(s):
                return font.getsize(s)
        elif hasattr(draw, 'textbbox'):
            def text_size(s):
                bbox = draw.textbbox((0, 0), s, font=font)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        else:
            def text_size(s):
                # fallback guess (monospace-like)
                return (6 * len(s), 11)

        total_h = sum([text_size(line)[1] for line in lines])
        cur_y = y0 + (h - total_h) // 2
        for line in lines:
            tw, th = text_size(line)
            draw.text((x0 + (w - tw) / 2, cur_y), line, font=font, fill=TEXT_COLOR)
            cur_y += th

    def arrow(draw, p0, p1, width=2):
        draw.line([p0, p1], fill=ARROW_COLOR, width=width)
        # small arrowhead
        ax, ay = p1
        bx, by = p0
        # direction
        import math
        ang = math.atan2(ay - by, ax - bx)
        ah = 10
        ang1 = ang + 0.35
        ang2 = ang - 0.35
        draw.polygon([(ax, ay), (ax - ah * math.cos(ang1), ay - ah * math.sin(ang1)), (ax - ah * math.cos(ang2), ay - ah * math.sin(ang2))], fill=ARROW_COLOR)

    def make_mlp_diagram(path):
        im = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(im)
        margin = 40
        y = HEIGHT // 2
        # positions for boxes
        xs = [margin + i * (WIDTH - 2 * margin) / 4 for i in range(4)]
        box_w = 200
        box_h = 80
        # Input
        draw_box(draw, (xs[0], y - box_h // 2, xs[0] + box_w, y + box_h // 2), 'Input vector\n(1570 dims)')
        draw_box(draw, (xs[1], y - box_h // 2, xs[1] + box_w, y + box_h // 2), 'Dense 64\nReLU')
        draw_box(draw, (xs[2], y - box_h // 2, xs[2] + box_w, y + box_h // 2), 'Dense 64\nReLU')
        draw_box(draw, (xs[3], y - box_h // 2, xs[3] + box_w, y + box_h // 2), 'Output logits\n(4 classes)\nSoftmax -> Categorical')
        # arrows
        arrow(draw, (xs[0] + box_w + 10, y), (xs[1] - 10, y))
        arrow(draw, (xs[1] + box_w + 10, y), (xs[2] - 10, y))
        arrow(draw, (xs[2] + box_w + 10, y), (xs[3] - 10, y))
        # title
        draw.text((WIDTH // 2 - 160, 10), 'SB3 MlpPolicy (flat observations)', font=font, fill=TEXT_COLOR)
        im.save(path)

    def make_cnn_diagram(path):
        im = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(im)
        title = 'CnnPolicy with SmallObsCNN feature extractor (conv observations)'
        draw.text((WIDTH // 2 - 340, 10), title, font=font, fill=TEXT_COLOR)
        # left column: conv stack
        x0 = 60
        w_box = 220
        box_h = 70
        y0 = 80
        spacing = 20
        draw_box(draw, (x0, y0, x0 + w_box, y0 + box_h), 'Input image\n(3x28x28)')
        y1 = y0 + box_h + spacing
        draw_box(draw, (x0, y1, x0 + w_box, y1 + box_h), 'Conv2d 3->32\n3x3 pad=1\nReLU')
        y2 = y1 + box_h + spacing
        draw_box(draw, (x0, y2, x0 + w_box, y2 + box_h), 'Conv2d 32->64\n3x3 pad=1\nReLU')
        y3 = y2 + box_h + spacing
        draw_box(draw, (x0, y3, x0 + w_box, y3 + box_h), 'MaxPool2d 2\n(28x28 -> 14x14)')
        # flatten + linear
        mid_x = x0 + w_box + 120
        draw_box(draw, (mid_x, y1 + 10, mid_x + w_box, y1 + 10 + box_h), 'Flatten -> Linear\n-> 256 features\nReLU')
        # arrows
        arrow(draw, (x0 + w_box + 10, y0 + box_h // 2), (mid_x - 10, y1 + 10 + box_h // 2))
        # from features to two heads
        head_x = mid_x + w_box + 120
        # actor head
        draw_box(draw, (head_x, y1 - 10, head_x + w_box, y1 - 10 + box_h), 'Actor MLP\nDense 64\nDense 64\nLogits (4)\nSoftmax -> action probs')
        # critic head
        draw_box(draw, (head_x, y2 + 40, head_x + w_box, y2 + 40 + box_h), 'Critic MLP\nDense 64\nDense 64\nValue (1)')
        # arrows
        arrow(draw, (mid_x + w_box + 10, y1 + 10 + box_h // 2), (head_x - 10, y1 - 10 + box_h // 2))
        arrow(draw, (mid_x + w_box + 10, y1 + 10 + box_h // 2), (head_x - 10, y2 + 40 + box_h // 2))
        im.save(path)

    # generate
    mlp_path = os.path.join(OUT_DIR, 'policy_mlp.png')
    cnn_path = os.path.join(OUT_DIR, 'policy_cnn.png')
    make_mlp_diagram(mlp_path)
    make_cnn_diagram(cnn_path)
    print('Wrote:', mlp_path)
    print('Wrote:', cnn_path)

else:
    # fallback using matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def make_simple_diagram_rects(rects, title, path):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis('off')
        for r in rects:
            x, y, w, h, label = r
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='navy', facecolor='#E6F0FF')
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, label, ha='center', va='center')
        ax.set_title(title)
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

    mlp_rects = [ (0.5, 1.5, 2, 1, 'Input (1570)'), (3.2, 1.5, 2, 1, 'Dense 64\nReLU'), (5.9, 1.5, 2, 1, 'Dense 64\nReLU'), (8.6, 1.5, 1.0, 1, 'Logits (4)') ]
    cnn_rects = [ (0.2, 0.8, 1.6, 1, 'Input 3x28x28'), (2.0, 0.8, 1.8, 1, 'Conv 3->32\n3x3 ReLU'), (4.0, 0.8, 1.8, 1, 'Conv 32->64\n3x3 ReLU'), (6.0, 0.8, 1.2, 1, 'MaxPool 2'), (7.6, 0.8, 1.6, 1, 'Flatten->Linear 256') ]
    mlp_path = os.path.join(OUT_DIR, 'policy_mlp.png')
    cnn_path = os.path.join(OUT_DIR, 'policy_cnn.png')
    make_simple_diagram_rects(mlp_rects, 'SB3 MlpPolicy (flat obs)', mlp_path)
    make_simple_diagram_rects(cnn_rects, 'CnnPolicy + SmallObsCNN', cnn_path)
    print('Wrote (matplotlib):', mlp_path)
    print('Wrote (matplotlib):', cnn_path)
