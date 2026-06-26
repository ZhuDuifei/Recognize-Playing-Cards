import os
import random
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ===================== 可自定义配置区 =====================
TEXT_SIZE_MIN = 12
TEXT_SIZE_MAX = 38
LINE_GAP_MIN = 0
LINE_GAP_MAX = 8
JOK_INNER_GAP = 0
GROUP_MIN = 1
GROUP_MAX = 10
PADDING = 2
PAD_RIGHT_EXTRA = 8
PAD_DOWN_EXTRA = 16
# 文字透明度配置 0完全透明 255不透明
TEXT_ALPHA_MIN = 150
TEXT_ALPHA_MAX = 255
# 真实卡牌贴图透明度
REAL_CARD_ALPHA_MIN = 150
REAL_CARD_ALPHA_MAX = 255

REAL_CARD_MIN_NUM = 1
REAL_CARD_MAX_NUM = 15
REAL_SCALE_MIN = 0.3
REAL_SCALE_MAX = 0.6
REAL_CARD_FOLDER = "cards"
REAL_BOX_JSON = "box_info.json"

GEN_COUNT = 500
BG_FOLDER = "background"
OUTPUT_IMG_DIR = "images/val"
OUTPUT_LABEL_DIR = "labels/val"
OUTPUT_VIS_DIR = "images/visualization"
FONT_TXT_PATH = "selected_fonts.txt"
MAX_POS_RETRY = 300
DRAW_VISUALIZATION = False
# ==========================================================

card_full_list = [
    "10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S",
    "3C", "3D", "3H", "3S", "4C", "4D", "4H", "4S",
    "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S",
    "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S",
    "9C", "9D", "9H", "9S", "AC", "AD", "AH", "AS",
    "JC", "JD", "JH", "JS", "KC", "KD", "KH", "KS",
    "QC", "QD", "QH", "QS", "REDJOK", "BLACKJOK"
]
card_class_map = {card: idx for idx, card in enumerate(card_full_list)}

suit_symbol_map = {
    "C": ("♣", "black"), "D": ("♦", "red"),
    "H": ("♥", "red"), "S": ("♠", "black")
}
jok_config = {
    "REDJOK": ("J\nO\nK", "red"),
    "BLACKJOK": ("J\nO\nK", "black")
}

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
if DRAW_VISUALIZATION:
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)


def load_font_pool_from_txt():
    font_paths = []
    with open(FONT_TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "|" in line and not line.startswith(("字体名称", "-")):
                font_fp = line.split("|")[1].strip()
                if os.path.exists(font_fp):
                    try:
                        ImageFont.truetype(font_fp, 20)
                        font_paths.append(font_fp)
                    except:
                        pass
    return font_paths


GLOBAL_FONT_POOL = load_font_pool_from_txt()

BG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
BG_CACHE_POOL = []
for name in os.listdir(BG_FOLDER):
    if name.lower().endswith(BG_EXTS):
        BG_CACHE_POOL.append(Image.open(os.path.join(BG_FOLDER, name)).convert("RGB"))

real_card_meta = []


def load_real_labeled_cards():
    global real_card_meta
    if not os.path.exists(REAL_BOX_JSON):
        if os.path.exists(REAL_CARD_FOLDER):
            for file_name in os.listdir(REAL_CARD_FOLDER):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    real_card_meta.append((os.path.join(REAL_CARD_FOLDER, file_name), []))
        return

    with open(REAL_BOX_JSON, "r", encoding="utf-8") as f:
        box_data = json.load(f)

    all_card_images = {f for f in os.listdir(REAL_CARD_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))}
    processed = set()

    for img_name, box_list in box_data.items():
        img_path = os.path.join(REAL_CARD_FOLDER, img_name)
        processed.add(img_name)
        if not os.path.exists(img_path):
            continue

        if not box_list:
            real_card_meta.append((img_path, []))
            continue

        boxes_info = []
        for box in box_list:
            if len(box) >= 6:
                x1, y1, x2, y2, _, cid = box[:6]
            elif len(box) >= 5:
                x1, y1, x2, y2, cls_name = box[:5]
                cid = card_class_map.get(cls_name, -1)
            else:
                continue

            if x2 - x1 >= 10 and y2 - y1 >= 10 and 0 <= cid < len(card_full_list):
                boxes_info.append((cid, x1, y1, x2, y2))

        real_card_meta.append((img_path, boxes_info))

    for img_name in (all_card_images - processed):
        real_card_meta.append((os.path.join(REAL_CARD_FOLDER, img_name), []))


load_real_labeled_cards()


def get_random_background():
    return random.choice(BG_CACHE_POOL).copy()


def rect_intersect(r1, r2):
    return not (r1[2] < r2[0] or r2[2] < r1[0] or r1[3] < r2[1] or r2[3] < r1[1])


def xyxy2yolo(cls_id, xyxy, img_w, img_h):
    l, t, r, b = xyxy
    l, r = max(0, min(l, img_w)), max(0, min(r, img_w))
    t, b = max(0, min(t, img_h)), max(0, min(b, img_h))

    bw, bh = r - l, b - t
    if bw <= 0 or bh <= 0:
        return None

    xc = (l + bw / 2) / img_w
    yc = (t + bh / 2) / img_h
    wn, hn = bw / img_w, bh / img_h

    return f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def draw_validation_boxes(img, boxes, class_names):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        if len(box) >= 5:
            x1, y1, x2, y2, cls_id = box[:5]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

            if cls_id is not None and class_names and cls_id < len(class_names):
                label = class_names[cls_id]
                try:
                    font = ImageFont.load_default()
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except:
                    text_w, text_h = len(label) * 8, 12

                draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=(255, 0, 0))
                draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)
    return img


def generate_one_image(save_index):
    img = get_random_background()
    draw = ImageDraw.Draw(img)
    IMG_W, IMG_H = img.size
    used_boxes = []
    yolo_lines = []
    all_boxes = []

    # 文字生成卡牌
    for _ in range(random.randint(GROUP_MIN, GROUP_MAX)):
        full_card = random.choice(card_full_list)
        cls_id = card_class_map[full_card]

        font = ImageFont.truetype(random.choice(GLOBAL_FONT_POOL), random.randint(TEXT_SIZE_MIN, TEXT_SIZE_MAX))
        vertical_gap = random.randint(LINE_GAP_MIN, LINE_GAP_MAX)

        is_jok = full_card in ("REDJOK", "BLACKJOK")
        # 随机文字透明度
        alpha = random.randint(TEXT_ALPHA_MIN, TEXT_ALPHA_MAX)
        if (is_jok and jok_config[full_card][1] == "red") or (not is_jok and suit_symbol_map[full_card[-1]][1] == "red"):
            color = (200, 0, 0, alpha)
        else:
            color = (0, 0, 0, alpha)

        if is_jok:
            text = jok_config[full_card][0]
            bbox = draw.textbbox((0, 0), text, font=font, spacing=JOK_INNER_GAP)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            max_w, total_h = text_w, text_h
        else:
            rank, suit = full_card[:-1], suit_symbol_map[full_card[-1]][0]
            b1 = draw.textbbox((0, 0), rank, font=font)
            b2 = draw.textbbox((0, 0), suit, font=font)
            max_w = max(b1[2] - b1[0], b2[2] - b2[0])
            total_h = (b1[3] - b1[1]) * 2 + vertical_gap

        find_ok = False
        for _ in range(MAX_POS_RETRY):
            x = random.randint(10, IMG_W - max_w - 20)
            y = random.randint(10, IMG_H - total_h - 20)
            rect = (x - PADDING, y - PADDING, x + max_w + PADDING + PAD_RIGHT_EXTRA,
                    y + total_h + PADDING + PAD_DOWN_EXTRA)

            if not any(rect_intersect(rect, old) for old in used_boxes):
                used_boxes.append(rect)
                find_ok = True
                break

        if not find_ok:
            continue

        gray = random.randint(240, 252)
        draw.rectangle([x - PADDING, y, x + max_w + PADDING + PAD_RIGHT_EXTRA, y + total_h + PADDING + PAD_DOWN_EXTRA],
                       fill=(gray, gray, gray))

        if is_jok:
            text_x = x + (max_w - text_w) / 2
            text_y = y
            text_canvas = Image.new("RGBA", (IMG_W, IMG_H), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_canvas)
            text_draw.text((text_x, text_y), text, font=font, fill=color, spacing=JOK_INNER_GAP)
            img.paste(text_canvas, mask=text_canvas)
            text_bbox = text_draw.textbbox((text_x, text_y), text, font=font, spacing=JOK_INNER_GAP)
            ll, tt, rr, bb = text_bbox
        else:
            rank_w = b1[2] - b1[0]
            suit_w = b2[2] - b2[0]
            line_h = b1[3] - b1[1]

            pos1 = (x + (max_w - rank_w) / 2, y)
            pos2 = (x + (max_w - suit_w) / 2, y + line_h + vertical_gap)
            text_canvas = Image.new("RGBA", (IMG_W, IMG_H), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_canvas)
            text_draw.text(pos1, rank, font=font, fill=color)
            text_draw.text(pos2, suit, font=font, fill=color)
            img.paste(text_canvas, mask=text_canvas)

            b_rank = text_draw.textbbox(pos1, rank, font=font)
            b_suit = text_draw.textbbox(pos2, suit, font=font)
            ll = min(b_rank[0], b_suit[0])
            tt = min(b_rank[1], b_suit[1])
            rr = max(b_rank[2], b_suit[2])
            bb = max(b_rank[3], b_suit[3])

        yolo_line = xyxy2yolo(cls_id, [ll, tt, rr, bb], IMG_W, IMG_H)
        if yolo_line:
            yolo_lines.append(yolo_line)
            all_boxes.append((ll, tt, rr, bb, cls_id))

    # 粘贴真实卡牌（新增卡牌透明度）
    if real_card_meta:
        shuffled = real_card_meta.copy()
        random.shuffle(shuffled)

        for img_path, boxes_info in shuffled[:random.randint(REAL_CARD_MIN_NUM, REAL_CARD_MAX_NUM)]:
            try:
                card_img = Image.open(img_path).convert("RGB")
                w, h = card_img.size

                scale = random.uniform(REAL_SCALE_MIN, REAL_SCALE_MAX)
                new_w = int(w * scale)
                new_h = int(h * scale)

                max_w = int(IMG_W * 0.5)
                max_h = int(IMG_H * 0.5)
                if new_w > max_w:
                    new_w = max_w
                    new_h = int(h * (max_w / w))
                if new_h > max_h:
                    new_h = max_h
                    new_w = int(w * (max_h / h))

                new_w = max(30, new_w)
                new_h = max(30, new_h)

                resize_card = card_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                card_img.close()

                placed = False
                for _ in range(MAX_POS_RETRY):
                    px = random.randint(0, IMG_W - new_w)
                    py = random.randint(0, IMG_H - new_h)
                    rect = (px - 2, py - 2, px + new_w + 2, py + new_h + 2)

                    if not any(rect_intersect(rect, old) for old in used_boxes):
                        used_boxes.append(rect)
                        # ========== 真实卡牌透明度逻辑 ==========
                        card_alpha = random.randint(REAL_CARD_ALPHA_MIN, REAL_CARD_ALPHA_MAX)
                        # 转RGBA并统一透明度
                        rgba_card = resize_card.convert("RGBA")
                        r, g, b, a = rgba_card.split()
                        a = a.point(lambda i: card_alpha)
                        rgba_card = Image.merge("RGBA", (r, g, b, a))
                        # 叠加到背景图
                        img.paste(rgba_card, (px, py), mask=rgba_card)
                        resize_card.close()
                        rgba_card.close()
                        placed = True
                        break

                if not placed:
                    continue

                if boxes_info:
                    scale_x = new_w / w
                    scale_y = new_h / h
                    for cls_id, ox1, oy1, ox2, oy2 in boxes_info:
                        sx1 = max(0, min(px + ox1 * scale_x, IMG_W))
                        sy1 = max(0, min(py + oy1 * scale_y, IMG_H))
                        sx2 = max(0, min(px + ox2 * scale_x, IMG_W))
                        sy2 = max(0, min(py + oy2 * scale_y, IMG_H))

                        if sx2 - sx1 >= 5 and sy2 - sy1 >= 5:
                            yolo_line = xyxy2yolo(cls_id, [sx1, sy1, sx2, sy2], IMG_W, IMG_H)
                            if yolo_line:
                                yolo_lines.append(yolo_line)
                                all_boxes.append((sx1, sy1, sx2, sy2, cls_id))
            except Exception as e:
                continue

    # 保存
    img.save(os.path.join(OUTPUT_IMG_DIR, f"gen_{save_index}.png"), compress_level=1)

    if DRAW_VISUALIZATION and all_boxes:
        vis_img = draw_validation_boxes(img.copy(), all_boxes, card_full_list)
        vis_img.save(os.path.join(OUTPUT_VIS_DIR, f"vis_{save_index}.png"), compress_level=1)

    with open(os.path.join(OUTPUT_LABEL_DIR, f"gen_{save_index}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) if yolo_lines else "")


def main_batch():
    with tqdm(total=GEN_COUNT, desc="生成进度") as pbar:
        for idx in range(1, GEN_COUNT + 1):
            generate_one_image(idx)
            pbar.update(1)


if __name__ == "__main__":
    main_batch()