import os
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ===================== 可自定义配置区 =====================
TEXT_SIZE_MIN = 12
TEXT_SIZE_MAX = 38
LINE_GAP_MIN = 0
LINE_GAP_MAX = 8
# JOK竖排文字内部紧凑行距（固定更小，单独可调）
JOK_INNER_GAP = 0
GROUP_MIN = 1
GROUP_MAX = 30
GROUP_SPACE = 10
PADDING = 3
PAD_RIGHT_EXTRA = 10
PAD_DOWN_EXTRA = 18

GEN_COUNT = 10000
BG_FOLDER = "./background"
OUTPUT_IMG_DIR = "./images/train"
OUTPUT_LABEL_DIR = "./labels/train"
FONT_TXT_PATH = "selected_fonts.txt"
MAX_POS_RETRY = 120
# ==========================================================

card_full_list = [
    "10C","10D","10H","10S",
    "2C","2D","2H","2S",
    "3C","3D","3H","3S",
    "4C","4D","4H","4S",
    "5C","5D","5H","5S",
    "6C","6D","6H","6S",
    "7C","7D","7H","7S",
    "8C","8D","8H","8S",
    "9C","9D","9H","9S",
    "AC","AD","AH","AS",
    "JC","JD","JH","JS",
    "KC","KD","KH","KS",
    "QC","QD","QH","QS",
    "REDJOK", "BLACKJOK"
]
card_class_map = {card: idx for idx, card in enumerate(card_full_list)}

suit_symbol_map = {
    "C": ("♣", "black"),
    "D": ("♦", "red"),
    "H": ("♥", "red"),
    "S": ("♠", "black")
}
# JOK只保留一段竖排文本，不再分两行绘制
jok_config = {
    "REDJOK": ("J\nO\nK", "red"),
    "BLACKJOK": ("J\nO\nK", "black")
}

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def load_font_pool_from_txt():
    font_paths = []
    if not os.path.exists(FONT_TXT_PATH):
        raise FileNotFoundError(f"找不到字体文件：{FONT_TXT_PATH}")
    with open(FONT_TXT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or "|" not in line or line.startswith("字体名称") or line.startswith("-"):
            continue
        _, font_fp = line.split("|")
        font_fp = font_fp.strip()
        if os.path.exists(font_fp):
            try:
                ImageFont.truetype(font_fp, 20)
                font_paths.append(font_fp)
            except Exception:
                continue
    if not font_paths:
        raise RuntimeError("selected_fonts.txt 无有效字体！")
    return font_paths

GLOBAL_FONT_POOL = load_font_pool_from_txt()

BG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
if not os.path.exists(BG_FOLDER):
    os.makedirs(BG_FOLDER)
    raise FileNotFoundError(f"请创建 {BG_FOLDER} 并放入背景图")
BG_FILE_LIST = [f for f in os.listdir(BG_FOLDER) if f.lower().endswith(BG_EXTS)]
if not BG_FILE_LIST:
    raise FileNotFoundError(f"{BG_FOLDER} 文件夹内无图片")
BG_CACHE_POOL = []
for name in BG_FILE_LIST:
    path = os.path.join(BG_FOLDER, name)
    img = Image.open(path).convert("RGB")
    BG_CACHE_POOL.append(img)

def get_random_background():
    return random.choice(BG_CACHE_POOL).copy()

def rect_intersect(r1, r2):
    l1, t1, r1x, b1 = r1
    l2, t2, r2x, b2 = r2
    if r1x < l2 or r2x < l1:
        return False
    if b1 < t2 or b2 < t1:
        return False
    return True

def xyxy2yolo(cls_id, xyxy, img_w, img_h):
    l, t, r, b = xyxy
    bw = r - l
    bh = b - t
    xc = (l + bw / 2) / img_w
    yc = (t + bh / 2) / img_h
    wn = bw / img_w
    hn = bh / img_h
    return f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

def generate_one_image(save_index):
    img = get_random_background()
    draw = ImageDraw.Draw(img)
    IMG_W, IMG_H = img.size
    used_boxes = []
    yolo_lines = []

    group_num = random.randint(GROUP_MIN, GROUP_MAX)
    for _ in range(group_num):
        full_card = random.choice(card_full_list)
        cls_id = card_class_map[full_card]

        current_size = random.randint(TEXT_SIZE_MIN, TEXT_SIZE_MAX)
        vertical_gap = random.randint(LINE_GAP_MIN, LINE_GAP_MAX)
        rand_fp = random.choice(GLOBAL_FONT_POOL)
        card_font = ImageFont.truetype(rand_fp, current_size)

        is_jok = full_card in ("REDJOK", "BLACKJOK")
        # ========== 修复颜色取值逻辑 ==========
        if is_jok:
            _, color_tag = jok_config[full_card]
        else:
            suit_key = full_card[-1]
            _, color_tag = suit_symbol_map[suit_key]
        fill_rgb = (200, 0, 0) if color_tag == "red" else (0, 0, 0)

        if is_jok:
            # ========== 大小王逻辑：仅一段竖排JOK，紧凑行距 ==========
            text_content, _ = jok_config[full_card]
            # 计算文本整体宽高
            bbox_all = draw.textbbox((0, 0), text_content, font=card_font, spacing=JOK_INNER_GAP)
            text_w = bbox_all[2] - bbox_all[0]
            text_h = bbox_all[3] - bbox_all[1]
            max_text_width = text_w
            total_text_h = text_h
        else:
            # ========== 普通扑克牌：数字+花色两行 ==========
            rank_text = full_card[:-1]
            suit_key = full_card[-1]
            suit_char, _ = suit_symbol_map[suit_key]
            # 分别获取两行文字尺寸
            b1 = draw.textbbox((0, 0), rank_text, font=card_font)
            b2 = draw.textbbox((0, 0), suit_char, font=card_font)
            w1 = b1[2] - b1[0]
            w2 = b2[2] - b2[0]
            max_text_width = max(w1, w2)
            line_h = b1[3] - b1[1]
            total_text_h = line_h + vertical_gap + line_h

        # 碰撞区域定位
        find_ok = False
        retry = 0
        while retry < MAX_POS_RETRY:
            x_start = random.randint(20, IMG_W - 130)
            y_start = random.randint(20, IMG_H - total_text_h - 30)
            cl = x_start - PADDING
            ct = y_start - PADDING
            cr = x_start + max_text_width + PADDING + PAD_RIGHT_EXTRA + GROUP_SPACE
            cb = y_start + total_text_h + PADDING + PAD_DOWN_EXTRA + GROUP_SPACE
            cur_rect = (cl, ct, cr, cb)
            conflict = False
            for old in used_boxes:
                if rect_intersect(cur_rect, old):
                    conflict = True
                    break
            if not conflict:
                used_boxes.append(cur_rect)
                find_ok = True
                break
            retry += 1
        if not find_ok:
            continue

        # 绘制灰色底色框
        dl = x_start - PADDING
        dt = y_start
        dr = x_start + max_text_width + PADDING + PAD_RIGHT_EXTRA
        db = y_start + total_text_h + PADDING + PAD_DOWN_EXTRA
        gray = random.randint(240, 252)
        draw.rectangle([dl, dt, dr, db], fill=(gray, gray, gray))

        if is_jok:
            # JOK单行竖排，整体水平居中
            off_x = (max_text_width - text_w) / 2
            draw.text(
                (x_start + off_x, y_start),
                text_content,
                font=card_font,
                fill=fill_rgb,
                spacing=JOK_INNER_GAP
            )
            # 获取完整标注框
            text_bbox = draw.textbbox(
                (x_start + off_x, y_start),
                text_content,
                font=card_font,
                spacing=JOK_INNER_GAP
            )
            ll, tt, rr, bb = text_bbox
        else:
            # 普通牌两行分别居中绘制
            rank_w = b1[2] - b1[0]
            suit_w = b2[2] - b2[0]
            line_h = b1[3] - b1[1]
            # 第一行数字居中
            off_x1 = (max_text_width - rank_w) / 2
            pos1 = (x_start + off_x1, y_start)
            draw.text(pos1, rank_text, font=card_font, fill=fill_rgb)
            # 第二行花色居中
            off_x2 = (max_text_width - suit_w) / 2
            pos2 = (x_start + off_x2, y_start + line_h + vertical_gap)
            draw.text(pos2, suit_char, font=card_font, fill=fill_rgb)
            # 合并两行bbox做标注
            b_rank = draw.textbbox(pos1, rank_text, font=card_font)
            b_suit = draw.textbbox(pos2, suit_char, font=card_font)
            ll = min(b_rank[0], b_suit[0])
            tt = min(b_rank[1], b_suit[1])
            rr = max(b_rank[2], b_suit[2])
            bb = max(b_rank[3], b_suit[3])

        # 生成YOLO标签行
        yolo_line = xyxy2yolo(cls_id, [ll, tt, rr, bb], IMG_W, IMG_H)
        yolo_lines.append(yolo_line)
        # draw.rectangle([ll, tt, rr, bb], outline=(255,0,0), width=2)

    # 保存图片
    img_name = f"gen_{save_index}.png"
    img.save(os.path.join(OUTPUT_IMG_DIR, img_name), compress_level=1)
    # 保存标签
    txt_name = f"gen_{save_index}.txt"
    with open(os.path.join(OUTPUT_LABEL_DIR, txt_name), "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

def main_batch():
    total = GEN_COUNT
    with tqdm(total=total, desc="生成进度") as pbar:
        for idx in range(1, total + 1):
            generate_one_image(idx)
            pbar.update(1)
    print(f"\n✅ {GEN_COUNT} 张图片+标签全部生成完成")

if __name__ == "__main__":
    main_batch()