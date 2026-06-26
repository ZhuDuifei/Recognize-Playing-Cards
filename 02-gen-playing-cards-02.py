import os
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ===================== 可自定义配置区 =====================
TEXT_SIZE_MIN = 12
TEXT_SIZE_MAX = 38
# JOK字体缩放系数，比普通牌小，防止文字出界
JOK_FONT_SCALE = 0.72
# JOK竖排内部紧凑行距
JOK_LINE_GAP = 2
LINE_GAP_MIN = 0
LINE_GAP_MAX = 6
# 单叠卡牌数量范围
STACK_CARD_MIN = 1
STACK_CARD_MAX = 15
# 整张图最多几叠独立牌组
STACK_GROUP_MIN = 1
STACK_GROUP_MAX = 6
# 偏移缩放系数：0~1，1=几乎完全错开，0.6=适中紧凑
OFFSET_SCALE = 0.8
# 卡牌内边距、白色外边框宽度
PADDING = 6
WHITE_BORDER_WIDTH = 3
PAD_RIGHT_EXTRA = 8
PAD_DOWN_EXTRA = 10

GEN_COUNT = 10000
BG_FOLDER = "./background"
OUTPUT_IMG_DIR = "./images/train"
OUTPUT_LABEL_DIR = "./labels/train"
FONT_TXT_PATH = "selected_fonts.txt"
MAX_POS_RETRY = 120
# ==========================================================

# 新增大小王到牌库
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
# 大小王配置：竖排文字、颜色
jok_config = {
    "REDJOK": ("J\nO\nK", "red"),
    "BLACKJOK": ("J\nO\nK", "black")
}

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 读取筛选后的字体池
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
        raise RuntimeError("selected_fonts.txt 无有效字体")
    return font_paths

GLOBAL_FONT_POOL = load_font_pool_from_txt()

# 背景缓存
BG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
if not os.path.exists(BG_FOLDER):
    os.makedirs(BG_FOLDER)
    raise FileNotFoundError(f"请放入背景图到 {BG_FOLDER}")
BG_FILE_LIST = [f for f in os.listdir(BG_FOLDER) if f.lower().endswith(BG_EXTS)]
if not BG_FILE_LIST:
    raise FileNotFoundError(f"{BG_FOLDER} 文件夹为空")
BG_CACHE_POOL = []
for name in BG_FILE_LIST:
    path = os.path.join(BG_FOLDER, name)
    img = Image.open(path).convert("RGB")
    BG_CACHE_POOL.append(img)

def get_random_background():
    return random.choice(BG_CACHE_POOL).copy()

# 矩形相交判断
def rect_intersect(r1, r2):
    l1, t1, r1x, b1 = r1
    l2, t2, r2x, b2 = r2
    if r1x < l2 or r2x < l1:
        return False
    if b1 < t2 or b2 < t1:
        return False
    return True

# xyxy转YOLO归一化坐标
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
    stack_global_boxes = []
    yolo_lines = []
    # 所有堆叠统一底部Y坐标，保证同一水平面
    BASE_BOTTOM_Y = int(IMG_H * 0.72)

    stack_total = random.randint(STACK_GROUP_MIN, STACK_GROUP_MAX)
    for _ in range(stack_total):
        card_count_in_stack = random.randint(STACK_CARD_MIN, STACK_CARD_MAX)
        # 单堆叠统一字体参数
        current_size = random.randint(TEXT_SIZE_MIN, TEXT_SIZE_MAX)
        vertical_gap = random.randint(LINE_GAP_MIN, LINE_GAP_MAX)
        rand_fp = random.choice(GLOBAL_FONT_POOL)
        stack_font = ImageFont.truetype(rand_fp, current_size)
        single_text_h = current_size + vertical_gap + current_size

        # 文字区域总宽度（字体+左右内边距，代表卡片有效可视宽度）
        text_area_w = current_size * 1.2 + PADDING * 2 + PAD_RIGHT_EXTRA
        text_area_h = single_text_h + PADDING * 2 + PAD_DOWN_EXTRA

        # 偏移量基于完整文字卡片宽度计算
        CARD_OFFSET_X = int(text_area_w * OFFSET_SCALE)

        # 卡牌整体尺寸（包含白色边框）
        card_total_w = text_area_w + WHITE_BORDER_WIDTH * 2
        card_total_h = text_area_h + WHITE_BORDER_WIDTH * 2

        # 整堆叠整体尺寸
        stack_total_w = card_total_w + CARD_OFFSET_X * (card_count_in_stack - 1)
        stack_total_h = card_total_h

        # ========== 修复1：提前判断宽度是否超出画面，超了直接跳过这一叠 ==========
        usable_width = IMG_W - 60
        if stack_total_w >= usable_width:
            continue

        # 寻找不重叠的水平位置，底部固定在BASE_BOTTOM_Y
        find_stack_pos = False
        retry = 0
        stack_base_x = 0
        while retry < MAX_POS_RETRY:
            # 修复2：上限兜底，保证start < stop
            x_min = 30
            x_max = int(IMG_W - stack_total_w - 30)
            if x_max <= x_min:
                break
            stack_base_x = random.randint(x_min, x_max)

            stack_top_y = BASE_BOTTOM_Y - card_total_h
            cl = stack_base_x - 10
            ct = stack_top_y - 10
            cr = stack_base_x + stack_total_w + 10
            cb = BASE_BOTTOM_Y + 10
            stack_rect = (cl, ct, cr, cb)
            conflict = False
            for old_rect in stack_global_boxes:
                if rect_intersect(stack_rect, old_rect):
                    conflict = True
                    break
            if not conflict:
                stack_global_boxes.append(stack_rect)
                find_stack_pos = True
                break
            retry += 1
        if not find_stack_pos:
            continue

        # 逐张绘制，偏移量随卡片宽度自适应
        for card_idx in range(card_count_in_stack):
            off_x = card_idx * CARD_OFFSET_X
            # 卡牌整体左上角（包含白边）
            card_all_x = stack_base_x + off_x
            card_all_y = stack_top_y

            # 1. 绘制外层白色边框
            draw.rectangle(
                [card_all_x, card_all_y, card_all_x + card_total_w, card_all_y + card_total_h],
                fill="white"
            )
            # 2. 绘制内部浅灰底色卡片区域
            inner_x = card_all_x + WHITE_BORDER_WIDTH
            inner_y = card_all_y + WHITE_BORDER_WIDTH
            inner_r = inner_x + text_area_w
            inner_b = inner_y + text_area_h
            gray = random.randint(240, 252)
            draw.rectangle([inner_x, inner_y, inner_r, inner_b], fill=(gray, gray, gray))

            # 文字绘制起点（内边距）
            text_x = inner_x + PADDING
            text_y = inner_y + PADDING

            full_card = random.choice(card_full_list)
            cls_id = card_class_map[full_card]
            is_jok = full_card in ("REDJOK", "BLACKJOK")

            if is_jok:
                # ========== 大小王逻辑：字体缩小、竖排居中，不溢出 ==========
                jok_text, color_tag = jok_config[full_card]
                # JOK单独缩小字号
                jok_font_size = int(current_size * JOK_FONT_SCALE)
                jok_font = ImageFont.truetype(rand_fp, jok_font_size)
                fill_rgb = (200, 0, 0) if color_tag == "red" else (0, 0, 0)

                # 计算竖排文字整体宽高
                text_bbox = draw.textbbox((0, 0), jok_text, font=jok_font, spacing=JOK_LINE_GAP)
                tw = text_bbox[2] - text_bbox[0]
                # 水平居中偏移
                x_offset = 0
                draw.text(
                    (text_x + x_offset, text_y),
                    jok_text,
                    font=jok_font,
                    fill=fill_rgb,
                    spacing=JOK_LINE_GAP
                )
                # YOLO标注框
                full_bbox = draw.textbbox(
                    (text_x + x_offset, text_y),
                    jok_text,
                    font=jok_font,
                    spacing=JOK_LINE_GAP
                )
                ll, tt, rr, bb = full_bbox
            else:
                # ========== 普通牌逻辑：数字&花色水平居中对齐，修复10错位 ==========
                rank_text = full_card[:-1]
                suit_key = full_card[-1]
                suit_char, color_tag = suit_symbol_map[suit_key]
                fill_rgb = (200, 0, 0) if color_tag == "red" else (0, 0, 0)

                # 分别计算两行文字宽度
                b_rank = draw.textbbox((0, 0), rank_text, font=stack_font)
                b_suit = draw.textbbox((0, 0), suit_char, font=stack_font)
                w_rank = b_rank[2] - b_rank[0]
                w_suit = b_suit[2] - b_suit[0]
                line_h = b_rank[3] - b_rank[1]
                max_w = max(w_rank, w_suit)

                # 数字居中绘制
                off1 = (max_w - w_rank) / 2
                pos1_x = text_x + off1
                pos1_y = text_y
                draw.text((pos1_x, pos1_y), rank_text, font=stack_font, fill=fill_rgb)

                # 花色居中绘制
                off2 = (max_w - w_suit) / 2
                pos2_x = text_x + off2
                pos2_y = text_y + line_h + vertical_gap
                draw.text((pos2_x, pos2_y), suit_char, font=stack_font, fill=fill_rgb)

                # 合并两行文字边界做YOLO标注
                box1 = draw.textbbox((pos1_x, pos1_y), rank_text, font=stack_font)
                box2 = draw.textbbox((pos2_x, pos2_y), suit_char, font=stack_font)
                ll = min(box1[0], box2[0])
                tt = min(box1[1], box2[1])
                rr = max(box1[2], box2[2])
                bb = max(box1[3], box2[3])

            yolo_lines.append(xyxy2yolo(cls_id, [ll, tt, rr, bb], IMG_W, IMG_H))
            # 绘制红色校验框
            # draw.rectangle([ll, tt, rr, bb], outline=(255, 0, 0), width=2)


    # 保存图片
    img_save_path = os.path.join(OUTPUT_IMG_DIR, f"playing_card_{save_index}.png")
    img.save(img_save_path, compress_level=1)
    # 保存标签
    txt_save_path = os.path.join(OUTPUT_LABEL_DIR, f"playing_card_{save_index}.txt")
    with open(txt_save_path, "w", encoding="utf-8") as f:
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