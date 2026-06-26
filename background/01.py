from PIL import Image
import os


def resize_image_fixed_height(input_path, output_path, target_height=329):
    # 打开图片
    img = Image.open(input_path)
    original_w, original_h = img.size

    # 计算等比例宽度
    scale = target_height / original_h
    target_width = int(original_w * scale)

    # 高质量缩放
    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # 保存
    resized_img.save(output_path)
    print(f"缩放完成：原图({original_w}×{original_h}) → 新图({target_width}×{target_height})")


# 单张图片使用示例
if __name__ == "__main__":
    # 替换为你的图片路径
    input_img = "img_20.png"
    output_img = "img20.png"
    resize_image_fixed_height(input_img, output_img, target_height=400)