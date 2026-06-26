from ultralytics import YOLO
import cv2

# 完整权重路径
weight_path = r'E:\Python-Code\recognize-playing-cards\train\runs\detect\train-2\weights\best.pt'
model_raw = YOLO(weight_path)

while True:
    inputs = input("请输入图片路径：")
    if inputs == 'q':
        break
    try:
        file_path = r"E:\Python-Code\creat_font\test-image\\" + inputs + ".png"
        # 推理
        results = model_raw(file_path, conf=0.6)
        res = results[0]
        # 仅使用当前版本支持的参数，缩小文字+细框
        img = res.plot(
            line_width=1,    # 细边框，减少遮挡
            font_size=7      # 文字缩小，数值越小字体越小，推荐6~9
        )
        # OpenCV弹窗展示
        cv2.imshow("poker detect", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"处理失败：{e}")