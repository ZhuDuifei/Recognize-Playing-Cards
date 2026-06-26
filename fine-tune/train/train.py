from ultralytics import YOLO
import torch


def train_yolo():
    model = YOLO(r"E:\Python-Code\recognize-playing-cards\fine-tune\train\best.pt")

    train_device = 0 if torch.cuda.is_available() else "cpu"
    print(f"训练使用设备: {train_device}")

    # 开始训练
    results = model.train(
        data=r'train.yaml',
        epochs=15,
        batch=19,
        device=train_device,
        imgsz=(680, 460),
        workers=1,
        save=True,
        # 全部数据增强置0/关闭
        mosaic=0,  # 关闭马赛克增强（最核心）
        mixup=0,  # 关闭mixup混合增强
        copy_paste=0,  # 关闭复制粘贴增强
        hsv_h=0,  # HSV色相扰动
        hsv_s=0,  # HSV饱和度扰动
        hsv_v=0,  # HSV亮度扰动
        degrees=0,  # 旋转角度
        perspective=0,  # 透视变换
        flipud=0,  # 上下翻转
        fliplr=0,  # 左右翻转
        shear=0,  # 剪切变换
        scale=1,  # 缩放（1=不缩放）
        translate=0,  # 平移
    )

    return results

if __name__ == "__main__":
    train_yolo()