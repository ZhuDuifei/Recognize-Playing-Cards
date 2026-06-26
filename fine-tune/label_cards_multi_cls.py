import os
import json
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# 配置常量
CARD_FOLDER = "cards"
BOX_JSON = "box_info.json"
CLASS_JSON = "class_mapping.json"

# 全局数据缓存
class CardAnnotatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("卡牌标注工具 - 一体化GUI")
        self.root.geometry("1200x750")

        # 数据存储
        self.class_mapping = {}
        self.box_info = {}
        self.card_list = []
        self.curr_card_idx = 0
        self.curr_img = None
        self.pil_img = None
        self.tk_img = None
        self.curr_boxes = []  # [x1,y1,x2,y2,cls_name,cid]
        self.selected_box_idx = None
        self.drag_corner = None  # 拖拽角：tl/tr/bl/br / move 移动
        self.drag_origin = (0, 0)

        # 鼠标画框临时变量
        self.drawing = False
        self.draw_start = (0, 0)

        # 绑定键盘退格删除
        self.root.bind("<BackSpace>", self.on_backspace)

        # 加载文件与卡牌列表
        self.load_mapping_file()
        self.load_box_file()
        self.load_card_list()

        # ========== 新增：补齐所有图片到box_info ==========
        self.fill_all_cards_in_boxinfo()

        # 构建界面布局
        self.build_widgets()
        if self.card_list:
            self.load_current_card()

    def load_mapping_file(self):
        if os.path.exists(CLASS_JSON):
            with open(CLASS_JSON, "r", encoding="utf-8") as f:
                self.class_mapping = json.load(f)

    def save_mapping_file(self):
        with open(CLASS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.class_mapping, f, indent=2, ensure_ascii=False)

    def load_box_file(self):
        if os.path.exists(BOX_JSON):
            with open(BOX_JSON, "r", encoding="utf-8") as f:
                self.box_info = json.load(f)

    def save_box_file(self):
        with open(BOX_JSON, "w", encoding="utf-8") as f:
            json.dump(self.box_info, f, indent=2, ensure_ascii=False)

    def load_card_list(self):
        if not os.path.exists(CARD_FOLDER):
            os.makedirs(CARD_FOLDER)
            self.card_list = []
            return
        self.card_list = sorted([
            f for f in os.listdir(CARD_FOLDER)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        self.curr_card_idx = 0

    # ========== 新增函数：自动补齐所有cards图片进入box_info ==========
    def fill_all_cards_in_boxinfo(self):
        for img_name in self.card_list:
            if img_name not in self.box_info:
                self.box_info[img_name] = []
        self.save_box_file()

    def build_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=4)
        self.canvas = tk.Canvas(left_frame, bg="#222222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        right_frame = ttk.Frame(main_pane, width=260)
        main_pane.add(right_frame, weight=1)

        nav_frame = ttk.LabelFrame(right_frame, text="图片导航")
        nav_frame.pack(fill=tk.X, pady=6, padx=4)
        ttk.Button(nav_frame, text="上一张", command=self.prev_card).grid(row=0, column=0, padx=2, pady=3, sticky=tk.EW)
        ttk.Button(nav_frame, text="下一张", command=self.next_card).grid(row=0, column=1, padx=2, pady=3, sticky=tk.EW)
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)
        self.card_label = ttk.Label(nav_frame, text="当前：无图片")
        self.card_label.grid(row=1, column=0, columnspan=2, pady=4)

        cls_frame = ttk.LabelFrame(right_frame, text="类别管理")
        cls_frame.pack(fill=tk.X, pady=6, padx=4)
        self.cls_var = tk.StringVar()
        self.cls_combo = ttk.Combobox(cls_frame, textvariable=self.cls_var, state="readonly")
        self.cls_combo.pack(fill=tk.X, pady=3)
        ttk.Button(cls_frame, text="新建类别", command=self.create_new_class).pack(fill=tk.X, pady=2)
        ttk.Button(cls_frame, text="修改选中框类别", command=self.change_box_class).pack(fill=tk.X, pady=2)

        box_frame = ttk.LabelFrame(right_frame, text="标注框操作")
        box_frame.pack(fill=tk.X, pady=6, padx=4)
        ttk.Button(box_frame, text="删除选中框", command=self.delete_selected_box).pack(fill=tk.X, pady=2)
        ttk.Button(box_frame, text="保存当前标注", command=self.save_current_boxes).pack(fill=tk.X, pady=2)
        ttk.Button(box_frame, text="刷新画布", command=self.redraw_canvas).pack(fill=tk.X, pady=2)

        tip_label = ttk.Label(right_frame, text="操作提示：\n拖拽绘制新框，松开弹窗选类别\n选中框拖动四角缩放、拖动内部移动\n选中框按Backspace快速删除", foreground="#555")
        tip_label.pack(pady=12)

        self.refresh_class_combo()

    def refresh_class_combo(self):
        cls_names = list(self.class_mapping.keys())
        self.cls_combo["values"] = cls_names
        if cls_names:
            self.cls_combo.current(0)

    def create_new_class(self):
        new_name = tk.simpledialog.askstring("新建类别", "输入类别名称：")
        if not new_name or new_name.strip() == "":
            return None
        new_name = new_name.strip()
        if new_name in self.class_mapping:
            messagebox.showwarning("提示", "该类别已存在！")
            return new_name
        new_id = len(self.class_mapping)
        self.class_mapping[new_name] = new_id
        self.save_mapping_file()
        self.refresh_class_combo()
        self.cls_var.set(new_name)
        messagebox.showinfo("成功", f"新增类别 {new_name} ID={new_id}")
        return new_name

    def popup_select_class(self):
        win = tk.Toplevel(self.root)
        win.title("选择当前框类别")
        win.geometry("340x180")
        win.attributes("-topmost", True)
        win.grab_set()

        res_data = {"cls": None}
        var = tk.StringVar()
        cls_list = list(self.class_mapping.keys())

        cb = ttk.Combobox(win, textvariable=var, values=cls_list, state="readonly", width=30)
        if cls_list:
            cb.current(0)
        cb.pack(pady=15)

        def confirm():
            res_data["cls"] = var.get()
            win.destroy()

        def new_cls():
            new_n = self.create_new_class()
            if new_n:
                res_data["cls"] = new_n
            win.destroy()

        frame = tk.Frame(win)
        frame.pack()
        ttk.Button(frame, text="确认选择", command=confirm).grid(row=0, column=0, padx=5)
        ttk.Button(frame, text="新建类别", command=new_cls).grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="取消本次框", command=lambda: win.destroy()).grid(row=0, column=2, padx=5)

        self.root.wait_window(win)
        return res_data["cls"]

    def load_current_card(self):
        if not self.card_list:
            self.curr_img = None
            self.canvas.delete(tk.ALL)
            self.curr_boxes = []
            self.card_label.config(text="当前：无卡牌文件")
            return

        card_name = self.card_list[self.curr_card_idx]
        self.card_label.config(text=f"当前：{card_name} ({self.curr_card_idx+1}/{len(self.card_list)})")
        img_path = os.path.join(CARD_FOLDER, card_name)
        cv_img = cv2.imread(img_path)
        if cv_img is None:
            messagebox.showerror("错误", f"图片损坏：{card_name}")
            self.curr_img = None
            self.curr_boxes = []
            return

        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(cv_img_rgb)
        self.curr_img = cv_img
        self.curr_boxes = self.box_info.get(card_name, [])
        self.selected_box_idx = None
        self.drag_corner = None
        self.redraw_canvas()

    def redraw_canvas(self):
        self.canvas.delete(tk.ALL)
        if self.curr_img is None or self.pil_img is None:
            return
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        for idx, box in enumerate(self.curr_boxes):
            x1, y1, x2, y2, cls_name, cid = box
            label_text = f"{cls_name}"
            if idx == self.selected_box_idx:
                color = "#ff3333"
                width = 3
                # 绘制四角拖拽控制点
                r = 5
                self.canvas.create_oval(x1-r, y1-r, x1+r, y1+r, fill=color)
                self.canvas.create_oval(x2-r, y1-r, x2+r, y1+r, fill=color)
                self.canvas.create_oval(x1-r, y2-r, x1+r, y2+r, fill=color)
                self.canvas.create_oval(x2-r, y2-r, x2+r, y2+r, fill=color)
            else:
                color = "#33ff33"
                width = 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)

            text_x = x1
            text_y = y1
            self.canvas.create_text(text_x + 1, text_y + 1, anchor=tk.NW, fill="#000000", font=("Microsoft YaHei", 9), text=label_text)
            self.canvas.create_text(text_x, text_y, anchor=tk.NW, fill=color, font=("Microsoft YaHei", 9), text=label_text)

    # 键盘退格删除选中框
    def on_backspace(self, event):
        if self.selected_box_idx is not None:
            self.delete_selected_box()

    # 判断鼠标点落在哪个角 / 框内部
    def get_drag_target(self, mx, my, box):
        x1, y1, x2, y2, _, _ = box
        r = 6
        # 四角
        if abs(mx - x1) < r and abs(my - y1) < r:
            return "tl"
        if abs(mx - x2) < r and abs(my - y1) < r:
            return "tr"
        if abs(mx - x1) < r and abs(my - y2) < r:
            return "bl"
        if abs(mx - x2) < r and abs(my - y2) < r:
            return "br"
        # 框内部移动
        if x1 <= mx <= x2 and y1 <= my <= y2:
            return "move"
        return None

    def mouse_down(self, event):
        mx, my = event.x, event.y
        self.drag_corner = None
        self.drawing = True
        self.draw_start = (mx, my)
        self.selected_box_idx = None

        # 遍历框判断是否点中控制点/框
        for idx, box in enumerate(self.curr_boxes):
            target = self.get_drag_target(mx, my, box)
            if target is not None:
                self.selected_box_idx = idx
                self.drag_corner = target
                self.drag_origin = (mx, my)
                break
        self.redraw_canvas()

    def mouse_move(self, event):
        mx, my = event.x, event.y
        # 拖拽调整框大小/移动框
        if self.selected_box_idx is not None and self.drag_corner is not None:
            dx = mx - self.drag_origin[0]
            dy = my - self.drag_origin[1]
            box = self.curr_boxes[self.selected_box_idx]
            x1, y1, x2, y2, cls, cid = box
            if self.drag_corner == "move":
                nx1 = x1 + dx
                ny1 = y1 + dy
                nx2 = x2 + dx
                ny2 = y2 + dy
            elif self.drag_corner == "tl":
                nx1 = x1 + dx
                ny1 = y1 + dy
                nx2 = x2
                ny2 = y2
            elif self.drag_corner == "tr":
                nx1 = x1
                ny1 = y1 + dy
                nx2 = x2 + dx
                ny2 = y2
            elif self.drag_corner == "bl":
                nx1 = x1 + dx
                ny1 = y1
                nx2 = x2
                ny2 = y2 + dy
            elif self.drag_corner == "br":
                nx1 = x1
                ny1 = y1
                nx2 = x2 + dx
                ny2 = y2 + dy
            # 限制宽高最小
            if abs(nx2 - nx1) > 10 and abs(ny2 - ny1) > 10:
                self.curr_boxes[self.selected_box_idx] = [nx1, ny1, nx2, ny2, cls, cid]
            self.drag_origin = (mx, my)
            self.redraw_canvas()
            return
        # 新建框拖拽虚线
        if not self.drawing:
            return
        self.redraw_canvas()
        sx, sy = self.draw_start
        self.canvas.create_rectangle(sx, sy, mx, my, outline="#ffff00", width=2, dash=(4,2))

    def mouse_up(self, event):
        # 拖拽结束保存
        if self.selected_box_idx is not None and self.drag_corner is not None:
            self.save_current_boxes()
            self.drag_corner = None
            self.redraw_canvas()
            return
        # 新建标注框逻辑
        if not self.drawing:
            return
        self.drawing = False
        sx, sy = self.draw_start
        ex, ey = event.x, event.y
        x1 = min(sx, ex)
        y1 = min(sy, ey)
        x2 = max(sx, ex)
        y2 = max(sy, ey)
        w = x2 - x1
        h = y2 - y1
        if w < 10 or h < 10:
            self.redraw_canvas()
            return

        cls_name = self.popup_select_class()
        if not cls_name:
            self.redraw_canvas()
            return
        cid = self.class_mapping[cls_name]
        new_box = [x1, y1, x2, y2, cls_name, cid]
        self.curr_boxes.append(new_box)
        self.save_current_boxes()
        self.redraw_canvas()
        # messagebox.showinfo("新增框", f"添加标注 {cls_name}")

    def prev_card(self):
        if self.curr_card_idx > 0:
            self.curr_card_idx -= 1
            self.load_current_card()

    def next_card(self):
        if self.curr_card_idx < len(self.card_list) - 1:
            self.curr_card_idx += 1
            self.load_current_card()

    def change_box_class(self):
        if self.selected_box_idx is None:
            messagebox.showinfo("提示", "请先点击选中一个标注框！")
            return
        new_cls = self.cls_var.get()
        if new_cls not in self.class_mapping:
            messagebox.showwarning("错误", "类别不存在")
            return
        cid = self.class_mapping[new_cls]
        box = self.curr_boxes[self.selected_box_idx]
        box[4] = new_cls
        box[5] = cid
        self.save_current_boxes()
        self.redraw_canvas()
        messagebox.showinfo("完成", "已修改框类别，画布已刷新文字")

    def delete_selected_box(self):
        if self.selected_box_idx is None:
            messagebox.showinfo("提示", "请先选中框")
            return
        del self.curr_boxes[self.selected_box_idx]
        self.selected_box_idx = None
        self.drag_corner = None
        self.save_current_boxes()
        self.redraw_canvas()

    def save_current_boxes(self):
        if not self.card_list:
            return
        card_name = self.card_list[self.curr_card_idx]
        self.box_info[card_name] = self.curr_boxes
        self.save_box_file()

if __name__ == "__main__":
    import tkinter.simpledialog as simpledialog
    root = tk.Tk()
    app = CardAnnotatorGUI(root)
    root.mainloop()