import os
import json
import threading
from threading import Event
from datetime import datetime
from time import sleep

import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from config_test import (
    CAPTURE_DIR,
    ROI_DIR,
    OCR_DIR,
    PICAM1_INDEX,
    PICAM2_INDEX,
    USB_DEVICE,
    PREVIEW_SIZE,
    PREVIEW_INTERVAL
)

from camera_test import PiCameraTest, UsbCameraTest
from roi_test import crop_center, crop_by_box, preprocess_ocr, save_roi_files
from ocr_test import run_easyocr, load_ocr

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)

running = False
last_preview_time = 0
current_camera = None

result_ok_event = Event()
check_labels = {}
capture_state_label = None
capture_state_dot = None

delivery_no = ""
quantity = 0
current_index = 0

def now_name(prefix):
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.jpg")

def set_status(text):
    """
    Update small status text in the right panel and the camera preview overlay.
    Uses a Canvas dot instead of emoji, so it works the same on Raspberry Pi.
    """
    try:
        status_label.config(text=text)

        color = "#f1c40f"  # yellow = waiting / default
        lower = text.lower()

        if "capture" in lower or "capturing" in lower or "opening" in lower:
            color = "#3498db"  # blue
        elif "ocr" in lower or "reading" in lower:
            color = "#e67e22"  # orange
        elif "ready" in lower:
            color = "#27ae60"  # green
        elif "stop" in lower or "error" in lower:
            color = "#c0392b"  # red

        if capture_state_label is not None:
            capture_state_label.config(text=text)

        if capture_state_dot is not None:
            capture_state_dot.delete("all")
            capture_state_dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

        root.update_idletasks()
    except Exception:
        pass

    print(text)

def set_step(text):
    try:
        step_label.config(text=text)
        root.update_idletasks()
    except Exception:
        pass
    print("STEP:", text)

def update_preview(frame, is_bgr=False):
    global last_preview_time

    import time
    now = time.time()

    if now - last_preview_time < PREVIEW_INTERVAL:
        return

    last_preview_time = now

    try:
        display = cv2.resize(frame, PREVIEW_SIZE)

        if is_bgr:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=img)

        preview_label.imgtk = imgtk
        preview_label.config(image=imgtk, text="")

    except Exception as e:
        print("Preview error:", e)


def make_thumbnail_image(image_path, size=(120, 60)):
    """
    เนเธซเธฅเธ”เธ เธฒเธ ROI เนเธฅเนเธงเธ—เธณ thumbnail เธชเธณเธซเธฃเธฑเธเนเธชเธ”เธเธเธเธซเธเนเธฒ Result
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return None

        img = cv2.imread(image_path)

        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)

        pil_img = Image.fromarray(img)
        return ImageTk.PhotoImage(image=pil_img)

    except Exception as e:
        print("Thumbnail error:", e)
        return None

def show_page(page):
    for p in [page_input, page_capture, page_result, page_ready_toc, page_complete]:
        p.pack_forget()
    page.pack(fill="both", expand=True)

def update_info():
    info_delivery.config(text=delivery_no or "-")
    info_en.config(text="-")
    info_progress.config(text=f"{current_index} / {quantity}")
    info_pack.config(text="-")

def set_progress(text):
    progress_label.config(text=text)

    try:
        if quantity > 0:
            percent = int((current_index / quantity) * 100)
        else:
            percent = 0

        progress_percent_label.config(text=f"{percent}%")
        progress_bar_fill.place(relx=0, rely=0, relwidth=percent / 100, relheight=1)
    except Exception:
        pass

    root.update_idletasks()

def set_checklist(active=None, done=None):
    for name, label in check_labels.items():
        if done and name in done:
            label.config(text=f"[OK]  {name}", fg="#229954", bg="#eafaf1")
        elif active == name:
            label.config(text=f"[RUN] {name}", fg="#b9770e", bg="#fff3cd")
        else:
            label.config(text=f"[ ]   {name}", fg=TEXT, bg=SOFT)

def capture_path(prefix):
    return os.path.join(CAPTURE_DIR, now_name(prefix))

def process_center_roi_ocr(image_path, prefix):
    img = cv2.imread(image_path)

    if img is None:
        return {
            "image_path": image_path,
            "error": "cannot_read_image",
            "raw": "",
            "clean": ""
        }

    crop, box = crop_center(img)
    processed = preprocess_ocr(crop)
    roi_path, processed_path = save_roi_files(image_path, crop, processed, prefix)

    ocr = run_easyocr(processed)

    return {
        "name": prefix,
        "source": "center_roi",
        "image_path": image_path,
        "roi_path": roi_path,
        "processed_path": processed_path,
        "box": box,
        "raw": ocr["raw"],
        "clean": ocr["clean"],
        "items": ocr["items"]
    }

def process_yolo_roi_ocr(image_path, detections):
    img = cv2.imread(image_path)

    if img is None:
        return []

    results = []

    for det in detections:
        name = det["name"]
        box = det["box"]

        crop, fixed_box = crop_by_box(img, box, pad=0)

        if crop is None or crop.size == 0:
            continue

        processed = preprocess_ocr(crop)

        prefix = f"toc_{name}"
        roi_path, processed_path = save_roi_files(image_path, crop, processed, prefix)

        ocr = run_easyocr(processed)

        results.append({
            "name": name,
            "source": "yolo_roi",
            "detect_conf": det["conf"],
            "image_path": image_path,
            "roi_path": roi_path,
            "processed_path": processed_path,
            "box": fixed_box,
            "raw": ocr["raw"],
            "clean": ocr["clean"],
            "items": ocr["items"]
        })

    return results

def add_result_row(title, data):
    """
    V5 compact result row:
    เนเธชเธ”เธ ROI + OCR + RAW + source/conf เนเธ•เนเนเธกเนเธชเธนเธเธเธเธฅเนเธเธเธญ
    """
    row = tk.Frame(
        result_list_frame,
        bg="#ffffff",
        padx=6,
        pady=5,
        relief="solid",
        borderwidth=1
    )
    row.pack(fill="x", padx=8, pady=4)

    # ROI image
    roi_path = data.get("roi_path", "")
    thumb = make_thumbnail_image(roi_path, size=(120, 60))

    img_box = tk.Label(
        row,
        bg="#17202a",
        width=120,
        height=60
    )
    img_box.pack(side="left", padx=(0, 8))

    if thumb is not None:
        img_box.imgtk = thumb
        img_box.config(image=thumb, text="")
    else:
        img_box.config(
            text="NO ROI",
            fg="white",
            font=("Arial", 8, "bold")
        )

    # Class / source
    info_box = tk.Frame(row, bg="#ffffff", width=115)
    info_box.pack(side="left", fill="y", padx=(0, 8))
    info_box.pack_propagate(False)

    tk.Label(
        info_box,
        text=title,
        bg="#ffffff",
        fg="#0b2a4a",
        font=("Arial", 11, "bold"),
        anchor="w"
    ).pack(anchor="w", pady=(2, 0))

    source = data.get("source", "-")
    detect_conf = data.get("detect_conf", None)

    if detect_conf is not None:
        source_text = f"{source} / DET {detect_conf:.2f}"
    else:
        source_text = source

    tk.Label(
        info_box,
        text=source_text,
        bg="#ffffff",
        fg="#7f8c8d",
        font=("Arial", 8),
        anchor="w"
    ).pack(anchor="w")

    # OCR clean
    clean_text_value = data.get("clean", "") or "(empty)"

    tk.Label(
        row,
        text=clean_text_value,
        bg="#eafaf1",
        fg="#145a32",
        font=("Arial", 16, "bold"),
        anchor="w",
        padx=10
    ).pack(side="left", fill="x", expand=True, padx=(0, 8))

    # Raw text
    raw_text = data.get("raw", "")

    tk.Label(
        row,
        text=f"RAW: {raw_text}",
        bg="#ffffff",
        fg="#34495e",
        font=("Arial", 9),
        anchor="w",
        justify="left",
        wraplength=220,
        width=30
    ).pack(side="left", padx=(0, 8))

    # Status
    is_read = clean_text_value != "(empty)"
    tk.Label(
        row,
        text="READ" if is_read else "EMPTY",
        bg="#27ae60" if is_read else "#e67e22",
        fg="white",
        font=("Arial", 10, "bold"),
        width=8,
        padx=4,
        pady=5
    ).pack(side="right")


def clear_result_list():
    for w in result_list_frame.winfo_children():
        w.destroy()

def show_product_result(product_no, cam1_data, cam2_data):
    show_page(page_result)
    result_title.config(text=f"OCR Result : Product {product_no}")
    result_status_badge.config(text="WAIT REVIEW", bg="#f1c40f", fg="#17202a")
    clear_result_list()

    add_result_row("CAM1 Center ROI", cam1_data)
    add_result_row("CAM2 Center ROI", cam2_data)

    save_json({
        "type": "product",
        "product_no": product_no,
        "cam1": cam1_data,
        "cam2": cam2_data
    }, f"product_{product_no:03d}.json")

def show_toc_result(toc_items, fallback_data=None):
    show_page(page_result)
    result_title.config(text="OCR Result : TOC YOLO ROI")
    result_status_badge.config(text="WAIT REVIEW", bg="#f1c40f", fg="#17202a")
    clear_result_list()

    if toc_items:
        for item in toc_items:
            add_result_row(item["name"], item)
    elif fallback_data is not None:
        add_result_row("CAM3 Center ROI", fallback_data)
    else:
        add_result_row("NO ROI", {"clean": "", "raw": "No detection"})

    save_json({
        "type": "toc",
        "toc_items": toc_items,
        "fallback": fallback_data
    }, "toc_result.json")

def save_json(data, filename):
    path = os.path.join(OCR_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved JSON:", path)

def start_job():
    global running, delivery_no, quantity, current_index

    delivery_no = delivery_entry.get().strip()
    qty_text = qty_entry.get().strip()

    if not delivery_no:
        messagebox.showwarning("Warning", "Please input Delivery No")
        return

    if not qty_text.isdigit():
        messagebox.showwarning("Warning", "Please input Quantity")
        return

    quantity = int(qty_text)

    if quantity < 1 or quantity > 100:
        messagebox.showwarning("Warning", "Quantity must be 1-100")
        return

    current_index = 0
    running = True

    update_info()
    show_page(page_capture)

    set_checklist(active="TOA", done=[])
    set_status("Waiting Object")
    set_progress(f"Product 0 / {quantity}")

    thread = threading.Thread(target=job_loop, daemon=True)
    thread.start()

def job_loop():
    global current_index, running

    set_status("Loading OCR Model...")
    load_ocr()
    set_status("OCR Ready")

    for i in range(quantity):
        if not running:
            return

        current_index = i + 1
        root.after(0, update_info)
        root.after(0, lambda i=i: set_progress(f"Product {i+1} / {quantity}"))

        ok = process_product(i + 1)

        if not ok:
            running = False
            return

        sleep(1.0)

    running = False
    root.after(0, lambda: show_page(page_ready_toc))

def process_product(product_no):
    global current_camera

    # =====================
    # CAM1
    # =====================
    root.after(0, lambda: show_page(page_capture))
    root.after(0, lambda: set_checklist(active="TOA", done=[]))
    root.after(0, lambda: set_step("CAM1 : Detect object by background"))
    root.after(0, lambda: set_status("CAM1 opening"))

    cam1_path = capture_path(f"cam1_product{product_no}")

    cam1 = PiCameraTest(
        index=PICAM1_INDEX,
        name="CAM1",
        status_cb=set_status,
        preview_cb=update_preview
    )
    current_camera = cam1

    try:
        cam1.open()
        bg = cam1.wait_background(lambda: running)

        if bg is None:
            return False

        detected = cam1.wait_object(bg, lambda: running)

        if not detected:
            return False

        cam1.focus_delay(lambda: running)
        cam1.capture_file(cam1_path)

    except Exception as e:
        root.after(0, lambda: messagebox.showerror("CAM1 Error", str(e)))
        return False

    finally:
        cam1.close()
        current_camera = None

    # =====================
    # CAM2
    # =====================
    if not running:
        return False

    root.after(0, lambda: set_checklist(active="TOB", done=["TOA"]))
    root.after(0, lambda: set_step("CAM2 : Capture directly"))
    root.after(0, lambda: set_status("CAM2 opening"))

    cam2_path = capture_path(f"cam2_product{product_no}")

    cam2 = PiCameraTest(
        index=PICAM2_INDEX,
        name="CAM2",
        status_cb=set_status,
        preview_cb=update_preview
    )
    current_camera = cam2

    try:
        cam2.open()
        cam2.focus_delay(lambda: running)
        cam2.capture_file(cam2_path)

    except Exception as e:
        root.after(0, lambda: messagebox.showerror("CAM2 Error", str(e)))
        return False

    finally:
        cam2.close()
        current_camera = None

    # =====================
    # OCR
    # =====================
    if not running:
        return False

    root.after(0, lambda: set_step("OCR : Center ROI CAM1/CAM2"))
    root.after(0, lambda: set_status("OCR reading"))

    cam1_data = process_center_roi_ocr(cam1_path, "cam1")
    cam2_data = process_center_roi_ocr(cam2_path, "cam2")

    root.after(0, lambda: set_checklist(active=None, done=["TOA", "TOB"]))
    root.after(0, lambda: show_product_result(product_no, cam1_data, cam2_data))

    # เธฃเธญเนเธซเนเธเธนเนเนเธเนเธเธ” OK / NEXT เธเนเธญเธเธเนเธญเธขเนเธเธ–เนเธฒเธขเธ•เธฑเธงเธ–เธฑเธ”เนเธ
    wait_result_ok()

    return running

def continue_toc():
    global running

    running = True
    show_page(page_capture)
    set_checklist(active="TOC", done=["TOA", "TOB"])
    update_info()

    thread = threading.Thread(target=toc_loop, daemon=True)
    thread.start()

def toc_loop():
    global running

    set_progress("TOC 1 / 1")

    ok = process_toc()

    running = False

    if ok:
        root.after(0, lambda: show_page(page_complete))

def process_toc():
    global current_camera

    root.after(0, lambda: set_step("CAM3 : YOLO detect โ’ capture โ’ crop ROI โ’ OCR"))
    root.after(0, lambda: set_status("CAM3 opening"))

    toc_path = capture_path("cam3_toc")

    cam3 = UsbCameraTest(
        device=USB_DEVICE,
        name="CAM3",
        status_cb=set_status,
        preview_cb=update_preview
    )

    current_camera = cam3

    try:
        cam3.open()
        capture_result = cam3.capture_direct_or_yolo(toc_path, lambda: running)

        if not capture_result:
            return False

    except Exception as e:
        root.after(0, lambda: messagebox.showerror("CAM3 Error", str(e)))
        return False

    finally:
        cam3.close()
        current_camera = None

    image_path = capture_result["image_path"]
    detections = capture_result.get("detections", [])

    root.after(0, lambda: set_step(f"TOC OCR : YOLO ROI count = {len(detections)}"))
    root.after(0, lambda: set_status("TOC OCR reading from YOLO ROI"))

    toc_items = process_yolo_roi_ocr(image_path, detections)

    fallback = None

    if not toc_items:
        root.after(0, lambda: set_status("No YOLO ROI OCR, fallback center ROI"))
        fallback = process_center_roi_ocr(image_path, "toc_center")

    root.after(0, lambda: set_checklist(active=None, done=["TOA", "TOB", "TOC"]))
    root.after(0, lambda: show_toc_result(toc_items, fallback))

    # เธฃเธญเนเธซเนเธเธนเนเนเธเนเธเธ” OK / NEXT เธเนเธญเธเธเนเธญเธขเธเธ TOC
    wait_result_ok()

    return running

def stop_job():
    global running, current_camera

    running = False
    result_ok_event.set()

    try:
        if current_camera is not None:
            current_camera.close()
    except Exception:
        pass

    set_status("Stop Requested")

def on_result_ok():
    result_ok_event.set()


def wait_result_ok():
    result_ok_event.clear()

    while running and not result_ok_event.is_set():
        sleep(0.1)

def reset_to_input():
    global running, current_index

    running = False
    current_index = 0

    delivery_entry.delete(0, tk.END)
    qty_entry.delete(0, tk.END)

    info_delivery.config(text="-")
    info_en.config(text="-")
    info_progress.config(text="-")
    info_pack.config(text="-")

    show_page(page_input)

def on_close():
    stop_job()
    root.destroy()

# =========================================================
# UI
# =========================================================
root = tk.Tk()
root.title("AI Camera TEST UI - YOLO ROI OCR")
root.geometry("1200x720")
root.resizable(False, False)

# =====================
# THEME
# =====================
BG = "#eef3f7"
CARD = "#ffffff"
TEXT = "#17202a"
MUTED = "#6c7a89"
BLUE = "#0b5cab"
BLUE_DARK = "#084b8a"
RED = "#c0392b"
RED_DARK = "#922b21"
BORDER = "#dfe6e9"
SOFT = "#f7f9fa"
SOFT_2 = "#f2f5f7"
GREEN = "#229954"
GREEN_BG = "#eafaf1"
YELLOW_TEXT = "#b9770e"
YELLOW_BG = "#fff3cd"
ORANGE = "#e67e22"
DARK = "#17202a"

root.configure(bg=BG)

# =====================
# ICONS
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(BASE_DIR, "assets", "icons")
icons = {}

def load_icon_file(filename, size=(24, 24)):
    """
    Load PNG icon from assets/icons.
    If icon is missing, return None so UI can still run.
    """
    path = os.path.join(ICON_DIR, filename)
    if not os.path.exists(path):
        print("Icon not found:", path)
        return None

    try:
        img = Image.open(path).convert("RGBA")
        try:
            img = img.resize(size, Image.Resampling.LANCZOS)
        except Exception:
            img = img.resize(size)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print("Icon load error:", path, e)
        return None

def load_icons():
    # Your current downloaded filenames
    icons["lot"] = load_icon_file("assignment.png")
    icons["delivery"] = load_icon_file("description.png")
    icons["barcode"] = load_icon_file("label.png")
    icons["progress"] = load_icon_file("analytics.png")
    icons["pack"] = load_icon_file("package.png")
    icons["camera"] = load_icon_file("photo.png")
    icons["status"] = load_icon_file("dashboard.png")
    icons["check"] = load_icon_file("checklist.png")
    icons["setting"] = load_icon_file("settings.png")
    icons["info"] = load_icon_file("info.png")
    icons["cancel"] = load_icon_file("cancel.png")
    icons["stop"] = load_icon_file("stop.png")

    # Optional icon. If you later rename refresh icon to refresh.png, it will be used.
    icons["refresh"] = load_icon_file("refresh.png")

def icon_label(parent, key, bg=CARD, size=None):
    icon = icons.get(key)
    if icon is not None:
        return tk.Label(parent, image=icon, bg=bg)

    # fallback when icon file missing
    fallback = key.upper()[:4]
    return tk.Label(
        parent,
        text=fallback,
        bg=SOFT,
        fg=BLUE,
        font=("Arial", 8, "bold"),
        width=5,
        height=2
    )

def title_with_icon(parent, icon_key, text, bg=CARD):
    frame = tk.Frame(parent, bg=bg)
    frame.pack(fill="x")

    icon_label(frame, icon_key, bg=bg).pack(side="left", padx=(0, 10))

    tk.Label(
        frame,
        text=text,
        bg=bg,
        fg=TEXT,
        font=("Arial", 14, "bold")
    ).pack(side="left")

    return frame

def flat_button(parent, text, bg, fg="white", command=None, icon_key=None, width=None):
    kwargs = {
        "text": text,
        "bg": bg,
        "fg": fg,
        "activebackground": RED_DARK if bg == RED else BLUE_DARK,
        "activeforeground": "white",
        "font": ("Arial", 10, "bold"),
        "relief": "flat",
        "command": command,
        "cursor": "hand2",
        "padx": 10,
        "pady": 5
    }

    if width is not None:
        kwargs["width"] = width

    icon = icons.get(icon_key) if icon_key else None
    if icon is not None:
        kwargs["image"] = icon
        kwargs["compound"] = "left"

    return tk.Button(parent, **kwargs)

load_icons()

# Hidden one-pixel top bar, kept only for page packing consistency
top = tk.Frame(root, bg=BG, height=1)
top.pack(side="top", fill="x")
top.pack_propagate(False)

# Placeholder labels before real capture-page labels are created
info_delivery = tk.Label(root)
info_en = tk.Label(root)
info_progress = tk.Label(root)
info_pack = tk.Label(root)

# =========================================================
# PAGE INPUT
# =========================================================
page_input = tk.Frame(root, bg=BG)

input_container = tk.Frame(page_input, bg=BG)
input_container.pack(fill="both", expand=True, padx=42, pady=34)

# Left information panel
input_left = tk.Frame(input_container, bg="#0b2a4a", padx=34, pady=30)
input_left.pack(side="left", fill="both", expand=True)

tk.Label(
    input_left,
    text="AI Camera Validation",
    font=("Arial", 26, "bold"),
    bg="#0b2a4a",
    fg="white"
).pack(anchor="w", pady=(22, 8))

tk.Label(
    input_left,
    text="TEST MODE",
    font=("Arial", 14, "bold"),
    bg="#0b2a4a",
    fg="#f1c40f"
).pack(anchor="w")

tk.Label(
    input_left,
    text="Capture  ->  ROI  ->  OCR  ->  Review",
    font=("Arial", 15),
    bg="#0b2a4a",
    fg="#d6eaf8"
).pack(anchor="w", pady=(24, 8))

tk.Label(
    input_left,
    text="Use this screen to test camera flow, YOLO ROI and EasyOCR before integrating with the main system.",
    font=("Arial", 12),
    bg="#0b2a4a",
    fg="#d6eaf8",
    wraplength=430,
    justify="left"
).pack(anchor="w", pady=(4, 20))

for item in [
    "CAM1 : Background detect + center ROI OCR",
    "CAM2 : Direct capture + center ROI OCR",
    "CAM3 : YOLO detect + crop ROI OCR",
    "Result page : Review ROI and press OK / NEXT"
]:
    tk.Label(
        input_left,
        text="[OK] " + item,
        font=("Arial", 12, "bold"),
        bg="#0b2a4a",
        fg="white",
        anchor="w"
    ).pack(anchor="w", pady=6)

# Right input card
input_card = tk.Frame(input_container, bg=CARD, padx=34, pady=32, highlightbackground=BORDER, highlightthickness=1)
input_card.pack(side="right", fill="y", padx=(28, 0))
input_card.pack_propagate(False)
input_card.config(width=380)

tk.Label(
    input_card,
    text="Start Test Job",
    font=("Arial", 22, "bold"),
    bg=CARD,
    fg=TEXT
).pack(anchor="w", pady=(14, 4))

tk.Label(
    input_card,
    text="Input data to start capture test.",
    font=("Arial", 11),
    bg=CARD,
    fg=MUTED
).pack(anchor="w", pady=(0, 22))

tk.Label(
    input_card,
    text="Delivery No",
    bg=CARD,
    fg="#34495e",
    font=("Arial", 11, "bold")
).pack(anchor="w")

delivery_entry = tk.Entry(
    input_card,
    font=("Arial", 18),
    justify="center",
    width=22,
    relief="solid",
    borderwidth=1
)
delivery_entry.pack(fill="x", pady=(6, 18), ipady=5)

tk.Label(
    input_card,
    text="Quantity",
    bg=CARD,
    fg="#34495e",
    font=("Arial", 11, "bold")
).pack(anchor="w")

qty_entry = tk.Entry(
    input_card,
    font=("Arial", 18),
    justify="center",
    width=10,
    relief="solid",
    borderwidth=1
)
qty_entry.pack(anchor="w", pady=(6, 24), ipady=5)

tk.Button(
    input_card,
    text="START TEST",
    font=("Arial", 14, "bold"),
    bg=BLUE,
    fg="white",
    activebackground=BLUE_DARK,
    activeforeground="white",
    width=22,
    relief="flat",
    command=start_job
).pack(fill="x", ipady=6, pady=(4, 12))

tk.Label(
    input_card,
    text="Quantity = number of products for CAM1 / CAM2 capture test.",
    bg=CARD,
    fg=MUTED,
    font=("Arial", 9),
    wraplength=300,
    justify="left"
).pack(anchor="w", pady=(8, 0))

# =========================================================
# PAGE CAPTURE
# =========================================================
page_capture = tk.Frame(root, bg=BG)

capture_container = tk.Frame(page_capture, bg=BG)
capture_container.pack(fill="both", expand=True, padx=14, pady=14)

# =====================
# LOT INFORMATION
# =====================
lot_frame = tk.Frame(
    capture_container,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    padx=18,
    pady=12
)
lot_frame.pack(fill="x", pady=(0, 10))

lot_top = tk.Frame(lot_frame, bg=CARD)
lot_top.pack(fill="x")

lot_title = tk.Frame(lot_top, bg=CARD)
lot_title.pack(side="left")

icon_label(lot_title, "lot", bg=CARD).pack(side="left", padx=(0, 10))

tk.Label(
    lot_title,
    text="Lot Information",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 14, "bold")
).pack(side="left")

flat_button(
    lot_top,
    text="  Cancel Job",
    bg=RED,
    fg="white",
    command=reset_to_input,
    icon_key="cancel",
    width=130
).pack(side="right")

lot_info_row = tk.Frame(lot_frame, bg=CARD)
lot_info_row.pack(fill="x", pady=(14, 4))

def lot_info_box(parent, icon_key, title, width=14):
    box = tk.Frame(parent, bg=CARD)
    box.pack(side="left", padx=(0, 46))

    icon_label(box, icon_key, bg=CARD).pack(side="left", padx=(0, 10))

    text_box = tk.Frame(box, bg=CARD)
    text_box.pack(side="left")

    tk.Label(
        text_box,
        text=title,
        bg=CARD,
        fg=MUTED,
        font=("Arial", 9, "bold")
    ).pack(anchor="w")

    value = tk.Label(
        text_box,
        text="-",
        bg=CARD,
        fg=TEXT,
        font=("Arial", 14, "bold"),
        width=width,
        anchor="w"
    )
    value.pack(anchor="w", pady=(2, 0))

    return value

info_delivery = lot_info_box(lot_info_row, "delivery", "Delivery No", 14)
info_en = lot_info_box(lot_info_row, "barcode", "EN", 10)
info_progress = lot_info_box(lot_info_row, "progress", "Progress", 10)
info_pack = lot_info_box(lot_info_row, "pack", "Pack Type", 12)

# =====================
# BODY AREA
# =====================
body_frame = tk.Frame(capture_container, bg=BG)
body_frame.pack(fill="both", expand=True)

# LEFT : CAPTURE IMAGE
capture_left = tk.Frame(
    body_frame,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    padx=18,
    pady=14
)
capture_left.pack(side="left", fill="both", expand=True, padx=(0, 10))

title_with_icon(capture_left, "camera", "Capture Images", bg=CARD)

tk.Label(
    capture_left,
    text="Preview",
    bg=CARD,
    fg=MUTED,
    font=("Arial", 9, "bold")
).pack(anchor="w", pady=(8, 4))

preview_frame = tk.Frame(
    capture_left,
    bg=DARK,
    width=760,
    height=430,
    highlightbackground=DARK,
    highlightthickness=2
)
preview_frame.pack(pady=(4, 10))
preview_frame.pack_propagate(False)

preview_label = tk.Label(
    preview_frame,
    text="Camera Preview",
    bg=DARK,
    fg="white",
    font=("Arial", 16)
)
preview_label.place(x=0, y=0, relwidth=1, relheight=1)

# Preview overlay
preview_overlay = tk.Frame(preview_frame, bg=DARK)
preview_overlay.place(x=10, y=8)

capture_state_dot = tk.Canvas(
    preview_overlay,
    width=14,
    height=14,
    bg=DARK,
    highlightthickness=0
)
capture_state_dot.pack(side="left", padx=(0, 5))
capture_state_dot.create_oval(2, 2, 12, 12, fill="#f1c40f", outline="#f1c40f")

capture_state_label = tk.Label(
    preview_overlay,
    text="Waiting Object",
    bg=DARK,
    fg="white",
    font=("Arial", 11, "bold"),
    padx=2,
    pady=2
)
capture_state_label.pack(side="left")

reset_btn = tk.Button(
    capture_left,
    text="  Reset Background",
    bg="#dfe6e9",
    fg=TEXT,
    activebackground="#cfd8dc",
    activeforeground=TEXT,
    font=("Arial", 10, "bold"),
    relief="flat",
    height=2,
    cursor="hand2"
)

if icons.get("refresh") is not None:
    reset_btn.config(image=icons["refresh"], compound="left")

reset_btn.pack(fill="x", pady=(8, 0))

# RIGHT : STATUS
capture_right = tk.Frame(
    body_frame,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    padx=18,
    pady=14,
    width=330
)
capture_right.pack(side="right", fill="y")
capture_right.pack_propagate(False)

title_with_icon(capture_right, "status", "Status", bg=CARD)

# Product block
product_header = tk.Frame(capture_right, bg=CARD)
product_header.pack(fill="x", pady=(18, 2))

icon_label(product_header, "pack", bg=CARD).pack(side="left", padx=(0, 8))

tk.Label(
    product_header,
    text="Product",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 10, "bold")
).pack(side="left")

product_row = tk.Frame(capture_right, bg=CARD)
product_row.pack(fill="x")

progress_label = tk.Label(
    product_row,
    text="Product 0 / 0",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 10)
)
progress_label.pack(side="left")

progress_percent_label = tk.Label(
    product_row,
    text="0%",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 10)
)
progress_percent_label.pack(side="right")

progress_bar_bg = tk.Frame(
    capture_right,
    bg="#c4c4c8",
    height=18
)
progress_bar_bg.pack(fill="x", pady=(4, 20))
progress_bar_bg.pack_propagate(False)

progress_bar_fill = tk.Frame(
    progress_bar_bg,
    bg=BLUE
)
progress_bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

# Checklist block
check_header = tk.Frame(capture_right, bg=CARD)
check_header.pack(fill="x", pady=(4, 8))

icon_label(check_header, "check", bg=CARD).pack(side="left", padx=(0, 8))

tk.Label(
    check_header,
    text="Checklist",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 10, "bold")
).pack(side="left")

checklist_frame = tk.Frame(capture_right, bg=CARD)
checklist_frame.pack(fill="x")

def make_check_item(name, text):
    item = tk.Label(
        checklist_frame,
        text=text,
        bg=SOFT,
        fg=TEXT,
        font=("Arial", 12, "bold"),
        anchor="w",
        padx=14,
        pady=9,
        relief="flat",
        borderwidth=0
    )
    item.pack(fill="x", pady=5)
    check_labels[name] = item

make_check_item("TOA", "[ ]   TOA")
make_check_item("TOB", "[ ]   TOB")
make_check_item("TOC", "[ ]   TOC")

# Step / Status detail block
step_status_box = tk.Frame(capture_right, bg=SOFT, padx=12, pady=10)
step_status_box.pack(fill="x", pady=(16, 10))

step_header = tk.Frame(step_status_box, bg=SOFT)
step_header.pack(fill="x")

icon_label(step_header, "setting", bg=SOFT).pack(side="left", padx=(0, 8))

tk.Label(
    step_header,
    text="Step",
    bg=SOFT,
    fg="#34495e",
    font=("Arial", 9, "bold")
).pack(side="left")

step_label = tk.Label(
    step_status_box,
    text="-",
    bg=SOFT,
    fg=TEXT,
    font=("Arial", 10),
    wraplength=280,
    justify="left"
)
step_label.pack(anchor="w", pady=(2, 8))

status_detail_header = tk.Frame(step_status_box, bg=SOFT)
status_detail_header.pack(fill="x")

icon_label(status_detail_header, "info", bg=SOFT).pack(side="left", padx=(0, 8))

tk.Label(
    status_detail_header,
    text="Status",
    bg=SOFT,
    fg="#34495e",
    font=("Arial", 9, "bold")
).pack(side="left")

status_label = tk.Label(
    step_status_box,
    text="-",
    bg=SOFT,
    fg=TEXT,
    font=("Arial", 10),
    wraplength=280,
    justify="left"
)
status_label.pack(anchor="w", pady=(2, 0))

tk.Button(
    capture_right,
    text="  STOP",
    bg=RED,
    fg="white",
    activebackground=RED_DARK,
    activeforeground="white",
    font=("Arial", 13, "bold"),
    height=2,
    relief="flat",
    cursor="hand2",
    image=icons["stop"] if icons.get("stop") is not None else "",
    compound="left",
    command=stop_job
).pack(side="bottom", fill="x", pady=(10, 0))

# =========================================================
# PAGE RESULT
# =========================================================
page_result = tk.Frame(root, bg=BG)

result_header = tk.Frame(page_result, bg=BG)
result_header.pack(fill="x", padx=20, pady=(12, 6))

result_title = tk.Label(
    result_header,
    text="OCR Result",
    font=("Arial", 18, "bold"),
    bg=BG,
    fg=TEXT
)
result_title.pack(side="left")

result_status_badge = tk.Label(
    result_header,
    text="WAIT REVIEW",
    bg="#f1c40f",
    fg=TEXT,
    font=("Arial", 11, "bold"),
    padx=12,
    pady=5
)
result_status_badge.pack(side="left", padx=14)

result_button_frame = tk.Frame(result_header, bg=BG)
result_button_frame.pack(side="right")

tk.Button(
    result_button_frame,
    text="OK / NEXT",
    bg="#27ae60",
    fg="white",
    font=("Arial", 12, "bold"),
    width=13,
    relief="flat",
    command=on_result_ok
).pack(side="right", padx=5)

tk.Button(
    result_button_frame,
    text="STOP",
    bg=RED,
    fg="white",
    font=("Arial", 12, "bold"),
    width=11,
    relief="flat",
    command=stop_job
).pack(side="right", padx=5)

result_hint = tk.Label(
    page_result,
    text="Review ROI image and OCR result, then press OK / NEXT to continue.",
    bg=BG,
    fg="#34495e",
    font=("Arial", 10)
)
result_hint.pack(anchor="w", padx=22, pady=(0, 4))

result_table_header = tk.Frame(page_result, bg="#0b2a4a", padx=8, pady=5)
result_table_header.pack(fill="x", padx=20)

for txt, width in [
    ("ROI", 16),
    ("Class / Source", 16),
    ("OCR Clean", 42),
    ("RAW", 30),
    ("Status", 8),
]:
    tk.Label(
        result_table_header,
        text=txt,
        bg="#0b2a4a",
        fg="white",
        font=("Arial", 10, "bold"),
        width=width,
        anchor="w"
    ).pack(side="left", padx=2)

result_list_frame = tk.Frame(page_result, bg="#f7f9fa", padx=6, pady=6)
result_list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))

# =========================================================
# READY TOC
# =========================================================
page_ready_toc = tk.Frame(root, bg=BG)

toc_card = tk.Frame(page_ready_toc, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
toc_card.place(relx=0.5, rely=0.45, anchor="center", width=540, height=280)

tk.Label(
    toc_card,
    text="CAM1 / CAM2 Test Completed",
    font=("Arial", 21, "bold"),
    bg=CARD,
    fg="#229954"
).pack(pady=15)

tk.Label(
    toc_card,
    text="Press continue to test CAM3 TOC YOLO ROI OCR",
    font=("Arial", 13),
    bg=CARD,
    fg="#34495e"
).pack(pady=8)

tk.Button(
    toc_card,
    text="CONTINUE TO TOC",
    bg=BLUE,
    fg="white",
    font=("Arial", 14, "bold"),
    width=20,
    relief="flat",
    command=continue_toc
).pack(pady=20)

# =========================================================
# COMPLETE
# =========================================================
page_complete = tk.Frame(root, bg=BG)

complete_card = tk.Frame(page_complete, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
complete_card.place(relx=0.5, rely=0.45, anchor="center", width=520, height=260)

tk.Label(
    complete_card,
    text="TEST COMPLETED",
    font=("Arial", 24, "bold"),
    bg=CARD,
    fg="#229954"
).pack(pady=25)

tk.Button(
    complete_card,
    text="OK",
    bg=BLUE,
    fg="white",
    font=("Arial", 14, "bold"),
    width=16,
    relief="flat",
    command=reset_to_input
).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_close)
show_page(page_input)
root.mainloop()
