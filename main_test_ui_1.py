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

delivery_no = ""
quantity = 0
current_index = 0

def now_name(prefix):
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.jpg")

def set_status(text):
    try:
        status_label.config(text=text)

        icon = "🟨"
        if "capture" in text.lower() or "capturing" in text.lower():
            icon = "🟦"
        elif "ocr" in text.lower() or "reading" in text.lower():
            icon = "🟧"
        elif "ready" in text.lower():
            icon = "🟩"

        if capture_state_label is not None:
            capture_state_label.config(text=f"{icon} {text}")

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
    โหลดภาพ ROI แล้วทำ thumbnail สำหรับแสดงบนหน้า Result
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
            label.config(text=f"✔ {name}", fg="#229954")
        elif active == name:
            label.config(text=f"⏳ {name}", fg="#f39c12")
        else:
            label.config(text=f"□ {name}", fg="#17202a")

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
    แสดง ROI + OCR + RAW + source/conf แต่ไม่สูงจนล้นจอ
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

    # รอให้ผู้ใช้กด OK / NEXT ก่อนค่อยไปถ่ายตัวถัดไป
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

    root.after(0, lambda: set_step("CAM3 : YOLO detect → capture → crop ROI → OCR"))
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

    # รอให้ผู้ใช้กด OK / NEXT ก่อนค่อยจบ TOC
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
root.geometry("1000x680")
root.resizable(False, False)
root.configure(bg="#eef3f7")

# TOP BAR
# TOP BAR
top = tk.Frame(root, bg="#ffffff", height=1)
top.pack(side="top", fill="x")
top.pack_propagate(False)

info_delivery = tk.Label(root)
info_en = tk.Label(root)
info_progress = tk.Label(root)
info_pack = tk.Label(root)

# TOP BAR
top = tk.Frame(root, bg="#ffffff", height=1)
top.pack(side="top", fill="x")
top.pack_propagate(False)

info_delivery = tk.Label(root)
info_en = tk.Label(root)
info_progress = tk.Label(root)
info_pack = tk.Label(root)

# PAGE INPUT
page_input = tk.Frame(root, bg="#eef3f7")

input_container = tk.Frame(page_input, bg="#eef3f7")
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
    text="Capture  →  ROI  →  OCR  →  Review",
    font=("Arial", 15),
    bg="#0b2a4a",
    fg="#d6eaf8"
).pack(anchor="w", pady=(24, 8))

tk.Label(
    input_left,
    text="ใช้สำหรับทดสอบ Flow กล้อง, YOLO ROI และ EasyOCR ก่อนรวมเข้าระบบหลัก",
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
    "Result page : ดู ROI และกด OK / NEXT ก่อนทำขั้นตอนถัดไป"
]:
    tk.Label(
        input_left,
        text="✓ " + item,
        font=("Arial", 12, "bold"),
        bg="#0b2a4a",
        fg="white",
        anchor="w"
    ).pack(anchor="w", pady=6)

# Right input card
input_card = tk.Frame(input_container, bg="#ffffff", padx=34, pady=32)
input_card.pack(side="right", fill="y", padx=(28, 0))
input_card.pack_propagate(False)
input_card.config(width=380)

tk.Label(
    input_card,
    text="Start Test Job",
    font=("Arial", 22, "bold"),
    bg="#ffffff",
    fg="#0b2a4a"
).pack(anchor="w", pady=(14, 4))

tk.Label(
    input_card,
    text="กรอกข้อมูลเพื่อเริ่มทดสอบ",
    font=("Arial", 11),
    bg="#ffffff",
    fg="#7f8c8d"
).pack(anchor="w", pady=(0, 22))

tk.Label(
    input_card,
    text="Delivery No",
    bg="#ffffff",
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
    bg="#ffffff",
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
    bg="#0b5cab",
    fg="white",
    activebackground="#084b8a",
    activeforeground="white",
    width=22,
    command=start_job
).pack(fill="x", ipady=6, pady=(4, 12))

tk.Label(
    input_card,
    text="Quantity = จำนวนรอบที่ CAM1/CAM2 จะถ่ายทดสอบ",
    bg="#ffffff",
    fg="#7f8c8d",
    font=("Arial", 9),
    wraplength=300,
    justify="left"
).pack(anchor="w", pady=(8, 0))

# PAGE CAPTURE
page_capture = tk.Frame(root, bg="#f4f6f7")

capture_container = tk.Frame(page_capture, bg="#f4f6f7")
capture_container.pack(fill="both", expand=True, padx=10, pady=10)

# =====================
# LOT INFORMATION
# =====================
lot_frame = tk.Frame(
    capture_container,
    bg="#ffffff",
    highlightbackground="#ffd84d",
    highlightthickness=3,
    padx=14,
    pady=10
)
lot_frame.pack(fill="x", pady=(0, 10))

lot_top = tk.Frame(lot_frame, bg="#ffffff")
lot_top.pack(fill="x")

tk.Label(
    lot_top,
    text="Lot Information",
    bg="#ffffff",
    fg="#17202a",
    font=("Arial", 12, "bold")
).pack(side="left")

tk.Button(
    lot_top,
    text="Cancel Job",
    bg="#ff3333",
    fg="white",
    activebackground="#cc0000",
    activeforeground="white",
    font=("Arial", 10, "bold"),
    width=14,
    command=reset_to_input
).pack(side="right")

lot_info_row = tk.Frame(lot_frame, bg="#ffffff")
lot_info_row.pack(fill="x", pady=(12, 2))

def lot_info_box(parent, title, width=18):
    box = tk.Frame(parent, bg="#ffffff")
    box.pack(side="left", padx=(0, 36))

    tk.Label(
        box,
        text=title,
        bg="#ffffff",
        fg="#17202a",
        font=("Arial", 9, "bold")
    ).pack(anchor="w")

    value = tk.Label(
        box,
        text="-",
        bg="#ffffff",
        fg="#000000",
        font=("Arial", 11),
        width=width,
        anchor="w"
    )
    value.pack(anchor="w")

    return value

info_delivery = lot_info_box(lot_info_row, "Delivery No", 16)
info_en = lot_info_box(lot_info_row, "EN", 16)
info_progress = lot_info_box(lot_info_row, "Progress", 12)
info_pack = lot_info_box(lot_info_row, "Pack Type", 16)

# =====================
# BODY AREA
# =====================
body_frame = tk.Frame(capture_container, bg="#f4f6f7")
body_frame.pack(fill="both", expand=True)

# LEFT : CAPTURE IMAGE
capture_left = tk.Frame(
    body_frame,
    bg="#ffffff",
    highlightbackground="#ffd84d",
    highlightthickness=3,
    padx=14,
    pady=10
)
capture_left.pack(side="left", fill="both", expand=True, padx=(0, 10))

tk.Label(
    capture_left,
    text="Capture Images",
    bg="#ffffff",
    fg="#17202a",
    font=("Arial", 12, "bold")
).pack(anchor="w")

tk.Label(
    capture_left,
    text="Preview",
    bg="#ffffff",
    fg="#17202a",
    font=("Arial", 9)
).pack(anchor="w", pady=(6, 4))

preview_frame = tk.Frame(
    capture_left,
    bg="#17202a",
    width=640,
    height=360,
    highlightbackground="#ff3333",
    highlightthickness=5
)
preview_frame.pack(pady=(4, 10))
preview_frame.pack_propagate(False)

preview_label = tk.Label(
    preview_frame,
    text="Camera Preview",
    bg="#17202a",
    fg="white",
    font=("Arial", 16)
)
preview_label.place(x=0, y=0, relwidth=1, relheight=1)

capture_state_label = tk.Label(
    preview_frame,
    text="🟨 Waiting Object",
    bg="#17202a",
    fg="white",
    font=("Arial", 11, "bold"),
    padx=8,
    pady=4
)
capture_state_label.place(x=8, y=8)

tk.Button(
    capture_left,
    text="Reset Background",
    bg="#bfc1c5",
    fg="#000000",
    activebackground="#a9abb0",
    font=("Arial", 10),
    height=2
).pack(fill="x", pady=(4, 0))

# RIGHT : STATUS
capture_right = tk.Frame(
    body_frame,
    bg="#ffffff",
    highlightbackground="#ffd84d",
    highlightthickness=3,
    padx=18,
    pady=12,
    width=360
)
capture_right.pack(side="right", fill="y")
capture_right.pack_propagate(False)

tk.Label(
    capture_right,
    text="Status",
    bg="#ffffff",
    fg="#17202a",
    font=("Arial", 12, "bold")
).pack(anchor="w")

tk.Label(
    capture_right,
    text="Product",
    bg="#ffffff",
    fg="#17202a",
    font=("Arial", 10, "bold")
).pack(anchor="w", pady=(18, 2))

product_row = tk.Frame(capture_right, bg="#ffffff")
product_row.pack(fill="x")

progress_label = tk.Label(
    product_row,
    text="Product 0 / 0",
    bg="#ffffff",
    fg="#17202a",
    font=("Arial", 10)
)
progress_label.pack(side="left")

progress_percent_label = tk.Label(
    product_row,
    text="0%",
    bg="#ffffff",
    fg="#17202a",
    font=("Arial", 10)
)
progress_percent_label.pack(side="right")

progress_bar_bg = tk.Frame(
    capture_right,
    bg="#c4c4c8",
    height=24
)
progress_bar_bg.pack(fill="x", pady=(4, 20))
progress_bar_bg.pack_propagate(False)

progress_bar_fill = tk.Frame(
    progress_bar_bg,
    bg="#0b5cab"
)
progress_bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

tk.Label(
    capture_right,
    text="Checklist",
    bg="#ffffff",
    fg="#17202a",
    font=("Arial", 10, "bold")
).pack(anchor="w", pady=(4, 8))

checklist_frame = tk.Frame(capture_right, bg="#ffffff")
checklist_frame.pack(fill="x")

def make_check_item(name, text):
    item = tk.Label(
        checklist_frame,
        text=text,
        bg="#ffffff",
        fg="#17202a",
        font=("Arial", 12, "bold"),
        anchor="w",
        padx=14,
        pady=8,
        relief="solid",
        borderwidth=3,
        width=14
    )
    item.pack(anchor="w", pady=6)
    check_labels[name] = item

make_check_item("TOA", "□ TOA")
make_check_item("TOB", "□ TOB")
make_check_item("TOC", "□ TOC")

step_status_box = tk.Frame(capture_right, bg="#f4f6f7", padx=10, pady=8)
step_status_box.pack(fill="x", pady=(16, 10))

tk.Label(
    step_status_box,
    text="Step",
    bg="#f4f6f7",
    fg="#34495e",
    font=("Arial", 9, "bold")
).pack(anchor="w")

step_label = tk.Label(
    step_status_box,
    text="-",
    bg="#f4f6f7",
    fg="#17202a",
    font=("Arial", 10),
    wraplength=280,
    justify="left"
)
step_label.pack(anchor="w")

tk.Label(
    step_status_box,
    text="Status",
    bg="#f4f6f7",
    fg="#34495e",
    font=("Arial", 9, "bold")
).pack(anchor="w", pady=(8, 0))

status_label = tk.Label(
    step_status_box,
    text="-",
    bg="#f4f6f7",
    fg="#17202a",
    font=("Arial", 10),
    wraplength=280,
    justify="left"
)
status_label.pack(anchor="w")

tk.Button(
    capture_right,
    text="STOP",
    bg="#c0392b",
    fg="white",
    activebackground="#922b21",
    activeforeground="white",
    font=("Arial", 12, "bold"),
    height=2,
    command=stop_job
).pack(side="bottom", fill="x", pady=(10, 0))

# PAGE RESULT
page_result = tk.Frame(root, bg="#eef3f7")

result_header = tk.Frame(page_result, bg="#eef3f7")
result_header.pack(fill="x", padx=20, pady=(12, 6))

result_title = tk.Label(
    result_header,
    text="OCR Result",
    font=("Arial", 18, "bold"),
    bg="#eef3f7",
    fg="#0b2a4a"
)
result_title.pack(side="left")

result_status_badge = tk.Label(
    result_header,
    text="WAIT REVIEW",
    bg="#f1c40f",
    fg="#17202a",
    font=("Arial", 11, "bold"),
    padx=12,
    pady=5
)
result_status_badge.pack(side="left", padx=14)

result_button_frame = tk.Frame(result_header, bg="#eef3f7")
result_button_frame.pack(side="right")

tk.Button(
    result_button_frame,
    text="OK / NEXT",
    bg="#27ae60",
    fg="white",
    font=("Arial", 12, "bold"),
    width=13,
    command=on_result_ok
).pack(side="right", padx=5)

tk.Button(
    result_button_frame,
    text="STOP",
    bg="#c0392b",
    fg="white",
    font=("Arial", 12, "bold"),
    width=11,
    command=stop_job
).pack(side="right", padx=5)

result_hint = tk.Label(
    page_result,
    text="ตรวจสอบภาพ ROI และค่าที่ OCR อ่านได้ แล้วกด OK / NEXT เพื่อไปขั้นตอนถัดไป",
    bg="#eef3f7",
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

# READY TOC
page_ready_toc = tk.Frame(root, bg="#eef3f7")

toc_card = tk.Frame(page_ready_toc, bg="#ffffff", padx=35, pady=35)
toc_card.place(relx=0.5, rely=0.45, anchor="center", width=540, height=280)

tk.Label(
    toc_card,
    text="CAM1 / CAM2 Test Completed",
    font=("Arial", 21, "bold"),
    bg="#ffffff",
    fg="#229954"
).pack(pady=15)

tk.Label(
    toc_card,
    text="Press continue to test CAM3 TOC YOLO ROI OCR",
    font=("Arial", 13),
    bg="#ffffff",
    fg="#34495e"
).pack(pady=8)

tk.Button(
    toc_card,
    text="CONTINUE TO TOC",
    bg="#0b5cab",
    fg="white",
    font=("Arial", 14, "bold"),
    width=20,
    command=continue_toc
).pack(pady=20)

# COMPLETE
page_complete = tk.Frame(root, bg="#eef3f7")

complete_card = tk.Frame(page_complete, bg="#ffffff", padx=35, pady=35)
complete_card.place(relx=0.5, rely=0.45, anchor="center", width=520, height=260)

tk.Label(
    complete_card,
    text="TEST COMPLETED",
    font=("Arial", 24, "bold"),
    bg="#ffffff",
    fg="#229954"
).pack(pady=25)

tk.Button(
    complete_card,
    text="OK",
    bg="#0b5cab",
    fg="white",
    font=("Arial", 14, "bold"),
    width=16,
    command=reset_to_input
).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_close)
show_page(page_input)
root.mainloop()
