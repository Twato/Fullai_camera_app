import os
import math
import time
import threading
import glob
import subprocess
from datetime import datetime

import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from config_test import (
    PICAM1_INDEX,
    PICAM2_INDEX,
    USB_DEVICE,
    PREVIEW_SIZE,
    PREVIEW_INTERVAL,
)

# ใช้ของเดิม เพื่อให้ CAM3 YOLO ทำงานเหมือนโปรแกรม T8
from camera_test import UsbCameraTest
from roi_test import crop_by_box

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

# =========================================================
# CONFIG
# =========================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MANUAL_SAVE_ROOT = os.path.join(APP_DIR, "manual_capture")
os.makedirs(MANUAL_SAVE_ROOT, exist_ok=True)

FOCUS_DELAY_SEC = 1.2
USB_SCAN_DEVICES = [USB_DEVICE, 1, 0, 2, 3, 4, 5, 16, 17, 18]

CAM_WIDTH = 1280
CAM_HEIGHT = 720
PREVIEW_W, PREVIEW_H = PREVIEW_SIZE

# =========================================================
# STATE
# =========================================================
running = False
quantity = 0
outer_total = 0
inner_index = 1
outer_index = 1
current_stage = "input"   # input, cam1, cam2, cam3, complete
current_camera = None
current_job_dir = ""
last_preview_time = 0

capture_lock = threading.Lock()
preview_stop = threading.Event()

# =========================================================
# COLORS / UI THEME ใกล้เคียง T8 เดิม
# =========================================================
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
GREEN = "#229954"
GREEN_DARK = "#1e8449"
GREEN_BG = "#eafaf1"
YELLOW = "#f1c40f"
DARK = "#17202a"
GRAY = "#bfc3c7"

# =========================================================
# HELPERS
# =========================================================
def now_name(prefix, ext="jpg"):
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.{ext}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def safe_close_camera():
    global current_camera
    try:
        if current_camera is not None:
            if hasattr(current_camera, "close"):
                current_camera.close()
            elif hasattr(current_camera, "stop"):
                current_camera.stop()
    except Exception as e:
        print("Camera close error:", e)
    current_camera = None


def stop_preview_loop():
    preview_stop.set()
    time.sleep(0.15)


def set_status(text, color=None):
    print(text)
    try:
        status_label.config(text=text)
        capture_state_label.config(text=text)
        if color is None:
            lower = text.lower()
            if "ready" in lower:
                color = GREEN
            elif "error" in lower or "not found" in lower or "fail" in lower:
                color = RED
            elif "capture" in lower or "focus" in lower or "opening" in lower:
                color = BLUE
            else:
                color = YELLOW
        capture_state_dot.delete("all")
        capture_state_dot.create_oval(2, 2, 12, 12, fill=color, outline=color)
        root.update_idletasks()
    except Exception:
        pass


def set_step(text):
    print("STEP:", text)
    try:
        step_label.config(text=text)
        root.update_idletasks()
    except Exception:
        pass


def update_preview(frame, is_bgr=False):
    global last_preview_time
    now = time.time()
    if now - last_preview_time < PREVIEW_INTERVAL:
        return
    last_preview_time = now

    try:
        if frame is None:
            return
        display = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        if is_bgr:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=img)
        preview_label.imgtk = imgtk
        preview_label.config(image=imgtk, text="")
    except Exception as e:
        print("Preview error:", e)


def show_page(page):
    for p in [page_input, page_capture, page_complete]:
        p.pack_forget()
    page.pack(fill="both", expand=True)
    root.update_idletasks()


def update_header():
    if current_stage in ("cam1", "cam2"):
        box_title = "Inner Box"
        box_value = f"{inner_index} / {quantity}"
        stage_text = "CAM1 : Vendor + TOA" if current_stage == "cam1" else "CAM2 : TOB"
        progress_title_label.config(text="Inner Box")
        progress_label.config(text=box_value)
        percent = int((inner_index / max(quantity, 1)) * 100)
    elif current_stage == "cam3":
        box_title = "Outer Box"
        box_value = f"{outer_index} / {outer_total}"
        stage_text = "CAM3 : USB / TOC YOLO"
        progress_title_label.config(text="Outer Box")
        progress_label.config(text=box_value)
        percent = int((outer_index / max(outer_total, 1)) * 100)
    else:
        box_title = "-"
        box_value = "-"
        stage_text = "-"
        percent = 0

    info_box_title.config(text=box_title)
    info_box.config(text=box_value)
    info_stage.config(text=stage_text)
    progress_percent_label.config(text=f"{percent}%")
    progress_bar_fill.place(relx=0, rely=0, relwidth=percent / 100, relheight=1)


def get_inner_dir(index):
    return ensure_dir(os.path.join(current_job_dir, "Inner", f"{index:03d}"))


def get_outer_dir(index):
    return ensure_dir(os.path.join(current_job_dir, "Outer", f"{index:03d}"))


def create_job_folder():
    global current_job_dir
    name = datetime.now().strftime("manual_%Y%m%d_%H%M%S")
    current_job_dir = ensure_dir(os.path.join(MANUAL_SAVE_ROOT, name))
    ensure_dir(os.path.join(current_job_dir, "Inner"))
    ensure_dir(os.path.join(current_job_dir, "Outer"))
    return current_job_dir

# =========================================================
# PI CAMERA MANUAL WRAPPER
# =========================================================
class ManualPiCamera:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.picam2 = None
        self.latest_frame = None
        self.thread = None

    def open(self):
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not available. Please install/use on Raspberry Pi.")
        self.picam2 = Picamera2(camera_num=self.index)
        config = self.picam2.create_preview_configuration(
            main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)

    def start_preview(self):
        def loop():
            while not preview_stop.is_set() and self.picam2 is not None:
                try:
                    frame = self.picam2.capture_array()
                    self.latest_frame = frame
                    root.after(0, lambda f=frame: update_preview(f, is_bgr=False))
                except Exception as e:
                    print(f"{self.name} preview error:", e)
                    break
                time.sleep(0.02)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def capture_file(self, path):
        if self.picam2 is None:
            raise RuntimeError(f"{self.name} not opened")
        frame = self.picam2.capture_array()
        self.latest_frame = frame
        # frame จาก Picamera2 เป็น RGB, cv2.imwrite ต้องใช้ BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        return path

    def close(self):
        try:
            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
        except Exception as e:
            print(f"{self.name} close error:", e)
        self.picam2 = None


# =========================================================
# USB CAMERA PREVIEW WRAPPER (CAM3)
# =========================================================
class ManualUsbPreview:
    def __init__(self, device, name="CAM3"):
        self.device = device
        self.name = name
        self.cap = None
        self.thread = None
        self.latest_frame = None

    def open(self):
        self.cap = cv2.VideoCapture(f"/dev/video{self.device}", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        if not self.cap.isOpened():
            raise RuntimeError(f"{self.name}: cannot open /dev/video{self.device}")

        ok = False
        frame = None
        for _ in range(10):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.08)
        if not ok or frame is None:
            self.close()
            raise RuntimeError(f"{self.name}: cannot read frame from /dev/video{self.device}")
        self.latest_frame = frame

    def start_preview(self):
        def loop():
            bad_count = 0
            ready_announced = False
            while not preview_stop.is_set() and self.cap is not None:
                try:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        bad_count += 1
                        if bad_count >= 20:
                            root.after(0, lambda: set_status(f"{self.name}: Signal lost / Cannot Read Frame. Reconnect then press RESET CURRENT CAMERA", RED))
                            break
                        time.sleep(0.08)
                        continue
                    bad_count = 0
                    self.latest_frame = frame
                    if not ready_announced:
                        ready_announced = True
                        root.after(0, lambda: set_status(f"{self.name} Ready on /dev/video{self.device}. Press CAPTURE", GREEN))
                    root.after(0, lambda f=frame: update_preview(f, is_bgr=True))
                except Exception as e:
                    print(f"{self.name} preview error:", e)
                    root.after(0, lambda e=e: set_status(f"{self.name}: Preview error {e}", RED))
                    break
                time.sleep(0.02)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as e:
            print(f"{self.name} close error:", e)
        self.cap = None

# =========================================================
# CAMERA OPEN / RESET
# =========================================================
def open_current_stage_camera():
    global current_camera, last_preview_time
    stop_preview_loop()
    safe_close_camera()
    preview_stop.clear()
    last_preview_time = 0
    update_header()
    show_page(page_capture)

    try:
        if current_stage == "cam1":
            set_step(f"Inner Box {inner_index}/{quantity} : Open CAM1 preview only")
            set_status("CAM1 opening...")
            cam = ManualPiCamera(PICAM1_INDEX, "CAM1")
            cam.open()
            current_camera = cam
            cam.start_preview()
            set_status("CAM1 Ready. Press CAPTURE CAM1")

        elif current_stage == "cam2":
            set_step(f"Inner Box {inner_index}/{quantity} : Open CAM2 preview only")
            set_status("CAM2 opening...")
            cam = ManualPiCamera(PICAM2_INDEX, "CAM2")
            cam.open()
            current_camera = cam
            cam.start_preview()
            set_status("CAM2 Ready. Press CAPTURE CAM2")

        elif current_stage == "cam3":
            set_step(f"Outer Box {outer_index}/{outer_total} : USB camera preview. Press CAPTURE to run YOLO capture")
            set_status("CAM3 scanning USB devices...")
            dev = find_usb_device()
            if dev is None:
                preview_label.config(image="", text="USB Camera not found\nReconnect USB camera and press RESET CURRENT CAMERA")
                set_status("USB Camera not found. Reconnect and press RESET CURRENT CAMERA.", RED)
                return
            cam = ManualUsbPreview(dev, "CAM3")
            cam.open()
            current_camera = cam
            cam.start_preview()
            set_status(f"CAM3 Ready on /dev/video{dev}. Press CAPTURE")

    except Exception as e:
        set_status(f"Open camera error: {e}", RED)
        messagebox.showerror("Camera Error", str(e))


def reset_current_camera():
    """Reset เฉพาะกล้อง/ขั้นตอนปัจจุบัน ไม่ย้อนกลับไปเริ่ม Job ใหม่"""
    if current_stage not in ("cam1", "cam2", "cam3"):
        return
    set_status("Reset current camera... closing and rescanning device")
    stop_preview_loop()
    safe_close_camera()
    try:
        cam = current_cam_holder.get("cam")
        if cam is not None:
            cam.close()
    except Exception:
        pass
    current_cam_holder["cam"] = None
    time.sleep(0.8)  # ให้ OS สร้าง /dev/video ใหม่หลังถอดเสียบ
    open_current_stage_camera()


def get_v4l2_usb_groups():
    """
    อ่านจากคำสั่ง v4l2-ctl --list-devices แล้วจัดกลุ่มเป็น:
    [{"name": "Rapoo Camera...", "videos": [0, 18]}, ...]

    เหตุผล: USB camera ถอดเสียบใหม่แล้ว /dev/video อาจเปลี่ยน
    เช่น /dev/video0 -> /dev/video1 แต่ v4l2-ctl จะบอกชื่อกล้องจริงเสมอ
    """
    groups = []
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=2,
        )
        print("========== v4l2-ctl --list-devices ==========")
        print(out)
        print("=============================================")
    except Exception as e:
        print("v4l2-ctl read error:", e)
        return groups

    current = None
    for raw in out.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        # Device header เช่น: Rapoo Camera: Rapoo Camera (usb-xhci-hcd.0-1):
        if not line.startswith("\t") and line.endswith(":"):
            current = {"name": line[:-1], "videos": []}
            groups.append(current)
            continue

        if "/dev/video" in line and current is not None:
            try:
                dev_no = int(line.strip().replace("/dev/video", ""))
                current["videos"].append(dev_no)
            except Exception:
                pass

    return groups


def _open_read_once(dev, backend=cv2.CAP_V4L2):
    cap = cv2.VideoCapture(f"/dev/video{dev}", backend)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            return False

        # USB camera บางตัวต้อง warm up หลาย frame หลังถอดเสียบใหม่
        for _ in range(25):
            ret, frame = cap.read()
            if ret and frame is not None and getattr(frame, "size", 0) > 0:
                return True
            time.sleep(0.06)
        return False
    finally:
        try:
            cap.release()
        except Exception:
            pass


def _can_read_usb_device(dev):
    """ลองอ่าน frame จริง ไม่ใช่แค่เช็คว่าเปิด device ได้"""
    # ลอง V4L2 ก่อน ถ้าไม่ได้ค่อย fallback เป็น CAP_ANY
    if _open_read_once(dev, cv2.CAP_V4L2):
        return True
    try:
        if _open_read_once(dev, cv2.CAP_ANY):
            return True
    except Exception:
        pass
    return False


def find_usb_device():
    """
    หา USB Camera จาก v4l2-ctl --list-devices ก่อน
    โดยเลือก device ตัวแรกของกลุ่ม Rapoo/USB camera เป็นหลัก
    เช่น Rapoo Camera มี /dev/video0 และ /dev/video18
    ปกติ /dev/video0 คือ video stream ส่วน /dev/video18 มักเป็น metadata
    """
    groups = get_v4l2_usb_groups()
    candidates = []

    # 1) ใช้ v4l2-ctl เป็น source หลัก
    for g in groups:
        name = g.get("name", "")
        lname = name.lower()
        videos = g.get("videos", [])

        is_real_usb_cam = (
            "rapoo" in lname
            or "usb" in lname
            or ("camera" in lname and "rpi" not in lname and "bcm" not in lname)
        )
        if not is_real_usb_cam:
            continue

        # เอาเลขน้อยสุดก่อน เพราะมักเป็น video stream จริง เช่น 0 หรือ 1
        for dev in sorted(videos):
            candidates.append((dev, name, "v4l2"))

    # 2) fallback จาก config/manual list
    for dev in USB_SCAN_DEVICES:
        candidates.append((dev, "manual_list", "manual"))

    # 3) fallback ทุก /dev/video*
    for path in sorted(glob.glob("/dev/video*")):
        try:
            dev = int(path.replace("/dev/video", ""))
            candidates.append((dev, "glob", "glob"))
        except Exception:
            pass

    seen = set()
    last_fail = []
    for dev, name, source in candidates:
        if dev in seen:
            continue
        seen.add(dev)

        # metadata node ของ USB camera มักเป็นเลขสูง เช่น 18/19 และอ่าน frame ไม่ได้
        # แต่ไม่ skip ทิ้งทันที ให้ลองเฉพาะเมื่อไม่มีตัว stream จริงผ่าน
        if dev >= 18 and source != "glob":
            # เก็บไว้ท้ายสุดแทน
            continue

        ok = _can_read_usb_device(dev)
        print(f"USB scan {source}: /dev/video{dev} ({name}) read={ok}")
        if ok:
            set_status(f"CAM3 selected /dev/video{dev} from {source}", BLUE)
            return dev
        last_fail.append(f"/dev/video{dev}")

    # รอบท้าย ลอง metadata/high number เผื่อกล้องบางรุ่น expose stream เป็นเลขสูงจริง
    for dev, name, source in candidates:
        if dev < 18:
            continue
        key = (dev, "high")
        ok = _can_read_usb_device(dev)
        print(f"USB scan high: /dev/video{dev} ({name}) read={ok}")
        if ok:
            set_status(f"CAM3 selected /dev/video{dev} from {source}", BLUE)
            return dev

    set_status("USB Camera not found/readable from v4l2-ctl list", RED)
    print("USB scan failed. Tried:", last_fail)
    return None

# =========================================================
# CAPTURE FLOW
# =========================================================
def start_job():
    global running, quantity, outer_total, inner_index, outer_index, current_stage
    qty_text = qty_entry.get().strip()
    if not qty_text.isdigit():
        messagebox.showwarning("Warning", "Please input Quantity")
        return
    quantity = int(qty_text)
    if quantity < 1 or quantity > 1000:
        messagebox.showwarning("Warning", "Quantity must be 1-1000")
        return

    create_job_folder()
    outer_total = math.ceil(quantity / 6)
    inner_index = 1
    outer_index = 1
    running = True
    current_stage = "cam1"
    open_current_stage_camera()


def capture_current():
    if not running:
        return
    if not capture_lock.acquire(blocking=False):
        return
    threading.Thread(target=_capture_current_worker, daemon=True).start()


def _capture_current_worker():
    try:
        if current_stage in ("cam1", "cam2"):
            capture_pi_stage()
        elif current_stage == "cam3":
            capture_cam3_stage()
    finally:
        try:
            capture_lock.release()
        except Exception:
            pass


def capture_pi_stage():
    global current_stage, inner_index, running
    cam_name = "CAM1" if current_stage == "cam1" else "CAM2"
    save_name = "cam1_vendor_toa.jpg" if current_stage == "cam1" else "cam2_tob.jpg"
    folder = get_inner_dir(inner_index)
    path = os.path.join(folder, save_name)

    try:
        set_status(f"{cam_name} focusing {FOCUS_DELAY_SEC:.1f}s...")
        time.sleep(FOCUS_DELAY_SEC)
        set_status(f"{cam_name} capturing...")
        if current_camera is None:
            raise RuntimeError(f"{cam_name} is not opened")
        current_camera.capture_file(path)
        set_status(f"Saved: {path}", GREEN)
    except Exception as e:
        set_status(f"{cam_name} capture error: {e}", RED)
        messagebox.showerror(f"{cam_name} Error", str(e))
        return

    # ไปขั้นตอนถัดไป
    if current_stage == "cam1":
        current_stage = "cam2"
        root.after(0, open_current_stage_camera)
    else:
        inner_index += 1
        if inner_index <= quantity:
            current_stage = "cam1"
            root.after(0, open_current_stage_camera)
        else:
            current_stage = "cam3"
            root.after(0, open_current_stage_camera)


def save_yolo_rois(image_path, detections, outer_dir):
    img = cv2.imread(image_path)
    if img is None:
        return 0
    roi_count = 0
    roi_dir = ensure_dir(os.path.join(outer_dir, "roi"))

    for det in detections or []:
        name = str(det.get("name", "roi")).lower()
        box = det.get("box", [])
        crop, fixed_box = crop_by_box(img, box, pad=0)
        if crop is None or crop.size == 0:
            continue
        roi_path = os.path.join(roi_dir, f"{name}.jpg")
        cv2.imwrite(roi_path, crop)
        roi_count += 1
    return roi_count


def capture_cam3_stage():
    global outer_index, current_stage, running
    folder = get_outer_dir(outer_index)
    image_path = os.path.join(folder, "cam3_toc_full.jpg")

    # Use the device that the preview already proved can read frames.
    dev = None
    try:
        if isinstance(current_camera, ManualUsbPreview):
            dev = current_camera.device
    except Exception:
        dev = None
    if dev is None:
        dev = find_usb_device()

    if dev is None:
        set_status("USB Camera not found. Reconnect and press RESET CURRENT CAMERA.", RED)
        messagebox.showerror("CAM3 Error", "USB Camera not found. Please reconnect USB camera and press RESET CURRENT CAMERA.")
        root.after(0, open_current_stage_camera)
        return

    # เช็คอีกครั้งก่อน capture เพราะ USB บางครั้ง preview ยังมีภาพค้างแต่ stream หลุดแล้ว
    if not _can_read_usb_device(dev):
        set_status(f"CAM3 /dev/video{dev} cannot read now. Rescanning from v4l2-ctl...", RED)
        dev = find_usb_device()
        if dev is None:
            set_status("USB Camera not readable. Reconnect and press RESET CURRENT CAMERA.", RED)
            root.after(0, open_current_stage_camera)
            return

    # Release preview before YOLO capture opens the same /dev/video device.
    stop_preview_loop()
    safe_close_camera()

    cam3 = None
    try:
        set_step(f"CAM3 Outer Box {outer_index}/{outer_total} : YOLO detect/capture, no OCR")
        set_status(f"CAM3 YOLO opening /dev/video{dev}...")
        cam3 = UsbCameraTest(device=dev, name="CAM3", status_cb=set_status, preview_cb=update_preview)
        current_cam_holder["cam"] = cam3
        cam3.open()
        result = cam3.capture_direct_or_yolo(image_path, lambda: running and current_stage == "cam3")
        cam3.close()
        current_cam_holder["cam"] = None

        if not result:
            set_status("CAM3 capture cancelled or failed. Press RESET CURRENT CAMERA and try again.", RED)
            root.after(0, open_current_stage_camera)
            return

        full_path = result.get("image_path", image_path)
        detections = result.get("detections", []) or []
        roi_count = save_yolo_rois(full_path, detections, folder)
        set_status(f"Saved CAM3 full + ROI count: {roi_count}", GREEN)
    except Exception as e:
        try:
            if cam3 is not None:
                cam3.close()
        except Exception:
            pass
        current_cam_holder["cam"] = None
        set_status(f"CAM3 error: {e}. Press RESET CURRENT CAMERA and try again.", RED)
        messagebox.showerror("CAM3 Error", str(e))
        root.after(0, open_current_stage_camera)
        return

    outer_index += 1
    if outer_index <= outer_total:
        current_stage = "cam3"
        root.after(0, open_current_stage_camera)
    else:
        running = False
        safe_close_camera()
        root.after(0, show_complete)

# ใช้ dict นี้เพื่อให้ CAM3 worker close ได้โดยไม่ชนกับ ManualPiCamera state
current_cam_holder = {"cam": None}


def show_complete():
    complete_title.config(text="MANUAL CAPTURE COMPLETED")
    complete_detail.config(text=f"Inner: {quantity} boxes | Outer: {outer_total} boxes\nSaved at:\n{current_job_dir}")
    show_page(page_complete)


def reset_to_input():
    global running, current_stage, quantity, outer_total, inner_index, outer_index, current_job_dir
    running = False
    stop_preview_loop()
    safe_close_camera()
    try:
        cam = current_cam_holder.get("cam")
        if cam is not None:
            cam.close()
    except Exception:
        pass
    current_cam_holder["cam"] = None
    current_stage = "input"
    quantity = 0
    outer_total = 0
    inner_index = 1
    outer_index = 1
    current_job_dir = ""
    qty_entry.delete(0, tk.END)
    show_page(page_input)


def exit_app():
    reset_running = False
    try:
        stop_preview_loop()
        safe_close_camera()
        cam = current_cam_holder.get("cam")
        if cam is not None:
            cam.close()
    except Exception:
        pass
    root.destroy()

# =========================================================
# UI BUILD
# =========================================================
root = tk.Tk()
root.title("AI Camera Manual Capture - Dataset Mode V3")
root.geometry("1200x720")
root.resizable(False, False)
root.configure(bg=BG)

# ---------- PAGE INPUT ----------
page_input = tk.Frame(root, bg=BG)
input_card = tk.Frame(page_input, bg=CARD, padx=70, pady=42, highlightbackground=BORDER, highlightthickness=1)
input_card.place(relx=0.5, rely=0.50, anchor="center", width=760, height=430)

tk.Label(input_card, text="Manual Dataset Capture", font=("Arial", 30, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(0, 20))
tk.Label(input_card, text="Quantity / Inner Box", bg=CARD, fg=TEXT, font=("Arial", 18, "bold")).pack(anchor="w", pady=(0, 6))
qty_entry = tk.Entry(input_card, font=("Arial", 26), relief="solid", borderwidth=1)
qty_entry.pack(fill="x", ipady=8, pady=(0, 28))

tk.Button(input_card, text="START MANUAL CAPTURE", font=("Arial", 21, "bold"), bg=BLUE, fg="white", activebackground=BLUE_DARK, activeforeground="white", relief="flat", command=start_job).pack(fill="x", ipady=10, pady=(4, 10))
tk.Button(input_card, text="EXIT APP", font=("Arial", 21, "bold"), bg=RED, fg="white", activebackground=RED_DARK, activeforeground="white", relief="flat", command=exit_app).pack(fill="x", ipady=9)

# ---------- PAGE CAPTURE ----------
page_capture = tk.Frame(root, bg=BG)
capture_container = tk.Frame(page_capture, bg=BG)
capture_container.pack(fill="both", expand=True, padx=14, pady=14)

header_frame = tk.Frame(capture_container, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=30, pady=30)
header_frame.pack(fill="x", pady=(0, 12))

header_top = tk.Frame(header_frame, bg=CARD)
header_top.pack(fill="x")
info_left = tk.Frame(header_top, bg=CARD)
info_left.pack(side="left", fill="x", expand=True)
header_buttons = tk.Frame(header_top, bg=CARD)
header_buttons.pack(side="right")

def action_button(parent, text, bg, fg="white", command=None, width=230, height=58):
    wrap = tk.Frame(parent, bg=CARD, width=width, height=height)
    wrap.pack(side="left", padx=(0, 8))
    wrap.pack_propagate(False)
    btn = tk.Button(wrap, text=text, bg=bg, fg=fg, font=("Arial", 14, "bold"), relief="flat", cursor="hand2", command=command, activebackground=BLUE_DARK if bg == BLUE else (RED_DARK if bg == RED else GREEN_DARK), activeforeground="white")
    btn.pack(fill="both", expand=True)
    return btn

action_button(header_buttons, "RESET CURRENT CAMERA", GRAY, fg=TEXT, command=reset_current_camera, width=245)
action_button(header_buttons, "CAPTURE", GREEN, command=capture_current, width=180)
action_button(header_buttons, "EXIT", RED, command=exit_app, width=120)

info_row = tk.Frame(info_left, bg=CARD)
info_row.pack(fill="x")

def header_info_box(parent, title, width=14):
    box = tk.Frame(parent, bg=CARD)
    box.pack(side="left", padx=(0, 70))
    tk.Label(box, text=title, bg=CARD, fg=MUTED, font=("Arial", 21, "bold"), anchor="w").pack(anchor="w")
    val = tk.Label(box, text="-", bg=CARD, fg=TEXT, font=("Arial", 25, "bold"), width=width, anchor="w")
    val.pack(anchor="w", pady=(6, 0))
    return val

info_box_wrap = tk.Frame(info_row, bg=CARD)
info_box_wrap.pack(side="left", padx=(0, 70))
info_box_title = tk.Label(info_box_wrap, text="Inner Box", bg=CARD, fg=MUTED, font=("Arial", 21, "bold"), anchor="w")
info_box_title.pack(anchor="w")
info_box = tk.Label(info_box_wrap, text="-", bg=CARD, fg=TEXT, font=("Arial", 25, "bold"), width=12, anchor="w")
info_box.pack(anchor="w", pady=(6, 0))
info_stage = header_info_box(info_row, "Stage", 24)

body_frame = tk.Frame(capture_container, bg=BG)
body_frame.pack(fill="both", expand=True)

capture_left = tk.Frame(body_frame, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=14, width=800)
capture_left.pack(side="left", fill="both", expand=False, padx=(0, 10))
capture_left.pack_propagate(False)

tk.Label(capture_left, text="Capture Images", bg=CARD, fg=TEXT, font=("Arial", 24, "bold")).pack(anchor="w")
tk.Label(capture_left, text="Preview", bg=CARD, fg=MUTED, font=("Arial", 14, "bold")).pack(anchor="w", pady=(8, 4))

preview_frame = tk.Frame(capture_left, bg=DARK, width=760, height=430, highlightbackground=DARK, highlightthickness=2)
preview_frame.pack(pady=(4, 10))
preview_frame.pack_propagate(False)
preview_label = tk.Label(preview_frame, text="Camera Preview", bg=DARK, fg="white", font=("Arial", 20))
preview_label.place(x=0, y=0, relwidth=1, relheight=1)

preview_overlay = tk.Frame(preview_frame, bg=DARK)
preview_overlay.place(x=10, y=8)
capture_state_dot = tk.Canvas(preview_overlay, width=14, height=14, bg=DARK, highlightthickness=0)
capture_state_dot.pack(side="left", padx=(0, 5))
capture_state_dot.create_oval(2, 2, 12, 12, fill=YELLOW, outline=YELLOW)
capture_state_label = tk.Label(preview_overlay, text="Waiting", bg=DARK, fg="white", font=("Arial", 14, "bold"), padx=2, pady=2)
capture_state_label.pack(side="left")

preview_button_row = tk.Frame(capture_left, bg=CARD)
preview_button_row.pack(fill="x", pady=(8, 0))

reset_preview_btn = tk.Button(
    preview_button_row,
    text="RESET CURRENT CAMERA",
    bg=GRAY,
    fg=TEXT,
    activebackground="#aeb4b8",
    activeforeground=TEXT,
    font=("Arial", 13, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
    command=reset_current_camera,
)
reset_preview_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

capture_preview_btn = tk.Button(
    preview_button_row,
    text="CAPTURE",
    bg=GREEN,
    fg="white",
    activebackground=GREEN_DARK,
    activeforeground="white",
    font=("Arial", 13, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
    command=capture_current,
)
capture_preview_btn.pack(side="left", fill="x", expand=True)

capture_right = tk.Frame(body_frame, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=22, pady=18)
capture_right.pack(side="right", fill="both", expand=True)

tk.Label(capture_right, text="Status", bg=CARD, fg=TEXT, font=("Arial", 24, "bold")).pack(anchor="w")

product_header = tk.Frame(capture_right, bg=CARD)
product_header.pack(fill="x", pady=(18, 2))
progress_title_label = tk.Label(product_header, text="Inner Box", bg=CARD, fg=TEXT, font=("Arial", 18, "bold"))
progress_title_label.pack(side="left")

product_row = tk.Frame(capture_right, bg=CARD)
product_row.pack(fill="x")
progress_label = tk.Label(product_row, text="0 / 0", bg=CARD, fg=TEXT, font=("Arial", 19, "bold"))
progress_label.pack(side="left")
progress_percent_label = tk.Label(product_row, text="0%", bg=CARD, fg=TEXT, font=("Arial", 17, "bold"))
progress_percent_label.pack(side="right")

progress_bar_bg = tk.Frame(capture_right, bg="#c4c4c8", height=26)
progress_bar_bg.pack(fill="x", pady=(4, 20))
progress_bar_bg.pack_propagate(False)
progress_bar_fill = tk.Frame(progress_bar_bg, bg=BLUE)
progress_bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

step_status_box = tk.Frame(capture_right, bg=SOFT, padx=12, pady=10)
step_status_box.pack(fill="both", expand=True, pady=(16, 10))

tk.Label(step_status_box, text="Step", bg=SOFT, fg="#34495e", font=("Arial", 16, "bold")).pack(anchor="w")
step_label = tk.Label(step_status_box, text="-", bg=SOFT, fg=TEXT, font=("Arial", 14), wraplength=360, justify="left")
step_label.pack(anchor="w", pady=(4, 18))

tk.Label(step_status_box, text="Status", bg=SOFT, fg="#34495e", font=("Arial", 16, "bold")).pack(anchor="w")
status_label = tk.Label(step_status_box, text="-", bg=SOFT, fg=TEXT, font=("Arial", 14), wraplength=360, justify="left")
status_label.pack(anchor="w", pady=(4, 0))

hint = tk.Label(step_status_box, text="Manual Mode:\n- CAM1/CAM2 ไม่มี background detection\n- กด Capture เองทุกครั้ง\n- CAM3 ใช้ YOLO เดิม แต่ไม่ OCR\n- Reset Current Camera ไม่เริ่มงานใหม่", bg=SOFT, fg=MUTED, font=("Arial", 13), justify="left", wraplength=360)
hint.pack(anchor="w", pady=(28, 0))

# ---------- PAGE COMPLETE ----------
page_complete = tk.Frame(root, bg=BG)
complete_card = tk.Frame(page_complete, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
complete_card.place(relx=0.5, rely=0.45, anchor="center", width=680, height=330)
complete_title = tk.Label(complete_card, text="MANUAL CAPTURE COMPLETED", font=("Arial", 24, "bold"), bg=CARD, fg=GREEN)
complete_title.pack(pady=(10, 16))
complete_detail = tk.Label(complete_card, text="-", font=("Arial", 13), bg=CARD, fg=TEXT, wraplength=600, justify="center")
complete_detail.pack(pady=(0, 22))
tk.Button(complete_card, text="OK", bg=BLUE, fg="white", font=("Arial", 16, "bold"), width=18, relief="flat", command=reset_to_input).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", exit_app)
show_page(page_input)
root.mainloop()
