import cv2
import threading
import time
import queue
import keyboard
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO

# ================== GLOBALS ==================
running = True
camera_ready_count = 0
CAMERA_READY_THRESHOLD = 10
takeoff_done = False
manual_override = False

FRAME_W, FRAME_H = 640, 480
TOP_LINE_Y      = int(FRAME_H * 0.60)
BOTTOM_LINE_Y   = int(FRAME_H * 0.70)

frame_read    = None
frame_queue   = queue.Queue(maxsize=1)
command_queue = queue.Queue()

# threshold
DETECTION_CONF     = 0.3    # confidence threshold segmentasi
TRACK_CONF_THRESHOLD = 0.5  # confidence threshold untuk tracking

# toleransi
center_margin   = 20
vertical_margin = 20

# flags aksi satu kali
car_left_triggered  = False
car_right_triggered = False

# teks overlay aksi terakhir
last_action = "(Auto Track)"

# ================= UTIL =====================
def set_last(txt):
    global last_action
    last_action = txt

def queue_cmd(cmd, arg, label):
    """Kosongkan queue lalu masukkan perintah terbaru."""
    while not command_queue.empty():
        try: command_queue.get_nowait()
        except queue.Empty: break
    set_last(label)
    command_queue.put((cmd, arg))

# ========== COMMAND THREAD =============
def command_thread(drone):
    global running
    while running:
        try:
            cmd, arg = command_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.01)
            continue
        try:
            if   cmd=="takeoff":      drone.takeoff()
            elif cmd=="land":         drone.land(); running=False; break
            elif cmd=="move_left":    drone.move_left(arg)
            elif cmd=="move_right":   drone.move_right(arg)
            elif cmd=="move_forward": drone.move_forward(arg)
            elif cmd=="move_back":    drone.move_back(arg)
            elif cmd=="move_up":      drone.move_up(arg)
            elif cmd=="move_down":    drone.move_down(arg)
            elif cmd=="rotate_ccw":   drone.rotate_counter_clockwise(arg)
            elif cmd=="rotate_cw":    drone.rotate_clockwise(arg)
            elif cmd=="flip_left":    drone.flip_left()
            elif cmd=="flip_right":   drone.flip_right()
        except Exception as e:
            print("[ERROR]", e)

# =========== YOLO THREAD =============
def yolo_thread(model, drone):
    """Ambil frame dari frame_queue, deteksi segmentasi, dan gambar mask + bbox."""
    global car_left_triggered, car_right_triggered

    while running:
        try:
            frame = frame_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # 1) prediksi
        results = model.predict(source=frame, conf=DETECTION_CONF, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()      # Nx4
        confs = results.boxes.conf.cpu().numpy()      # N
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        names   = [model.names[i] for i in cls_ids]

        # 2) ambil polygon mask
        polys = []
        if results.masks is not None and hasattr(results.masks, "xy") and results.masks.xy:
            for poly in results.masks.xy:
                pts = [[int(p[0]), int(p[1])] for p in poly]
                polys.append(pts)
        else:
            polys = [None]*len(boxes)

        # 3) draw dan tracking logic
        latest_boxes = []
        overlay = frame.copy()
        for i, (box, conf, cls_id, name) in enumerate(zip(boxes, confs, cls_ids, names)):
            if conf < TRACK_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            # gambar mask
            poly = polys[i]
            if poly and len(poly)>2:
                pts = np.array(poly, dtype=np.int32).reshape((-1,1,2))
                cv2.fillPoly(overlay, [pts], (0,255,0))

            # gambar bbox + label pada frame
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

            # simpan untuk tracking
            latest_boxes.append(((x1,y1,x2,y2), conf, name))

        # blend mask overlay
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # kirim frame ke camera_thread utk ditampilkan
        if not frame_queue.full():
            frame_queue.put(frame)

# =========== CAMERA THREAD =============
def camera_thread(drone):
    global frame_read, camera_ready_count, takeoff_done, running
    fps_start, frames = time.time(), 0

    while running:
        if not frame_read or frame_read.frame is None:
            time.sleep(0.01)
            continue
        raw = frame_read.frame
        camera_ready_count += 1

        # auto-takeoff setelah buffer siap
        if camera_ready_count>=CAMERA_READY_THRESHOLD and not takeoff_done:
            queue_cmd("takeoff", None, "Takeoff")
            time.sleep(3)
            takeoff_done=True

        # resize & compute fps
        small = cv2.resize(raw, (FRAME_W, FRAME_H))
        frames += 1
        fps = frames / max(1e-6, time.time()-fps_start)

        # masukkan ke frame_queue utk yolo_thread
        if not frame_queue.full():
            frame_queue.put(small)

        # ambil hasil deteksi (frame sudah ter-overlay di yolo_thread)
        disp = small.copy()
        if not frame_queue.empty():
            disp = frame_queue.get()

        # gambar garis bantu & status
        cv2.line(disp, (0,TOP_LINE_Y), (FRAME_W,TOP_LINE_Y), (0,0,255), 2)
        cv2.line(disp, (0,BOTTOM_LINE_Y), (FRAME_W,BOTTOM_LINE_Y), (0,0,255), 2)
        cv2.line(disp, (FRAME_W//2,0), (FRAME_W//2,FRAME_H), (255,0,0), 2)
        status = "(Manual)" if manual_override else \
                 "(Belum Takeoff)" if not takeoff_done else last_action
        cv2.putText(disp, f"FPS:{fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(disp, status, (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("TelloCam", disp)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            running=False
            queue_cmd("land", None, "Landing")
            break

    cv2.destroyAllWindows()

# ===== MANUAL CONTROL THREAD =====
def manual_control():
    global running, manual_override
    key_map = {
        'w':("move_forward",30,"maju"),
        's':("move_back",30,"mundur"),
        'a':("move_left",30,"kiri"),
        'd':("move_right",30,"kanan"),
        'i':("move_up",30,"naik"),
        'k':("move_down",30,"turun"),
        'j':("rotate_ccw",30,"rot kiri"),
        'l':("rotate_cw",30,"rot kanan"),
        '9':("flip_left",None,"flip kiri"),
        '0':("flip_right",None,"flip kanan"),
        't':("takeoff",None,"Takeoff"),
        'q':("land",None,"Landing"),
    }
    while running:
        if keyboard.is_pressed('o'):
            manual_override = not manual_override
            set_last("(Manual)" if manual_override else "(Auto)")
            time.sleep(0.4)
        for k,(cmd,arg,lab) in key_map.items():
            if keyboard.is_pressed(k):
                queue_cmd(cmd,arg,lab)
                if cmd=="land": running=False
                time.sleep(0.2)
                break
        time.sleep(0.02)

# =========== MAIN ============
def main():
    global frame_read
    drone = Tello()
    drone.connect()
    print("[INFO] Battery:", drone.get_battery())
    drone.streamon()
    frame_read = drone.get_frame_read()

    # load YOLOv8 segmen
    model = YOLO(r"D:\Kuliah_ITS\Semester_8\TA Kelar Amin\Model\5 Segmentasi\weights\best.pt")

    # jalankan threads
    threads = [
        threading.Thread(target=command_thread, args=(drone,)),
        threading.Thread(target=yolo_thread,    args=(model,drone)),
        threading.Thread(target=camera_thread,  args=(drone,)),
        threading.Thread(target=manual_control)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    drone.streamoff()
    drone.end()
    print("[INFO] Selesai.")

if __name__=="__main__":
    main()
