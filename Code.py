import cv2
import threading
import time
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import keyboard
import queue

# ================== GLOBALS ==================
running = True

camera_ready_count = 0
CAMERA_READY_THRESHOLD = 10
takeoff_done = False

manual_override = False
tracking_mode = True

movement_text = ""
frame_read = None

# Frame size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TARGET_HEIGHT = 150

# Kita akan kirim frame ke YOLO tiap frame (bisa tiap N frame), 
# tapi agar tak menumpuk, kita batasi queue size=1.
frame_queue = queue.Queue(maxsize=1)

# Queue perintah Tello
command_queue = queue.Queue()

# Hasil YOLO
latest_boxes = []
latest_detection_time = 0

# Bypass fuse
def bypass_fuse(model):
    model.fuse = lambda *args, **kwargs: model

# ============ THREAD COMMAND ============
def command_thread(drone):
    global running
    while running:
        try:
            cmd, arg = command_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.01)
            continue

        if cmd == "takeoff":
            print("[CMD] takeoff")
            try:
                drone.takeoff()
            except Exception as e:
                print(f"[ERROR Takeoff] {e}")

        elif cmd == "land":
            print("[CMD] land")
            try:
                drone.land()
            except Exception as e:
                print(f"[ERROR Land] {e}")
            running = False
            break

        elif cmd == "move_left":
            print(f"[CMD] move_left({arg})")
            drone.move_left(arg)

        elif cmd == "move_right":
            print(f"[CMD] move_right({arg})")
            drone.move_right(arg)

        elif cmd == "move_forward":
            print(f"[CMD] move_forward({arg})")
            drone.move_forward(arg)

        elif cmd == "move_back":
            print(f"[CMD] move_back({arg})")
            drone.move_back(arg)

        elif cmd == "move_up":
            print(f"[CMD] move_up({arg})")
            drone.move_up(arg)

        elif cmd == "move_down":
            print(f"[CMD] move_down({arg})")
            drone.move_down(arg)

        elif cmd == "rotate_ccw":
            print(f"[CMD] rotate_ccw({arg})")
            drone.rotate_counter_clockwise(arg)

        elif cmd == "rotate_cw":
            print(f"[CMD] rotate_cw({arg})")
            drone.rotate_clockwise(arg)

        elif cmd == "flip_left":
            print("[CMD] flip_left")
            drone.flip_left()

        elif cmd == "flip_right":
            print("[CMD] flip_right")
            drone.flip_right()

        elif cmd == "custom":
            print(f"[CMD] custom({arg})")
            drone.send_control_command(arg)
        else:
            print(f"[WARNING] Unknown command: {cmd}")

# ============ THREAD YOLO (INFERENCE) ============
def yolo_thread(model, drone):
    """
    Mengambil frame dari frame_queue, melakukan YOLO detect,
    jika auto-track & takeoff_done & !manual_override => kirim command
    simpan bounding boxes ke global latest_boxes
    """
    global running, latest_boxes, latest_detection_time, takeoff_done
    global manual_override

    while running:
        try:
            frame = frame_queue.get(timeout=0.2)
        except queue.Empty:
            time.sleep(0.01)
            continue
        if frame is None:
            time.sleep(0.01)
            continue

        # YOLO detect
        results = model.predict(frame, conf=0.3)
        boxes_temp = []
        if len(results) > 0:
            names = results[0].names
            filtered = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                if cls_name.lower() == "wheel toy":
                    continue
                filtered.append(box)
            results[0].boxes = filtered
            boxes_temp = filtered

        # Simpan hasil bounding box
        latest_boxes = boxes_temp
        latest_detection_time = time.time()

        # Auto tracking?
        if takeoff_done and (not manual_override) and len(boxes_temp) > 0:
            # Ambil box pertama untuk auto track
            box = boxes_temp[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_center_x = (x1 + x2)//2
            obj_height = (y2 - y1)
            # Asumsi input frame di yolo_thread sama resolusi => 640x480
            frame_center_x = FRAME_WIDTH//2

            # Horizontal
            if obj_center_x < frame_center_x - 50:
                command_queue.put(("move_left", 20))
            elif obj_center_x > frame_center_x + 50:
                command_queue.put(("move_right", 20))

            # Maju/mundur
            if obj_height < TARGET_HEIGHT - 20:
                command_queue.put(("move_forward", 20))
            elif obj_height > TARGET_HEIGHT + 20:
                command_queue.put(("move_back", 20))

        time.sleep(0.01)

# ============ THREAD CAMERA/DISPLAY ============
def camera_thread(drone):
    global running, frame_read, camera_ready_count, takeoff_done
    global movement_text, latest_boxes

    fps_start = time.time()
    frame_count = 0

    while running:
        if frame_read is None:
            time.sleep(0.01)
            continue

        frame = frame_read.frame
        if frame is None:
            time.sleep(0.01)
            continue

        # hitung readiness
        camera_ready_count += 1
        if (camera_ready_count >= CAMERA_READY_THRESHOLD) and (not takeoff_done):
            # masukkan perintah takeoff
            command_queue.put(("takeoff", None))
            print("[INFO] Kamera siap, queue takeoff")
            time.sleep(3)
            takeoff_done = True

        # Tampilkan FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        fps = frame_count/elapsed if elapsed>0 else 0

        # resize => ringankan
        frame_small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # masukkan ke yolo queue
        # agar tidak menumpuk, cek penuh atau tidak
        if not frame_queue.full():
            # convert ke RGB
            rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            frame_queue.put(rgb_frame)

        # Gambar bounding box terakhir
        # latest_boxes => bounding box
        # Gambar di frame_small
        display_frame = frame_small.copy()
        for box in latest_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(display_frame, (x1,y1),(x2,y2),(0,255,0),2)

        # Tampilkan movement_text
        # movement_text diganti dengan penanda manual/auto
        if manual_override:
            movement_text = "(Manual Mode)"
        else:
            if not takeoff_done:
                movement_text = "(Belum Takeoff)"
            else:
                movement_text = "(Auto Track)"

        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(display_frame, f"Gerakan: {movement_text}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("Tello Camera", display_frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            running=False
            command_queue.put(("land",None))
            break

    cv2.destroyAllWindows()

# ============ THREAD MANUAL CONTROL ============
def manual_control():
    global running, manual_override, tracking_mode
    print("=== Manual Control ===")
    print("o: toggle manual/auto, q: land & exit")
    print("w/a/s/d: gerak, i/k: naik/turun, j/l: rotate, 9/0: flip, t:takeoff")

    while running:
        if keyboard.is_pressed('o'):
            manual_override = not manual_override
            tracking_mode = not manual_override
            print("== Manual Mode ==" if manual_override else "== Auto Track ==")
            time.sleep(0.4)

        if keyboard.is_pressed('w'):
            command_queue.put(("move_forward", 30))
            time.sleep(0.15)
        elif keyboard.is_pressed('s'):
            command_queue.put(("move_back", 30))
            time.sleep(0.15)
        elif keyboard.is_pressed('a'):
            command_queue.put(("move_left", 30))
            time.sleep(0.15)
        elif keyboard.is_pressed('d'):
            command_queue.put(("move_right", 30))
            time.sleep(0.15)
        elif keyboard.is_pressed('i'):
            command_queue.put(("move_up", 30))
            time.sleep(0.15)
        elif keyboard.is_pressed('k'):
            command_queue.put(("move_down", 30))
            time.sleep(0.15)
        elif keyboard.is_pressed('j'):
            command_queue.put(("rotate_ccw", 30))
            time.sleep(0.15)
        elif keyboard.is_pressed('l'):
            command_queue.put(("rotate_cw", 30))
            time.sleep(0.15)
        elif keyboard.is_pressed('9'):
            command_queue.put(("flip_left", None))
            time.sleep(0.2)
        elif keyboard.is_pressed('0'):
            command_queue.put(("flip_right", None))
            time.sleep(0.2)
        elif keyboard.is_pressed('t'):
            command_queue.put(("takeoff", None))
            time.sleep(1)
        elif keyboard.is_pressed('q'):
            print("[MANUAL] Quit & land...")
            command_queue.put(("land", None))
            time.sleep(1)
            running=False
            break

        time.sleep(0.02)

# ============ MAIN ============
def main():
    global frame_read

    drone = Tello()
    drone.connect()
    print("[INFO] Battery:", drone.get_battery())

    drone.streamon()
    frame_read = drone.get_frame_read()

    # Load YOLO
    model_path = r"D:\Kuliah_ITS\Semester_8\TA Kelar Amin\Code\Git\best.pt"
    model = YOLO(model_path)
    bypass_fuse(model)

    # Thread command
    cmd_thread = threading.Thread(target=command_thread, args=(drone,))
    cmd_thread.start()

    # Thread yolo
    yolo_t = threading.Thread(target=yolo_thread, args=(model,drone))
    yolo_t.start()

    # Thread camera
    cam_thread = threading.Thread(target=camera_thread, args=(drone,))
    cam_thread.start()

    # Thread manual
    ctrl_thread = threading.Thread(target=manual_control)
    ctrl_thread.start()

    cam_thread.join()
    ctrl_thread.join()
    yolo_t.join()
    cmd_thread.join()

    drone.streamoff()
    drone.end()
    print("[INFO] Program selesai")

if __name__=="__main__":
    main()
