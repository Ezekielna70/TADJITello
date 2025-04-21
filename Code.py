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

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# GARIS REFERENSI TRACKING
TOP_LINE_Y = int(FRAME_HEIGHT * 0.3)
BOTTOM_LINE_Y = int(FRAME_HEIGHT * 0.5)

frame_queue = queue.Queue(maxsize=1)
command_queue = queue.Queue()

latest_boxes = []
latest_detection_time = 0
TRACK_CONF_THRESHOLD = 0.8

car_left_triggered = False
car_right_triggered = False

def bypass_fuse(model):
    model.fuse = lambda *args, **kwargs: model

# =========== COMMAND THREAD ============
def command_thread(drone):
    global running
    while running:
        try:
            cmd, arg = command_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.01)
            continue

        try:
            if cmd == "takeoff":
                drone.takeoff()
            elif cmd == "land":
                drone.land()
                running = False
                break
            elif cmd == "move_left":
                drone.move_left(arg)
            elif cmd == "move_right":
                drone.move_right(arg)
            elif cmd == "move_forward":
                drone.move_forward(arg)
            elif cmd == "move_back":
                drone.move_back(arg)
            elif cmd == "move_up":
                drone.move_up(arg)
            elif cmd == "move_down":
                drone.move_down(arg)
            elif cmd == "rotate_ccw":
                drone.rotate_counter_clockwise(arg)
            elif cmd == "rotate_cw":
                drone.rotate_clockwise(arg)
            elif cmd == "flip_left":
                drone.flip_left()
            elif cmd == "flip_right":
                drone.flip_right()
            elif cmd == "custom":
                drone.send_control_command(arg)
            else:
                print(f"[WARNING] Unknown command: {cmd}")
        except Exception as e:
            print(f"[ERROR] Command '{cmd}': {e}")

# =========== YOLO THREAD ============
def yolo_thread(model, drone):
    global running, latest_boxes, latest_detection_time, takeoff_done
    global manual_override
    global car_left_triggered, car_right_triggered

    while running:
        try:
            frame = frame_queue.get(timeout=0.2)
        except queue.Empty:
            time.sleep(0.01)
            continue
        if frame is None:
            time.sleep(0.01)
            continue

        results = model.predict(frame, conf=0.3)
        new_boxlist = []

        if len(results) > 0:
            names = results[0].names
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                conf_val = float(box.conf[0])

                if conf_val < TRACK_CONF_THRESHOLD:
                    continue
                new_boxlist.append((box, conf_val, cls_name))

        latest_boxes = new_boxlist
        latest_detection_time = time.time()

        if takeoff_done and not manual_override and len(new_boxlist) > 0:
            (box_obj, conf_val, cname) = new_boxlist[0]
            x1, y1, x2, y2 = map(int, box_obj.xyxy[0])
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2

            if cname == "Car Toy Behind":
                car_left_triggered = False
                car_right_triggered = False

                if obj_center_x < FRAME_WIDTH // 2 - 50:
                    command_queue.put(("move_left", 20))
                elif obj_center_x > FRAME_WIDTH // 2 + 50:
                    command_queue.put(("move_right", 20))

                if obj_center_y < TOP_LINE_Y:
                    command_queue.put(("move_forward", 20))
                elif obj_center_y > BOTTOM_LINE_Y:
                    command_queue.put(("move_back", 20))

            elif cname == "Car Toy Left" and not car_left_triggered:
                print("[INFO] Car Toy Left terdeteksi -> geser kanan, maju, rotate kiri")
                command_queue.put(("move_right", 20))
                time.sleep(1)
                command_queue.put(("move_forward", 10))
                time.sleep(1)
                command_queue.put(("rotate_ccw", 90))
                time.sleep(2)
                car_left_triggered = True

            elif cname == "Car Toy Right" and not car_right_triggered:
                print("[INFO] Car Toy Right terdeteksi -> geser kiri, maju, rotate kanan")
                command_queue.put(("move_left", 20))
                time.sleep(1)
                command_queue.put(("move_forward", 10))
                time.sleep(1)
                command_queue.put(("rotate_cw", 90))
                time.sleep(2)
                car_right_triggered = True

        time.sleep(0.01)

# =========== CAMERA THREAD ============
def camera_thread(drone):
    global running, frame_read, camera_ready_count, takeoff_done
    global movement_text, latest_boxes

    fps_start = time.time()
    frame_count = 0

    while running:
        if frame_read is None or frame_read.frame is None:
            time.sleep(0.01)
            continue

        frame = frame_read.frame
        camera_ready_count += 1
        if camera_ready_count >= CAMERA_READY_THRESHOLD and not takeoff_done:
            command_queue.put(("takeoff", None))
            print("[INFO] Kamera siap, drone takeoff")
            time.sleep(3)
            takeoff_done = True

        frame_count += 1
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0

        frame_small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if not frame_queue.full():
            rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            frame_queue.put(rgb_frame)

        display_frame = frame_small.copy()

        for (box_obj, conf_val, cls_name) in latest_boxes:
            x1, y1, x2, y2 = map(int, box_obj.xyxy[0])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf_val:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.line(display_frame, (0, TOP_LINE_Y), (FRAME_WIDTH, TOP_LINE_Y), (0, 0, 255), 2)
        cv2.line(display_frame, (0, BOTTOM_LINE_Y), (FRAME_WIDTH, BOTTOM_LINE_Y), (0, 0, 255), 2)
        cv2.line(display_frame, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT), (255, 0, 0), 2)

        movement_text = "(Manual Mode)" if manual_override else "(Auto Track)" if takeoff_done else "(Belum Takeoff)"
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Gerakan: {movement_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Tello Camera", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            command_queue.put(("land", None))
            break

    cv2.destroyAllWindows()

# =========== MANUAL CONTROL THREAD ============
def manual_control():
    global running, manual_override, tracking_mode
    print("=== Manual Control ===")
    print("o: toggle manual/auto, q: land & exit")

    while running:
        if keyboard.is_pressed('o'):
            manual_override = not manual_override
            tracking_mode = not manual_override
            print("== Manual Mode ==" if manual_override else "== Auto Track ==")
            time.sleep(0.4)

        keys = {
            'w': ("move_forward", 30),
            's': ("move_back", 30),
            'a': ("move_left", 30),
            'd': ("move_right", 30),
            'i': ("move_up", 30),
            'k': ("move_down", 30),
            'j': ("rotate_ccw", 30),
            'l': ("rotate_cw", 30),
            '9': ("flip_left", None),
            '0': ("flip_right", None),
            't': ("takeoff", None),
            'q': ("land", None)
        }

        for key, cmd in keys.items():
            if keyboard.is_pressed(key):
                command_queue.put(cmd)
                time.sleep(0.15)
                if key == 'q':
                    running = False
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

    model_path = r"D:\Kuliah_ITS\Semester_8\TA Kelar Amin\Code\Git\best.pt"
    model = YOLO(model_path)
    bypass_fuse(model)

    threads = [
        threading.Thread(target=command_thread, args=(drone,)),
        threading.Thread(target=yolo_thread, args=(model, drone)),
        threading.Thread(target=camera_thread, args=(drone,)),
        threading.Thread(target=manual_control)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    drone.streamoff()
    drone.end()
    print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
