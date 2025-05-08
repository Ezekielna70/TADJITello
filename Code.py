import cv2, threading, time, queue, keyboard
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

frame_read = None
frame_queue   = queue.Queue(maxsize=1)
command_queue = queue.Queue()

latest_boxes          = []
TRACK_CONF_THRESHOLD = 0.8
center_margin   = 30       # toleransi horizontal (px)
vertical_margin = 20       # toleransi vertikal (px)

# flags untuk aksi satu-kali
car_left_triggered  = False
car_right_triggered = False

# teks overlay aksi terakhir
last_action = "(Auto Track)"
prev_move   = "diam"       # kiri/kanan/maju/mundur/diam

# =========== UTIL ============

def set_last(txt):
    global last_action
    last_action = txt

def queue_cmd(cmd, arg, label):
    """Flush queue lalu masukkan satu perintah terbaru; update teks overlay."""
    while not command_queue.empty():
        try: command_queue.get_nowait()
        except queue.Empty: break
    set_last(label)
    command_queue.put((cmd, arg))

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

# =========== YOLO THREAD ============
def yolo_thread(model, drone):
    global latest_boxes, takeoff_done, manual_override
    global prev_move, running, car_left_triggered, car_right_triggered

    while running:
        try:
            frame = frame_queue.get(timeout=0.2)
        except queue.Empty:
            time.sleep(0.01)
            continue
        if frame is None:
            time.sleep(0.01)
            continue

        res = model.predict(frame, conf=0.3)
        names = res[0].names if len(res)>0 else []
        latest_boxes = [
            (b, float(b.conf[0]), names[int(b.cls[0])])
            for b in res[0].boxes
        ] if names else []

        move_needed = "diam"
        if takeoff_done and not manual_override and latest_boxes:
            box, _, cls = latest_boxes[0]
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2

            if cls == "Car Toy Behind":
                # reset flags aksi khusus
                car_left_triggered = car_right_triggered = False
                # horizontal auto-track
                if cx < FRAME_W//2 - center_margin:
                    move_needed = "kiri"
                elif cx > FRAME_W//2 + center_margin:
                    move_needed = "kanan"
                # vertical auto-track
                if cy < TOP_LINE_Y - vertical_margin:
                    move_needed = "maju"
                elif cy > BOTTOM_LINE_Y + vertical_margin:
                    move_needed = "mundur"

            # aksi sekuensial untuk Car Toy Left
            elif cls == "Car Toy Left" and not car_left_triggered:
                car_left_triggered = True
                # flush queue
                while not command_queue.empty():
                    command_queue.get_nowait()
                # 1) geser kanan 100 cm
                queue_cmd("move_right", 20, "kanan 100cm")
                time.sleep(3)
                # 2) maju 100 cm
                queue_cmd("move_forward", 20, "maju 100cm")
                time.sleep(3)
                # 3) rotate kiri 90°
                queue_cmd("rotate_ccw", 90, "rot kiri 90°")

            # aksi sekuensial untuk Car Toy Kanan (Car Toy Right)
            elif cls == "Car Toy Right" and not car_right_triggered:
                car_right_triggered = True
                # flush queue
                while not command_queue.empty():
                    command_queue.get_nowait()
                # 1) geser kiri 100 cm
                queue_cmd("move_left", 20, "kiri 100cm")
                time.sleep(3)
                # 2) maju 100 cm
                queue_cmd("move_forward", 20, "maju 100cm")
                time.sleep(3)
                # 3) rotate kanan 90°
                queue_cmd("rotate_cw", 90, "rot kanan 90°")

        # kirim perintah auto-move jika perlu
        if move_needed != prev_move:
            prev_move = move_needed
            if   move_needed == "kiri":   queue_cmd("move_left", 20, "kiri")
            elif move_needed == "kanan":  queue_cmd("move_right", 20, "kanan")
            
            elif move_needed == "maju":   queue_cmd("move_forward", 20, "maju")
            elif move_needed == "mundur": queue_cmd("move_back", 20, "mundur")
            else:
                set_last("diam")
                while not command_queue.empty():
                    command_queue.get_nowait()

        time.sleep(0.01)

# =========== CAMERA THREAD ============
def camera_thread(drone):
    global frame_read, camera_ready_count, takeoff_done, running
    fps_start, frames = time.time(), 0

    while running:
        if not frame_read or frame_read.frame is None:
            time.sleep(0.01)
            continue
        frame = frame_read.frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        camera_ready_count += 1
        if camera_ready_count >= CAMERA_READY_THRESHOLD and not takeoff_done:
            queue_cmd("takeoff", None, "Takeoff")
            time.sleep(3)
            takeoff_done = True

        frames += 1
        fps = frames / max(1e-6, time.time() - fps_start)
        small = cv2.resize(frame, (FRAME_W, FRAME_H))
        if not frame_queue.full():
            frame_queue.put(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        disp = small.copy()
        for b,conf,c in latest_boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(disp, f"{c} {conf:.2f}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.line(disp, (0,TOP_LINE_Y), (FRAME_W,TOP_LINE_Y), (0,0,255), 2)
        cv2.line(disp, (0,BOTTOM_LINE_Y), (FRAME_W,BOTTOM_LINE_Y), (0,0,255), 2)
        cv2.line(disp, (FRAME_W//2,0), (FRAME_W//2,FRAME_H), (255,0,0), 2)

        status = "(Manual Mode)" if manual_override else \
                 "(Belum Takeoff)" if not takeoff_done else last_action
        cv2.putText(disp, f"FPS:{fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(disp, status, (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("TelloCam", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            queue_cmd("land", None, "Landing")
            break

    cv2.destroyAllWindows()

# =========== MANUAL CONTROL THREAD ============
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
        'q':("land",None,"Landing")
    }
    while running:
        if keyboard.is_pressed('o'):
            manual_override = not manual_override
            set_last("(Manual Mode)" if manual_override else "(Auto Track)")
            time.sleep(0.4)
        for k,(cmd,arg,lab) in key_map.items():
            if keyboard.is_pressed(k):
                queue_cmd(cmd, arg, lab)
                if k=='q': running=False
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

    model = YOLO(r"D:\Kuliah_ITS\Semester_8\TA Kelar Amin\Code\Git\best.pt")
    model.fuse = lambda *a, **k: model

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
    print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
