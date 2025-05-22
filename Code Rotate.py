import cv2, threading, time, queue, keyboard
from djitellopy import Tello
from collections import deque
import statistics

# ================== GLOBALS ==================
running = True
camera_ready_count = 0
CAMERA_READY_THRESHOLD = 10
takeoff_done = False
manual_override = False

FRAME_W, FRAME_H = 640, 480

frame_read = None
frame_queue   = queue.Queue(maxsize=1)
command_queue = queue.Queue()

# List untuk menyimpan detail command + response time
command_log = []
response_times = []
rotation_times = []  # Khusus untuk response time rotasi

# Variabel untuk pengujian rotasi otomatis
rotation_test_active = False
last_rotation_time = 0
rotation_count = 0
MAX_ROTATIONS = 20  # Total rotasi yang akan diuji (10 CW + 10 CCW)

# teks overlay aksi terakhir
last_action = "(Rotation Test Mode)"

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

def print_rotation_statistics():
    """Cetak statistik khusus rotasi"""
    if rotation_times:
        print("\n=== STATISTIK ROTASI 90 DERAJAT ===")
        print(f"Total rotasi diuji: {len(rotation_times)}")
        print(f"Rata-rata response time: {statistics.mean(rotation_times):.3f} s")
        print(f"Response time minimum: {min(rotation_times):.3f} s")
        print(f"Response time maksimum: {max(rotation_times):.3f} s")
        print(f"Standar deviasi: {statistics.stdev(rotation_times) if len(rotation_times) > 1 else 0:.3f} s")
        
        print("\n=== DETAIL SETIAP ROTASI ===")
        for i, rt in enumerate(rotation_times, 1):
            direction = "CW" if i % 2 == 1 else "CCW"
            print(f"Rotasi {i} ({direction}): {rt:.3f} s")

# =========== COMMAND THREAD ============
def command_thread(drone: Tello):
    global running, response_times, command_log, rotation_times

    while running:
        try:
            cmd, arg = command_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.01)
            continue

        # rekam waktu sebelum kirim
        t_start = time.time()
        try:
            # kirim perintah dan tunggu response dari drone
            if   cmd == "takeoff":
                res = drone.send_control_command("takeoff")
            elif cmd == "land":
                res = drone.send_control_command("land")
                running = False
            elif cmd == "rotate_ccw":
                res = drone.send_control_command(f"ccw {arg}")
            elif cmd == "rotate_cw":
                res = drone.send_control_command(f"cw {arg}")
            # Hapus semua perintah movement lainnya untuk fokus ke rotasi
            else:
                res = "COMMAND DISABLED FOR ROTATION TEST"

        except Exception as e:
            res = f"ERROR: {e}"

        # hitung response time
        t_end   = time.time()
        elapsed = t_end - t_start
        response_times.append(elapsed)

        # Simpan response time khusus untuk rotasi
        if cmd in ["rotate_cw", "rotate_ccw"]:
            rotation_times.append(elapsed)

        # simpan ke command_log
        command_log.append((cmd, arg, elapsed))

        # print seperti biasa
        response_text = res.strip() if isinstance(res, str) else str(res)
        print(f"[RT] Command `{cmd}` arg={arg} → response `{response_text}` in {elapsed:.3f} s")

    # setelah loop selesai, cetak rangkuman
    print("\n=== Rangkuman Command dan Response Time ===")
    for c, a, e in command_log:
        arg_str = f" {a}" if a is not None else ""
        print(f"{c}{arg_str} response {e:.3f}s")

    # Cetak statistik rotasi
    print_rotation_statistics()

# =========== ROTATION TEST THREAD ============
def rotation_test_thread():
    """Thread khusus untuk mengirim perintah rotasi setiap 5 detik"""
    global running, rotation_test_active, last_rotation_time, rotation_count, MAX_ROTATIONS
    
    while running and rotation_test_active:
        current_time = time.time()
        
        # Kirim perintah rotasi setiap 5 detik
        if current_time - last_rotation_time >= 5.0 and rotation_count < MAX_ROTATIONS:
            # Alternating antara CW dan CCW
            if rotation_count % 2 == 0:
                queue_cmd("rotate_cw", 90, f"Rotasi CW #{rotation_count//2 + 1}")
                print(f"[TEST] Mengirim rotasi CW ke-{rotation_count//2 + 1}")
            else:
                queue_cmd("rotate_ccw", 90, f"Rotasi CCW #{rotation_count//2 + 1}")
                print(f"[TEST] Mengirim rotasi CCW ke-{rotation_count//2 + 1}")
            
            rotation_count += 1
            last_rotation_time = current_time
            
        # Berhenti setelah mencapai maksimum rotasi
        elif rotation_count >= MAX_ROTATIONS:
            print(f"[TEST] Pengujian rotasi selesai! Total {MAX_ROTATIONS} rotasi telah diuji.")
            rotation_test_active = False
            
        time.sleep(0.1)

# =========== CAMERA THREAD (Simplified) ============
def camera_thread(drone):
    global frame_read, camera_ready_count, takeoff_done, running
    global manual_override, last_action, rotation_test_active, last_rotation_time

    # Buffer timestamp untuk 1 detik terakhir
    timestamps = deque()

    while running:
        # Tunggu frame dari drone
        if not frame_read or frame_read.frame is None:
            time.sleep(0.01)
            continue

        frame = frame_read.frame

        # Resize dan konversi ke RGB
        small = cv2.resize(frame, (FRAME_W, FRAME_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Auto‐takeoff setelah threshold
        camera_ready_count += 1
        if camera_ready_count >= CAMERA_READY_THRESHOLD and not takeoff_done:
            queue_cmd("takeoff", None, "Takeoff")
            print("[INFO] Takeoff otomatis...")
            time.sleep(5)  # Tunggu takeoff selesai
            takeoff_done = True
            
            # Mulai test rotasi setelah takeoff
            rotation_test_active = True
            last_rotation_time = time.time()
            print("[INFO] Memulai pengujian rotasi dalam 5 detik...")

        # Hitung FPS dengan window 1 detik
        now = time.time()
        timestamps.append(now)
        # Buang timestamp >1 detik yang lalu
        while timestamps and now - timestamps[0] > 1.0:
            timestamps.popleft()
        fps = len(timestamps)

        # Gambar display sederhana (tanpa deteksi objek)
        disp = rgb.copy()
        
        # Tampilkan informasi pengujian
        status_lines = [
            f"ROTATION TEST MODE",
            f"Rotasi selesai: {rotation_count}/{MAX_ROTATIONS}",
            f"Test aktif: {'YA' if rotation_test_active else 'TIDAK'}",
            last_action
        ]
        
        for i, line in enumerate(status_lines):
            cv2.putText(disp, line, (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Tampilkan countdown untuk rotasi berikutnya
        if rotation_test_active and rotation_count < MAX_ROTATIONS:
            time_until_next = 5.0 - (now - last_rotation_time)
            if time_until_next > 0:
                cv2.putText(disp, f"Rotasi berikutnya dalam: {time_until_next:.1f}s", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Tampilkan window
        cv2.imshow("Rotation Test", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            queue_cmd("land", None, "Landing")
            break

    cv2.destroyAllWindows()

# =========== MANUAL CONTROL THREAD (Simplified) ============
def manual_control():
    global running, manual_override, rotation_test_active
    
    while running:
        # Toggle manual override
        if keyboard.is_pressed('o'):
            manual_override = not manual_override
            set_last("(Manual Mode)" if manual_override else "(Rotation Test)")
            time.sleep(0.4)
            
        # Manual takeoff
        if keyboard.is_pressed('t') and manual_override:
            queue_cmd("takeoff", None, "Manual Takeoff")
            time.sleep(0.2)
            
        # Manual landing
        if keyboard.is_pressed('q'):
            running = False
            queue_cmd("land", None, "Landing")
            time.sleep(0.2)
            
        # Start/stop rotation test
        if keyboard.is_pressed('r') and takeoff_done:
            rotation_test_active = not rotation_test_active
            if rotation_test_active:
                global rotation_count, last_rotation_time
                rotation_count = 0
                last_rotation_time = time.time()
                print("[INFO] Memulai ulang pengujian rotasi...")
            else:
                print("[INFO] Menghentikan pengujian rotasi...")
            time.sleep(0.4)
            
        time.sleep(0.02)

# =========== MAIN ============
def main():
    global frame_read
    
    print("=== PENGUJIAN RESPONSE TIME ROTASI DRONE ===")
    print("Kontrol:")
    print("- 'r': Start/Stop rotation test")
    print("- 'o': Toggle manual override")
    print("- 't': Manual takeoff (saat manual mode)")
    print("- 'q': Landing dan keluar")
    print("=" * 50)
    
    drone = Tello()
    drone.connect()
    print("[INFO] Battery:", drone.get_battery())
    drone.streamon()
    frame_read = drone.get_frame_read()

    threads = [
        threading.Thread(target=command_thread, args=(drone,)),
        threading.Thread(target=rotation_test_thread),
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