import socket
import threading
import time
import numpy as np
import cv2
import os

PORT = 5050
HEADER = 128
FORMAT = "utf-8"

IMG_MSG = "!IMG"
CMD_MSG = "!CMD"

# For testing, save every frame. Change back to 10 later.
SAVE_EVERY_N_FRAMES = 30


class Server:
    def __init__(self):
        self.SERVER = "0.0.0.0"
        self.ADDR = (self.SERVER, PORT)

        self.start_time = 0.0
        self.frames = 0

        self.img_buff = None
        self.img_lock = threading.Lock()

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.running = True

        # Save images beside this server.py file
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(BASE_DIR, "Ai", "images")
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"[SAVE DIR] {self.save_dir}")

    def recv_exact(self, conn, size: int):
        data = b""

        while len(data) < size:
            try:
                packet = conn.recv(size - len(data))
            except ConnectionResetError:
                return None

            if not packet:
                return None

            data += packet

        return data

    def recv_header_str(self, conn):
        raw = self.recv_exact(conn, HEADER)

        if raw is None:
            return None

        return raw.decode(FORMAT, errors="ignore").strip()

    def receive(self, conn):
        msg_type = self.recv_header_str(conn)

        if not msg_type:
            return None

        if msg_type == IMG_MSG:
            return self.receive_img(conn)

        if msg_type == CMD_MSG:
            length_str = self.recv_header_str(conn)

            if not length_str:
                return None

            try:
                length = int(length_str)
            except ValueError:
                print(f"[ERROR] Invalid command length: {length_str}")
                return None

            payload = self.recv_exact(conn, length)

            if payload is None:
                return None

            return payload.decode(FORMAT, errors="ignore")

        print(f"[ERROR] Unknown message type: {msg_type}")
        return None

    def receive_img(self, conn):
        size_str = self.recv_header_str(conn)

        if not size_str:
            return None

        try:
            size = int(size_str)
        except ValueError:
            print(f"[ERROR] Invalid image size header: {size_str}")
            return None

        img_bytes = self.recv_exact(conn, size)

        if img_bytes is None:
            print("[ERROR] Failed to receive image bytes")
            return None

        self.frames += 1

        with self.img_lock:
            self.img_buff = bytes(img_bytes)

        print(f"[IMG RECEIVED] Frame {self.frames}, size={len(img_bytes)} bytes")

        if SAVE_EVERY_N_FRAMES and self.frames % SAVE_EVERY_N_FRAMES == 0:
            frame = cv2.imdecode(
                np.frombuffer(img_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                print(f"[ERROR] Could not decode frame {self.frames}")
                return IMG_MSG

            filename = f"frame_{self.frames:06d}.jpg"
            filepath = os.path.join(self.save_dir, filename)

            success = cv2.imwrite(filepath, frame)

            if success:
                print(f"[SAVED] {filepath}")
            else:
                print(f"[ERROR] Failed to save {filepath}")

        return IMG_MSG

    def get_latest_frame_bytes(self):
        with self.img_lock:
            if self.img_buff is None:
                return None

            return bytes(self.img_buff)

    def start(self):
        try:
            self.server.bind(self.ADDR)
            self.server.listen()
        except OSError as e:
            print(f"[ERROR] Could not start server: {e}")
            return

        print(f"[SERVER] Listening on {self.SERVER}:{PORT}")
        self.start_time = time.time()

        try:
            while self.running:
                conn, addr = self.server.accept()
                print(f"[NEW CONNECTION] {addr}")

                thread = threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr),
                    daemon=True
                )
                thread.start()

        except KeyboardInterrupt:
            print("\n[SERVER] Shutting down...")

        finally:
            self.running = False
            self.server.close()

    def handle_client(self, conn, addr):
        try:
            while self.running:
                msg = self.receive(conn)

                if msg is None:
                    break

        finally:
            try:
                conn.close()
            except Exception:
                pass

            print(f"[DISCONNECTED] {addr}")


if __name__ == "__main__":
    server = Server()
    server.start()