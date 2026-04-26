import socket
import threading
import cv2
from time import sleep

# ----------------- CONSTANTS -----------------

PORT = 5050
HEADER = 128
FORMAT = "utf-8"

IMG_MSG = "!IMG"
CMD_MSG = "!CMD"


# ----------------- CLIENT -----------------

class Client:
    def __init__(self, server_ip):
        self.server_ip = server_ip
        self.connected = True
        self.sock = None

    # ----------------- SOCKET HELPERS -----------------

    def connect(self):
        # Force IPv4 and clean routing on macOS
        self.sock = socket.create_connection(
            (self.server_ip, PORT),
            timeout=5,
            source_address=("", 0)
        )
        self.sock.settimeout(None)
        print("[CONNECTED]")

    def send_image(self, img_bytes):
        size = len(img_bytes)

        header = IMG_MSG.encode(FORMAT).ljust(HEADER, b" ")
        size_header = str(size).encode(FORMAT).ljust(HEADER, b" ")

        self.sock.sendall(header)
        self.sock.sendall(size_header)
        self.sock.sendall(img_bytes)

    # ----------------- RECEIVE (OPTIONAL) -----------------

    def recv_exact(self, size):
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def receive(self):
        header = self.recv_exact(HEADER)
        if not header:
            return None

        header = header.decode(FORMAT).strip()
        length_bytes = self.recv_exact(HEADER)
        if not length_bytes:
            return None

        length = int(length_bytes.decode(FORMAT).strip())
        payload = self.recv_exact(length)
        if not payload:
            return None

        return payload.decode(FORMAT, errors="ignore")

    def listen(self):
        while self.connected:
            try:
                msg = self.receive()
                if msg is None:
                    break
            except:
                break

    # ----------------- MAIN LOOP -----------------

    def start(self):
        print("Connecting to:", self.server_ip, PORT)
        self.connect()

        threading.Thread(target=self.listen, daemon=True).start()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not accessible")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue

                self.send_image(buf.tobytes())
                sleep(0.1)  # ~10 FPS
        finally:
            cap.release()
            self.sock.close()


# ----------------- ENTRY -----------------

if __name__ == "__main__":
    # server_ip = input("Server IP address: ").strip()
    server_ip = "10.0.0.245"
    Client(server_ip).start()