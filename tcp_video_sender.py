import cv2
import socket
import struct
import time
import argparse
import numpy as np

MAX_PAYLOAD = 900
HEADER_FMT = "!IHHH"   # frame_id, chunk_id, total_chunks, payload_len
HEADER_SIZE = struct.calcsize(HEADER_FMT)

#cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

def run_sender_tcp(device, ip, port, width, height, fps):
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Camera failed to open")
    print("[TX] Camera opened successfully")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    print("[TX] Connected to receiver at", ip, port)

    delay = 1.0 / fps
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ok:
            continue

        data = jpeg.tobytes()
        sock.sendall(struct.pack("!I", len(data)) + data)
        frame_id += 1
        time.sleep(delay)

def run_receiver_tcp(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", port))
    sock.listen(1)
    print("[RX] Waiting for connection...")
    conn, addr = sock.accept()
    print("[RX] Connected by", addr)

    cv2.namedWindow("TCP Stream", cv2.WINDOW_NORMAL)

    while True:
        # Read 4-byte length
        data = b""
        while len(data) < 4:
            packet = conn.recv(4 - len(data))
            if not packet:
                break
            data += packet
        if not data:
            break
        frame_len = struct.unpack("!I", data)[0]

        # Read the full frame
        frame_data = b""
        while len(frame_data) < frame_len:
            packet = conn.recv(frame_len - len(frame_data))
            if not packet:
                break
            frame_data += packet

        img = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("TCP Stream", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    conn.close()
    sock.close()

def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--mode", choices=["send", "recv"], required=True)
 args = parser.parse_args()

 if args.mode == "send":
  run_sender_tcp(
   "/dev/video0",
   "192.168.137.229", # "192.168.137.229"  "192.168.137.1"
   5005,
   1280,
   720,
   10
  )
 else:
  print("Rx!")
  run_receiver_tcp(5005)

if __name__ == "__main__":
 main()
