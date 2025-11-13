# scripts/test_video_open.py
import sys, cv2, time, os

if len(sys.argv) != 2:
    print('Usage: python scripts/test_video_open.py "path\\to\\video.mp4"')
    sys.exit(1)

path = sys.argv[1]
print("Video path:", os.path.abspath(path))

cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print("ERROR: Cannot open video. Try full path, different file, or reinstall opencv-python.")
    sys.exit(2)

# Print stream properties
w  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])

print(f"Opened OK  |  size={int(w)}x{int(h)}  fps={fps:.2f}  codec='{codec}'")

# Read a few frames just to confirm decoding
n = 0
t0 = time.time()
while n < 60:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
elapsed = time.time() - t0
cap.release()

print(f"Read {n} frames in {elapsed:.2f}s  (~{(n/elapsed) if elapsed>0 else 0:.1f} fps)")
print("Done. (Nothing was saved or opened.)")
