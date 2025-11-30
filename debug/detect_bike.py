import cv2
import os
import sys

VIDEO_PATH = "IMG_1242.MOV"
WINDOW_NAME = "CV Player AutoTrack"


def create_tracker():
    if hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()
    return cv2.TrackerCSRT_create()


def auto_detect_object(frame, backsub, MIN_AREA=5000):
    fg = backsub.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(
        fg,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA and area > best_area:
            x, y, w, h = cv2.boundingRect(c)
            best_area = area
            best_box = (x, y, w, h)

    return best_box


def main():
    if not os.path.exists(VIDEO_PATH):
        print("Video not found:", VIDEO_PATH)
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video:", VIDEO_PATH)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    backsub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )

    tracker = None
    tracking = False
    trail = []
    delay = 30  # ms / frame ≈ 33 FPS

    # ===== Seek bar (only for scrubbing, not for resetting tracking) =====
    def on_trackbar(_):
        # Do nothing here – we will read the value manually in the loop
        pass

    cv2.createTrackbar("Seek", WINDOW_NAME, 0, total_frames - 1, on_trackbar)
    last_seek_pos = 0

    while True:
        # ---- Handle seeking (only allowed when NOT tracking) ----
        seek_pos = cv2.getTrackbarPos("Seek", WINDOW_NAME)
        if not tracking and seek_pos != last_seek_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_pos)
            last_seek_pos = seek_pos
            trail = []

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        display = frame.copy()

        # =================== TRACK MODE ====================
        if tracking and tracker is not None:
            ok, box = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in box]
                cx = x + w // 2
                cy = y + h // 2

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

                trail.append((cx, cy))
                for i in range(1, len(trail)):
                    cv2.line(display, trail[i - 1], trail[i], (255, 0, 0), 2)

                text = f"Tracking | Frame {frame_idx}"
                color = (0, 255, 0)
            else:
                text = f"Tracking lost | Frame {frame_idx}"
                color = (0, 0, 255)

        # =================== PLAYBACK MODE ====================
        else:
            text = f"Playback | Frame {frame_idx} (press 't' to auto-track)"
            color = (0, 255, 255)

        cv2.putText(
            display,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        # Update seek bar to current frame (no-op callback)
        cv2.setTrackbarPos("Seek", WINDOW_NAME, frame_idx)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(delay) & 0xFF

        # Quit
        if key in (27, ord("q")):
            break

        # Pause / Play
        if key == ord(" "):
            delay = 0 if delay != 0 else 30

        # Faster / slower
        if key == ord("+") or key == ord("="):
            delay = max(1, delay - 5)
        if key == ord("-") or key == ord("_"):
            delay += 5

        # Step forward one frame
        if key == ord("d"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)

        # Auto-track trigger
        if key == ord("t"):
            box = auto_detect_object(frame, backsub)
            if box is not None:
                tracker = create_tracker()
                tracker.init(frame, box)
                tracking = True
                trail = []
                print("Auto-init tracker with box:", box)
            else:
                print("No moving object detected at this frame.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
