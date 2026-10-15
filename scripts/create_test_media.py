"""Generate synthetic test media for empty-room and one-person tracking scenarios."""

import os
import cv2
import numpy as np


def draw_background(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a realistic indoor room background (floor, wall, doorway)."""
    frame = np.full((height, width, 3), 220, dtype=np.uint8)  # Light gray wall
    # Floor (darker gray / brown)
    cv2.rectangle(frame, (0, 320), (width, height), (140, 140, 140), -1)
    # Baseboard line
    cv2.line(frame, (0, 320), (width, 320), (80, 80, 80), 3)
    # Doorway on the right
    cv2.rectangle(frame, (420, 100), (580, 320), (180, 120, 70), -1)
    cv2.rectangle(frame, (420, 100), (580, 320), (60, 40, 20), 4)
    # Door handle
    cv2.circle(frame, (440, 210), 5, (40, 40, 40), -1)
    return frame


def draw_synthetic_person(frame: np.ndarray, cx: int, cy: int, scale: float = 1.0):
    """Draw a stylized human figure resembling a person."""
    # Head
    head_radius = int(14 * scale)
    head_center = (cx, cy - int(60 * scale))
    cv2.circle(frame, head_center, head_radius, (60, 60, 60), -1)
    # Torso (jacket)
    torso_top_y = head_center[1] + head_radius
    torso_bottom_y = cy
    torso_w = int(24 * scale)
    cv2.rectangle(
        frame,
        (cx - torso_w, torso_top_y),
        (cx + torso_w, torso_bottom_y),
        (40, 80, 160),  # Blue jacket
        -1,
    )
    # Legs (trousers)
    leg_w = int(10 * scale)
    leg_bottom_y = cy + int(60 * scale)
    cv2.rectangle(frame, (cx - torso_w + 2, torso_bottom_y), (cx - 2, leg_bottom_y), (50, 50, 50), -1)
    cv2.rectangle(frame, (cx + 2, torso_bottom_y), (cx + torso_w - 2, leg_bottom_y), (50, 50, 50), -1)


def generate_empty_room_video(output_path: str, num_frames: int = 100):
    """Generate empty-room video clip with zero people."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    bg = draw_background()

    for _ in range(num_frames):
        # Add slight camera noise to simulate real sensor
        noise = np.random.normal(0, 1.5, bg.shape).astype(np.int16)
        noisy_frame = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(noisy_frame)

    out.release()
    print(f"Generated empty-room video: {output_path} ({num_frames} frames)")


def generate_walking_person_video(output_path: str, num_frames: int = 120):
    """Generate video of one person walking across line and into restricted zone."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    # Path from (150, 150) across y=240 and into zone (350, 150)-(550, 400)
    start_x, start_y = 150, 140
    end_x, end_y = 480, 300

    for i in range(num_frames):
        frame = draw_background()
        alpha = min(1.0, i / (num_frames - 20))
        cx = int(start_x + (end_x - start_x) * alpha)
        cy = int(start_y + (end_y - start_y) * alpha)
        
        # Draw person if within walking duration
        if i < num_frames - 10:
            draw_synthetic_person(frame, cx, cy)

        out.write(frame)

    out.release()
    print(f"Generated walking-person video: {output_path} ({num_frames} frames)")


def main():
    data_dir = "tests/data"
    os.makedirs(data_dir, exist_ok=True)
    generate_empty_room_video(os.path.join(data_dir, "empty_room_test.mp4"))
    generate_walking_person_video(os.path.join(data_dir, "one_person_walk.mp4"))


if __name__ == "__main__":
    main()
