"""
可视化脚本 - 生成机器人动作对比图和视频
用法:
    python visualize.py                          # 生成对比图 (png)
    python visualize.py --video                  # 生成对比视频 (mp4)
    python visualize.py --video --max_frames 200 # 限制帧数
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pybullet as p
import pybullet_data
import pandas as pd


def load_csv(csv_path):
    """加载关节序列 CSV，返回 (N, 10) numpy 数组（度）。"""
    df = pd.read_csv(csv_path)
    cols = [
        'torso_bend', 'torso_twist',
        'left_shoulder', 'left_elbow',
        'right_shoulder', 'right_elbow',
        'left_hip', 'left_knee',
        'right_hip', 'right_knee',
    ]
    data = df[cols].values.astype(float)
    return data


def setup_pybullet(urdf_path, width=640, height=480):
    """初始化 PyBullet DIRECT 模式，返回 (client_id, robot_id, joint_ids)。"""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=100)
    p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
    rid = p.loadURDF(urdf_path, basePosition=[0, 0, 1.0], useFixedBase=True)

    control_joints = [
        'torso_bend', 'torso_twist',
        'left_shoulder', 'left_elbow',
        'right_shoulder', 'right_elbow',
        'left_hip', 'left_knee',
        'right_hip', 'right_knee',
    ]
    joint_ids = []
    for i in range(p.getNumJoints(rid)):
        info = p.getJointInfo(rid, i)
        name = info[1].decode('utf-8')
        if name in control_joints and info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joint_ids.append(i)

    return cid, rid, joint_ids


def set_frame(rid, joint_ids, angles_deg, force=800):
    """设置一帧关节角度。"""
    angles_rad = np.deg2rad(angles_deg)
    n = min(len(angles_rad), len(joint_ids))
    for i in range(n):
        p.setJointMotorControl2(
            rid, joint_ids[i], p.POSITION_CONTROL,
            targetPosition=float(angles_rad[i]),
            targetVelocity=0.0, force=force,
            positionGain=0.4, velocityGain=0.8,
        )
    for _ in range(20):
        p.stepSimulation()


def render_frame(rid, width=640, height=480):
    """渲染一帧，返回 BGR 图像。"""
    # 从斜上方看机器人
    target = [0, 0, 0.8]
    cam_dist = 2.5
    cam_yaw = 45
    cam_pitch = -25
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=cam_dist,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=10,
    )
    _, _, px, _, _ = p.getCameraImage(
        width, height, view, proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = np.reshape(px, (height, width, 4))[:, :, :3].astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def draw_skeleton_on_frame(frame, landmarks_33):
    """在视频帧上绘制骨架。"""
    if landmarks_33 is None:
        return frame
    h, w = frame.shape[:2]
    out = frame.copy()
    # MediaPipe 骨架连接
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
        (11, 23), (12, 24), (23, 24),                       # torso
        (23, 25), (25, 27), (24, 26), (26, 28),             # legs
    ]
    for i, j in connections:
        x1, y1 = int(landmarks_33[i][0] * w), int(landmarks_33[i][1] * h)
        x2, y2 = int(landmarks_33[j][0] * w), int(landmarks_33[j][1] * h)
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for k in range(33):
        x, y = int(landmarks_33[k][0] * w), int(landmarks_33[k][1] * h)
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
    return out


def create_comparison_images(csv_path, urdf_path, video_path, output_dir, num_images=8):
    """生成若干张对比图：左边原视频+骨架，右边机器人。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    angles = load_csv(csv_path)
    total = len(angles)
    indices = np.linspace(0, total - 1, num_images, dtype=int)

    # PyBullet
    cid, rid, jids = setup_pybullet(urdf_path, width=640, height=480)

    # Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Import capture for skeleton drawing
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from capture import MotionCapturer
    capturer = MotionCapturer(model_complexity=2)

    images = []
    for idx in indices:
        # Read video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Resize to match robot render
        frame = cv2.resize(frame, (640, 480))
        lm = capturer.get_landmarks_from_frame(frame)
        left_img = draw_skeleton_on_frame(frame, capturer.landmarks_to_array(lm))

        # Add frame info
        t = idx / fps
        cv2.putText(left_img, f"Frame {idx}  t={t:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(left_img, "MediaPipe", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Render robot
        set_frame(rid, jids, angles[idx])
        right_img = render_frame(rid, 640, 480)
        cv2.putText(right_img, f"Frame {idx}  t={t:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(right_img, "PyBullet", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Combine
        combined = np.hstack([left_img, right_img])
        images.append(combined)

        out_path = output_dir / f"comparison_{idx:04d}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  saved: {out_path}")

    cap.release()
    capturer.close()
    p.disconnect(cid)

    # Save a contact sheet (2x4 grid)
    if len(images) >= 4:
        rows = []
        for r in range(0, len(images), 2):
            row_imgs = images[r:r+2]
            if len(row_imgs) < 2:
                row_imgs.append(np.zeros_like(row_imgs[0]))
            rows.append(np.hstack(row_imgs))
        while len(rows) < 4:
            rows.append(np.zeros_like(rows[0]))
        contact = np.vstack(rows[:4])
        cs_path = output_dir / "contact_sheet.png"
        cv2.imwrite(str(cs_path), contact)
        print(f"  contact sheet: {cs_path}")

    print(f"\nDone! {num_images} comparison images saved to {output_dir}")


def create_comparison_video(csv_path, urdf_path, video_path, output_path, max_frames=None):
    """生成对比视频：左边原视频+骨架，右边机器人。"""
    angles = load_csv(csv_path)
    total = len(angles)
    if max_frames:
        total = min(max_frames, total)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize video frames to 640x480 for consistency
    out_w, out_h = 640, 480

    cid, rid, jids = setup_pybullet(urdf_path, width=out_w, height=out_h)

    from capture import MotionCapturer
    capturer = MotionCapturer(model_complexity=2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w * 2, out_h))

    print(f"Generating comparison video: {total} frames @ {fps:.0f} FPS")
    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # Left: original + skeleton
        frame_resized = cv2.resize(frame, (out_w, out_h))
        lm = capturer.get_landmarks_from_frame(frame)
        left_img = draw_skeleton_on_frame(frame_resized, capturer.landmarks_to_array(lm))

        t = idx / fps
        cv2.putText(left_img, f"Frame {idx}  t={t:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Right: robot
        set_frame(rid, jids, angles[idx])
        right_img = render_frame(rid, out_w, out_h)

        combined = np.hstack([left_img, right_img])
        out.write(combined)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{total}")

    cap.release()
    out.release()
    capturer.close()
    p.disconnect(cid)
    print(f"Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Robot dance visualization")
    parser.add_argument('--csv', default='outputs/joint_sequence.csv')
    parser.add_argument('--urdf', default='assets/humanoid_robot_mediapipe_v2.urdf')
    parser.add_argument('--video', default='assets/video6330050713560816015.mp4')
    parser.add_argument('--output_dir', default='outputs/visualization')
    parser.add_argument('--mode', choices=['images', 'video', 'both'], default='both')
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    print("=" * 50)
    print("Robot Dance Visualization")
    print("=" * 50)

    if args.mode in ('images', 'both'):
        print("\n[1] Generating comparison images...")
        create_comparison_images(args.csv, args.urdf, args.video, args.output_dir)

    if args.mode in ('video', 'both'):
        print("\n[2] Generating comparison video...")
        vid_path = Path(args.output_dir) / "comparison_video.mp4"
        create_comparison_video(args.csv, args.urdf, args.video, vid_path, args.max_frames)

    print("\nDone!")


if __name__ == '__main__':
    main()
