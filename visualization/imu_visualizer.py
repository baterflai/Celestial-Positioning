from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_data(path: str):
    with open(path, "r") as f:
        return json.load(f)


def deg_or_rad_series(data, key_rad: str, key_deg: str):
    if key_deg in data[0]:
        return np.array([math.radians(float(row[key_deg])) for row in data], dtype=float)
    return np.array([float(row[key_rad]) for row in data], dtype=float)


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])

    return rz @ ry @ rx


def create_box_vertices(length=1.6, width=1.0, height=0.35):
    lx = length/2
    ly = width/2
    lz = height/2

    return np.array([
        [-lx,-ly,-lz],
        [ lx,-ly,-lz],
        [ lx, ly,-lz],
        [-lx, ly,-lz],
        [-lx,-ly, lz],
        [ lx,-ly, lz],
        [ lx, ly, lz],
        [-lx, ly, lz]
    ])


def box_faces(v):
    return [
        [v[0],v[1],v[2],v[3]],
        [v[4],v[5],v[6],v[7]],
        [v[0],v[1],v[5],v[4]],
        [v[2],v[3],v[7],v[6]],
        [v[1],v[2],v[6],v[5]],
        [v[0],v[3],v[7],v[4]],
    ]


def transform_vertices(vertices, R):
    return (R @ vertices.T).T


def set_axes_equal(ax, limit=1.5):
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-limit,limit)
    ax.set_zlim(-limit,limit)
    ax.set_box_aspect([1,1,1])


def draw_body_axes(ax, R, scale=1.2):
    origin = np.zeros(3)

    x = R @ np.array([scale,0,0])
    y = R @ np.array([0,scale,0])
    z = R @ np.array([0,0,scale])

    ax.plot([0,x[0]],[0,x[1]],[0,x[2]],linewidth=3)
    ax.plot([0,y[0]],[0,y[1]],[0,y[2]],linewidth=3)
    ax.plot([0,z[0]],[0,z[1]],[0,z[2]],linewidth=3)


def draw_world_axes(ax, scale=1.2):
    ax.plot([0,scale],[0,0],[0,0],linestyle="--",linewidth=1.5)
    ax.plot([0,0],[0,scale],[0,0],linestyle="--",linewidth=1.5)
    ax.plot([0,0],[0,0],[0,scale],linestyle="--",linewidth=1.5)

def draw_frame(ax, vertices, times, roll, pitch, yaw, i):
    ax.cla()

    r = roll[i]
    p = pitch[i]
    y = yaw[i]

    R = rotation_matrix_from_rpy(r, p, y)

    rv = transform_vertices(vertices, R)
    faces = box_faces(rv)

    box = Poly3DCollection(faces, alpha=0.7, edgecolor="k")
    ax.add_collection3d(box)

    draw_world_axes(ax)
    draw_body_axes(ax, R)
    set_axes_equal(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        f"3D IMU Orientation Viewer\n"
        f"t={times[i]:.2f}s  "
        f"roll={math.degrees(r):.1f}° "
        f"pitch={math.degrees(p):.1f}° "
        f"yaw={math.degrees(y):.1f}°",
        pad=30
    )

    ax.view_init(elev=20, azim=35)


def main():

    # input_path = Path("complementary_filter_output_with_cal.json")
    # input_path = Path("complementary_filter_output.json")
    input_path = Path("complementary_filter_output_with_cal_99.json")
    data = load_data(str(input_path))

    times = np.array([float(row["timestamp"]) for row in data])
    times = times - times[0]

    roll = deg_or_rad_series(data,"roll","roll_deg")
    pitch = deg_or_rad_series(data,"pitch","pitch_deg")
    yaw = deg_or_rad_series(data,"yaw","yaw_deg")

    vertices = create_box_vertices()

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111,projection="3d")

    def update(i):

        ax.cla()

        r = roll[i]
        p = pitch[i]
        y = yaw[i]

        R = rotation_matrix_from_rpy(r,p,y)

        rv = transform_vertices(vertices,R)
        faces = box_faces(rv)

        box = Poly3DCollection(faces,alpha=0.7,edgecolor="k")
        ax.add_collection3d(box)

        draw_world_axes(ax)
        draw_body_axes(ax,R)

        set_axes_equal(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # TITLE MOVED LOWER
        ax.set_title(
            f"3D IMU Orientation Viewer\n"
            f"t={times[i]:.2f}s  "
            f"roll={math.degrees(r):.1f}° "
            f"pitch={math.degrees(p):.1f}° "
            f"yaw={math.degrees(y):.1f}°",
            pad=30
        )

        ax.view_init(elev=20,azim=35)

    interval_ms = 50
    if len(times) > 1:
        avg_dt = np.mean(np.diff(times))
        interval_ms = max(20,int(1000*avg_dt))

    anim = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval_ms,
        repeat=False   # NO REPLAY
    )

    anim.save("imu_animation_with_cal_99.gif", fps=20)
    # anim.save("imu_animation_with_cal.gif", fps=20)
    # anim.save("imu_animation.gif", fps=20)

    # Save snapshots
    num_snapshots = 15   # change to any value from 10 to 20
    snapshot_dir = Path(f"{input_path.stem}_snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_indices = np.linspace(
        0, len(times) - 1, num=min(num_snapshots, len(times)), dtype=int
    )

    for k, idx in enumerate(snapshot_indices):
        draw_frame(ax, vertices, times, roll, pitch, yaw, idx)
        snapshot_path = snapshot_dir / f"{input_path.stem}_frame_{k+1:02d}.png"
        fig.savefig(snapshot_path, dpi=300, bbox_inches="tight")

    # print(f"Saved GIF: {gif_name}")
    print(f"Saved {len(snapshot_indices)} snapshots to: {snapshot_dir}")

    plt.tight_layout(rect=[0,0,1,0.93])  # leaves space for slider/time bar
    plt.show()


if __name__ == "__main__":
    main()
