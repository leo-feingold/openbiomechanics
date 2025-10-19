import ezc3d
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib.animation import FuncAnimation
import os

#rcParams['font.family'] = 'monospace'

# Define the skeletal connections
skeletal_connections = [
    ("LFHD", "RFHD"), ("LBHD", "RBHD"), ("LFHD", "LBHD"), ("RFHD", "RBHD"),  # Head
    ("CLAV", "LUPA"), ("CLAV", "RUPA"),  # Clavicle to upper arms
    ("LUPA", "LELB"), ("RUPA", "RELB"),  # Upper arms to elbows
    ("LELB", "LWRA"), ("RELB", "RWRA"),  # Elbows to wrists
    ("LWRA", "LFIN"), ("RWRA", "RFIN"),  # Wrists to fingers
    ("CLAV", "C7"), ("C7", "T10"), ("T10", "RPSI"), ("T10", "LPSI"),  # Spine
    ("RPSI", "RASI"), ("LPSI", "LASI"),  # Pelvis
    ("RASI", "RTHI"), ("LASI", "LTHI"),  # Pelvis to thighs
    ("RTHI", "RKNE"), ("LTHI", "LKNE"),  # Thighs to knees
    ("RKNE", "RTIB"), ("LKNE", "LTIB"),  # Knees to tibias
    ("RTIB", "RANK"), ("LTIB", "LANK"),  # Tibias to ankles
    ("RANK", "RTOE"), ("LANK", "LTOE"),  # Ankles to toes
    ("RHEE", "RMANK"), ("LHEE", "LMANK"),
    ("RHEE", "RANK"), ("LHEE", "LANK"),
    ("RTOE", "RMANK"), ("LTOE", "LMANK"),
    ("RMANK", "RMKNE"), ("LMANK", "LMKNE"),
]

def load_c3d_file(user_id, session_id, height, weight, side, swing_number, exit_velo, repo_root_path):
    c3d_files_path = f"{repo_root_path}/baseball_hitting/data/c3d"
    c3d_file_path = f"{c3d_files_path}/{user_id:06}/{user_id:06}_{session_id:06}_{height}_{weight}_{side}_{swing_number:03}_{str(exit_velo).replace('.', '')}.c3d"
    return ezc3d.c3d(c3d_file_path)

def init_plot():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return fig, ax

def set_plot_limits(ax, points):
    x_min, x_max = np.min(points[0]), np.max(points[0])
    y_min, y_max = np.min(points[1]), np.max(points[1])
    z_min, z_max = np.min(points[2]), np.max(points[2])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

def update_plot(frame, points, marker_indices, skeletal_connections, scatter_points, lines, bat_points, ax, num_frames):
    x = points[0, :, frame]
    y = points[1, :, frame]
    z = points[2, :, frame]

    # Update scatter points
    scatter_points._offsets3d = (x, y, z)

    # Update lines
    for line, (start_marker, end_marker) in zip(lines, skeletal_connections):
        start_idx = marker_indices[start_marker]
        end_idx = marker_indices[end_marker]
        line.set_data([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]])
        line.set_3d_properties([z[start_idx], z[end_idx]])

    # Update bat points
    bat_x = [x[marker_indices[marker]] for marker in marker_indices if "marker" in marker.lower()]
    bat_y = [y[marker_indices[marker]] for marker in marker_indices if "marker" in marker.lower()]
    bat_z = [z[marker_indices[marker]] for marker in marker_indices if "marker" in marker.lower()]

    bat_points._offsets3d = (bat_x, bat_y, bat_z)

    ax.set_title(f"Frame {frame} of {num_frames}")
    return scatter_points, *lines, bat_points

def interpret_c3d_filename(file_path):
    filename = os.path.basename(file_path)
    filename = os.path.splitext(filename)[0]
    components = filename.split('_')
    user_id = int(components[0])
    session_id = int(components[1])
    height = int(components[2])
    weight = int(components[3])
    side = components[4]
    swing_number = int(components[5])
    exit_velo_str = components[6]
    if len(exit_velo_str) == 3:
        exit_velo = float(exit_velo_str[0] + exit_velo_str[1] + '.' + exit_velo_str[2])
    else:
        exit_velo = float(exit_velo_str[0] + exit_velo_str[1:3] + '.' + exit_velo_str[3])
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print(f"Height: {height} inches")
    print(f"Weight: {weight} pounds")
    print(f"Side: {side}")
    print(f"Swing Number: {swing_number}")
    print(f"Exit Velocity: {exit_velo} mph")
    return user_id, session_id, height, weight, side, swing_number, exit_velo

def main(user_id, session_id, height, weight, side, swing_number, exit_velo, repo_root_path, approx_swing_init_frame=400):
    c = load_c3d_file(user_id, session_id, height, weight, side, swing_number, exit_velo, repo_root_path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    points = c["data"]["points"]
    marker_indices = {label: i for i, label in enumerate(labels)}

    fig, ax = init_plot()
    set_plot_limits(ax, points)
    plt.suptitle(f"User ID: {user_id}, Swing Number: {swing_number}, Height: {height} Inches, Weight: {weight} Pounds, Exit Velo: {exit_velo} MPH")
    
    scatter_points = ax.scatter([], [], [], alpha=0.75, color="royalblue")
    lines = [ax.plot([], [], [], color='black')[0] for _ in skeletal_connections]
    bat_points = ax.scatter([], [], [], alpha=0.75, color="brown")
    ax.view_init(elev=30, azim=-70, roll=0)


    meta_data_marker = c['header']['points']
    num_frames = meta_data_marker['last_frame'] - meta_data_marker['first_frame']

    anim = FuncAnimation(fig, update_plot, frames=range(approx_swing_init_frame, num_frames-50), 
                         fargs=(points, marker_indices, skeletal_connections, scatter_points, lines, bat_points, ax, num_frames), 
                         interval=10)
    # anim.save('test_swing_animation.mp4', writer='ffmpeg')
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/c3d/000059/000059_000017_67_179_R_006_974.c3d" 
    user_id, session_id, height, weight, side, swing_number, exit_velo = interpret_c3d_filename(file_path)
    repo_root_path = "/Users/leofeingold/Documents/GitHub/openbiomechanics"
    main(user_id, session_id, height, weight, side, swing_number, exit_velo, repo_root_path)
