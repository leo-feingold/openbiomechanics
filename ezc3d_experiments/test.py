import ezc3d
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib.animation import FuncAnimation
import os

#rcParams['font.family'] = 'monospace'

# define the skeletal connections
'''
['C7', 'CLAV', 'LANK', 'LASI', 'LBHD', 'LELB', 'LFHD', 'LFIN', 'LFRM',
'LHEE', 'LKNE', 'LMANK', 'LMELB', 'LMKNE', 'LPSI', 'LSHO',
'LTHI', 'LTIB', 'LTOE', 'LUPA', 'LWRA', 'LWRB', 'Marker1', 
'Marker10', 'Marker2', 'Marker3', 'Marker4', 'Marker5', 'Marker6', 
'Marker7', 'Marker8', 'Marker9', 'RANK', 'RASI', 'RBAK', 'RBHD', 
'RELB', 'RFHD', 'RFIN', 'RFRM', 'RHEE', 'RKNE', 'RMANK', 'RMELB', 
'RMKNE', 'RPSI', 'RSHO', 'RTHI', 'RTIB', 'RTOE', 'RUPA', 'RWRA', 
'RWRB', 'STRN', 'T10']

'''
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

def get_marker_index(labels, marker_name):
    return [i for i, label in enumerate(labels) if label == marker_name][0]

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

def update_plot(frame, points, marker_indices, skeletal_connections, scatter_points, lines, ax, num_frames):
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

    


    ax.set_title(f"Frame {frame} of {num_frames}")
    return scatter_points, *lines

def interpret_c3d_filename(file_path):
    # Extract the filename from the file path
    filename = os.path.basename(file_path)
    # Remove the file extension
    filename = os.path.splitext(filename)[0]
    
    # Split the filename into its components
    components = filename.split('_')
    
    # Interpret each component
    user_id = int(components[0])
    session_id = int(components[1])
    height = int(components[2])
    weight = int(components[3])
    side = components[4]
    swing_number = int(components[5])
    #exit_velo = float(components[6][:2] + '.' + components[6][2:])

    # Handle exit velocity
    exit_velo_str = components[6]
    if len(exit_velo_str) == 3:
        exit_velo = float(exit_velo_str[0] + exit_velo_str[1] + '.' + exit_velo_str[2])
    else:
        exit_velo = float(exit_velo_str[0] + exit_velo_str[1:3] + '.' + exit_velo_str[3])
    
    
    # Print the results
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print(f"Height: {height} inches")
    print(f"Weight: {weight} pounds")
    print(f"Side: {side}")
    print(f"Swing Number: {swing_number}")
    print(f"Exit Velocity: {exit_velo} mph")

    return user_id, session_id, height, weight, side, swing_number, exit_velo



def main(user_id, session_id, height, weight, side, swing_number, exit_velo, repo_root_path, approx_swing_init_frame=400):
    # load C3D file
    c = load_c3d_file(user_id, session_id, height, weight, side, swing_number, exit_velo, repo_root_path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    points = c["data"]["points"]
    
    # create a dictionary of marker indices for easy access
    marker_indices = {label: i for i, label in enumerate(labels)}

    # initialize plot
    fig, ax = init_plot()
    set_plot_limits(ax, points)
    plt.suptitle(f"User ID: {user_id}, Swing Number: {swing_number}, Height: {height} Inches, Weight: {weight} Pounds, Exit Velo: {exit_velo} MPH")

    # initialize scatter points
    scatter_points = ax.scatter([], [], [], alpha=0.75, color="royalblue")
    
    # initialize lines for skeletal connections
    lines = [ax.plot([], [], [], color='black')[0] for _ in skeletal_connections]

    ax.view_init(elev=30, azim=-70, roll=0)

    meta_data_marker = c['header']['points']
    num_frames = meta_data_marker['last_frame'] - meta_data_marker['first_frame'] 

    # Create the animation
    anim = FuncAnimation(fig, update_plot, frames=range(approx_swing_init_frame, num_frames-50), 
                         fargs=(points, marker_indices, skeletal_connections, scatter_points, lines, ax, num_frames), 
                         interval=10)

    # Save the animation to a .mp4 file
    #anim.save('test_swing_animation.mp4', writer='ffmpeg')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/c3d/000059/000059_000017_67_179_R_006_974.c3d" 
    #"/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/c3d/000080/000080_000282_71_188_R_001_1012.c3d"
    user_id, session_id, height, weight, side, swing_number, exit_velo = interpret_c3d_filename(file_path)
    repo_root_path = "/Users/leofeingold/Documents/GitHub/openbiomechanics"

    main(user_id, session_id, height, weight, side, swing_number, exit_velo, repo_root_path)

