import ezc3d
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib.animation import FuncAnimation

rcParams['font.family'] = 'monospace'



# The c3d file naming convention is detailed in the baseball_hitting README. 
# Each file is named according to the following values.

user_id = 103
session_id = 391
height = 73        # inches
weight = 183       # pounds
side = "R"
swing_number = 16
exit_velo = 97.7

obm_repo_root_path = "/Users/leofeingold/Documents/GitHub/openbiomechanics"
c3d_files_path = obm_repo_root_path + "/baseball_hitting/data/c3d"
c3d_file_path = c3d_files_path + f"/{user_id:06}/{user_id:06}_{session_id:06}_{height}_{weight}_{side}_{swing_number:03}_{str(exit_velo).replace('.', '')}.c3d"

# alternatively can just copy path of any specific file in baseball_hitting, data, c3d


# create the c3d object
c = ezc3d.c3d(c3d_file_path)
print(c["header"]["points"])



# labels of each marker
labels = c["parameters"]["POINT"]["LABELS"]["value"]
print(labels)




points = c["data"]["points"]
num_markers = c['header']['points']['size']
print(num_markers)

# the shape is 4Xnum_markersXnum_frames
print(points.shape)




marker5_index = [i for i, label in enumerate(labels) if label == "Marker5"][0]
approx_swing_init_frame = 550

# calculate the min and max coordiante point of each dimension
x_min, x_max = np.min(points[0]), np.max(points[0])
y_min, y_max = np.min(points[1]), np.max(points[1])
z_min, z_max = np.min(points[2]), np.max(points[2])

# Set up the figure and axis
fig = plt.figure(figsize=(8, 6))

# 111 effectively creates a single subplot that spans the entire figure
ax = fig.add_subplot(111, projection='3d')

# no labels on the axis
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# ensure the whole swing fits in the plot
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_zlim([z_min, z_max])

# set the title of the plot
plt.suptitle(f"User ID: {user_id}, Swing Number: {swing_number}, Exit Velo: {exit_velo}")


# initialize the scatter plots
marker_points = ax.scatter([], [], [], alpha=0.75, color="royalblue")
barrel_points = ax.scatter([], [], [], color="darkgoldenrod", alpha=0.5)

# set the view angle of the 3D plot
ax.view_init(elev=30, azim=-70, roll=0)

# function to update the scatter plots
def update(frame):
    x = points[0, :, frame]
    y = points[1, :, frame]
    z = points[2, :, frame]

    # updates the positions of the scatter plot points to the new coordinates
    marker_points._offsets3d = (x, y, z)

    barrel_x = points[0, marker5_index, approx_swing_init_frame:frame]
    barrel_y = points[1, marker5_index, approx_swing_init_frame:frame]
    barrel_z = points[2, marker5_index, approx_swing_init_frame:frame]

    # updates the positions of the scatter plot points to the new coordinates
    barrel_points._offsets3d = (barrel_x, barrel_y, barrel_z)

    # update the title
    ax.set_title(f"Frame {frame} of 768")

    # set the view angle of the 3D plot
    #ax.view_init(elev=30, azim=-55, roll=0)
    return marker_points, barrel_points

# create the animation
anim = FuncAnimation(fig, update, frames=range(approx_swing_init_frame-300, 768-60), interval=10)
#anim = FuncAnimation(fig, update, frames=range(0, 768), interval=100)

# save the animation to a .mp4 file
#anim.save('test_swing_animation.mp4', writer='ffmpeg')

plt.show()