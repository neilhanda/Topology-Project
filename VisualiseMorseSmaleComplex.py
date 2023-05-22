import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

def meshfunction(point1, point2, point3, color):
    # Define the number of points for the patch
    n = 100
    # Define the vertices of the patch
    verts = []
    for i in range(n+1):
        for j in range(n+1-i):
            alpha = i/n
            beta = j/n
            gamma = 1 - alpha - beta
            point = np.array(point1)*alpha + np.array(point2)*beta + np.array(point3)*gamma
            point[2] = f(point[0], point[1])
            verts.append(point)
    # Define the faces of the patch
    faces = []
    for i in range(n):
        for j in range(n-i):
            k = (n+1)*i + j
            faces.append([k, k+1, k+n+2-i])
            faces.append([k, k+n+2-i, k+n+1-i])

    # Create a triangle mesh between the three points
    tri_mesh = Poly3DCollection(verts=[verts], alpha=1, facecolor=color, linewidths=0)
    # Add the triangle mesh to the plot
    ax.add_collection3d(tri_mesh)


# Define the function to plot
def f(x, y):
    return np.sin(x) * np.sin(y)

# Generate x and y values
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Compute the function values for each x and y pair
Z = f(X, Y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# set the elevation angle to 90 degrees - VIEW FROM TOP
ax.view_init(elev=90)

# Create a triangulation of the surface
tri = Triangulation(X.flatten(), Y.flatten())

# Plot the surface with different colors for each triangle
#surf = ax.plot_trisurf(tri, Z.flatten(), cmap='viridis')
surf = ax.plot_trisurf(tri, Z.flatten(), shade=False, color='gray', alpha=0.4)
#surf = ax.plot_trisurf(tri, Z.flatten(), cmap='viridis', alpha=0.7, linewidth=0.2, edgecolor='gray', antialiased=True, shade=False)


# Add labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Morse Smale Complex Visualisation')

#----------------------------------------------------------------------------------
point1 = [-4.7115, -1.5384, f(-4.7115,-1.5384)]
point2 = [-3.173, 0, f(-3.173, 0)]
point3 = [-3.173, -3.173, f(-3.173,-3.173)]
# point4 = [-1.5384, -1.5384, f(-1.5384, -1.5384)]

meshfunction(point1, point2, point3, 'red')

point1 = [-3.173, 0, f(-3.173, 0)]
point2 = [-3.173, -3.173, f(-3.173,-3.173)]
point3 = [-1.5384, -1.5384, f(-1.5384, -1.5384)]

meshfunction(point1, point2, point3, 'red')
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
point1 = [-4.7115, 1.5384, f(-4.7115, 1.5384)]
point2 = [-3.173, 0, f(-3.173, 0)]
point3 = [-3.173, 3.173, f(-3.173, 3.173)]

meshfunction(point1, point2, point3, 'blue')

point1 = [-3.173, 0, f(-3.173, 0)]
point2 = [-3.173, 3.173, f(-3.173, 3.173)]
point3 = [-1.5384, 1.5384, f(-1.5384, 1.5384)]

meshfunction(point1, point2, point3, 'blue')
#----------------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [-3.173, 0, f(-3.173, 0)]
point2 = [-1.5384, -1.5384, f(-1.5384, -1.5384)]
point3 = [-1.5384, 1.5384, f(-1.5384, 1.5384)]

meshfunction(point1, point2, point3, 'green')

point1 = [-1.5384, -1.5384, f(-1.5384, -1.5384)]
point2 = [-1.5384, 1.5384, f(-1.5384, 1.5384)]
point3 = [0, 0, f(0,0)]

meshfunction(point1, point2, point3, 'green')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [-3.173, -3.173, f(-3.173, -3.173)]
point2 = [-1.5384, -4.7115, f(-1.5384, -4.7115)]
point3 = [-1.5384, -1.5384, f(-1.5384,-1.5384)]

meshfunction(point1, point2, point3, 'yellow')

point1 = [-1.5384, -4.7115, f(-1.5384, -4.7115)]
point2 = [-1.5384, -1.5384, f(-1.5384,-1.5384)]
point3 = [0, -3.173, f(0,-3.173)]

meshfunction(point1, point2, point3, 'yellow')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [-3.173, 3.173, f(-3.173, 3.173)]
point2 = [-1.5384, 1.5348, f(-1.5384, 1.5348)]
point3 = [-1.5384, 4.7115, f(-1.5384,4.7115)]

meshfunction(point1, point2, point3, 'pink')

point1 = [0, 3.173, f(0,3.173)]
point2 = [-1.5384, 1.5348, f(-1.5384, 1.5348)]
point3 = [-1.5384, 4.7115, f(-1.5384,4.7115)]

meshfunction(point1, point2, point3, 'pink')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [-1.5384, -1.5348, f(-1.5384, -1.5348)]
point2 = [0, 0, f(0, 0)]
point3 = [0, -3.173, f(0, -3.173)]

meshfunction(point1, point2, point3, 'orange')

point1 = [0, 0, f(0, 0)]
point2 = [0, -3.173, f(0, -3.173)]
point3 = [1.5384, -1.5348, f(1.5384, -1.5348)]

meshfunction(point1, point2, point3, 'orange')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [-1.5384, 1.5348, f(-1.5384, 1.5348)]
point2 = [0, 0, f(0, 0)]
point3 = [0, 3.173, f(0, 3.173)]

meshfunction(point1, point2, point3, 'purple')

point1 = [0, 0, f(0, 0)]
point2 = [0, 3.173, f(0, 3.173)]
point3 = [1.5384, 1.5348, f(1.5384, 1.5348)]

meshfunction(point1, point2, point3, 'purple')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [0, -3.173, f(0, -3.173)]
point2 = [1.5384, -4.7115, f(1.5384, -4.7115)]
point3 = [1.5384, -1.5384, f(1.5384, -1.5384)]


meshfunction(point1, point2, point3, 'cyan')

point1 = [1.5384, -4.7115, f(1.5384, -4.7115)]
point2 = [1.5384, -1.5384, f(1.5384, -1.5384)]
point3 = [3.173, -3.173, f(3.173,-3.173)]

meshfunction(point1, point2, point3, 'cyan')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [0, 0, f(0, 0)]
point2 = [1.5384, -1.5348, f(1.5384, -1.5348)]
point3 = [1.5384, 1.5348, f(1.5384, 1.5348)]


meshfunction(point1, point2, point3, 'gold')

point1 = [1.5384, -1.5348, f(1.5384, -1.5348)]
point2 = [1.5384, 1.5348, f(1.5384, 1.5348)]
point3 = [3.173, 0, f(3.173,0)]

meshfunction(point1, point2, point3, 'gold')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [0, 3.173, f(0, 3.173)]
point2 = [1.5384, 1.5348, f(1.5384, 1.5348)]
point3 = [1.5384, 4.7115, f(1.5384, 4.7115)]


meshfunction(point1, point2, point3, 'red')

point1 = [1.5384, 1.5348, f(1.5384, 1.5348)]
point2 = [1.5384, 4.7115, f(1.5384, 4.7115)]
point3 = [3.173, 3.173, f(3.173,3.173)]

meshfunction(point1, point2, point3, 'red')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [1.5384, -1.5348, f(1.5384, -1.5348)]
point2 = [3.173, -3.173, f(3.173,-3.173)]
point3 = [3.173, 0, f(3.173, 0)]


meshfunction(point1, point2, point3, 'purple')

point1 = [3.173, -3.173, f(3.173,-3.173)]
point2 = [3.173, 0, f(3.173, 0)]
point3 = [4.7115, -1.5384, f(4.7115, -1.5384)]

meshfunction(point1, point2, point3, 'purple')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
point1 = [1.5384, 1.5348, f(1.5384, 1.5348)]
point2 = [3.173, 0, f(3.173, 0)]
point3 = [3.173, 3.173, f(3.173,3.173)]


meshfunction(point1, point2, point3, 'blue')

point1 = [3.173, 0, f(3.173, 0)]
point2 = [3.173, 3.173, f(3.173,3.173)]
point3 = [4.7115, 1.5384, f(4.7115, 1.5384)]

meshfunction(point1, point2, point3, 'blue')
#------------------------------------------------------------------------

# # Define the rotation angles
# angles = range(0, 360, 5)  # Change the step size (5 degrees in this example) as desired

# # Define the update function for the animation
# def update(angle):
#     ax.view_init(elev=75, azim=angle)  # Change the elevation (elev) and azimuth (azim) angles as desired

# # Create the animation
# animation = FuncAnimation(fig, update, frames=angles, interval=1)  # Adjust the interval (in milliseconds) as desired

# Show the plot
plt.show()