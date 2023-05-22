# Necessary import for our code
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# Define the function to plot
def f(x,y,choice):
    if choice==1:
        return np.sin(x) * np.sin(y)
    elif choice==2:
        return np.sin(x) + np.sin(y) 
    elif choice==3:
        return np.sin(x)*np.sin(y) + np.cos(x+y)
    elif choice==4:
        return np.cos(x)*np.sin(y) + 0.2*(x+y)

# Generate x and y values
x = np.linspace(-5, 5,105)
y = np.linspace(-5, 5,105)
X, Y = np.meshgrid(x, y)


# Prompt to choose a function
print("Please choose one of the following functions for which you want to plot a Morse Smale Complex:")
print("Choice 1 - sin(x)*sin(y)")
print("Choice 2 - sin(x)+sin(y)")
print("Choice 3 - sin(x)*sin(y) + cos(x+y)")
print("Choice 4 - np.cos(x)np.sin(y) + 0.2(x+y)")
print("Note that first 3 functions have been given by sir and 4th one is from Sir's Research Paper")

# Get the user's choice
choice = int(input("Enter your choice: \n"))
till=0

# Choosing appropriate values for while loop
if choice == 1:
    till=20
elif choice == 2:
    till=25
elif choice == 3:
    till=25
elif choice == 4:
    till=25
else:
    print("Invalid choice, please try again. Exiting program now")
    exit()

# Compute the function values for each x and y pair
print("Please wait while we generate Morse Smale Complex for given function\n")
Z = f(X,Y,choice)


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)

# set the elevation angle to 90 degrees - VIEW FROM TOP
ax.view_init(elev=90)

# Create a triangulation of the surface
tri = Triangulation(X.flatten(), Y.flatten())

# Plot the surface
surf = ax.plot_trisurf(tri, Z.flatten(), shade=False, color='gray', alpha=0.4)

# Add labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Plot of the given function')

# Compute the gradient of the function
gradx, grady = np.gradient(Z)

# Compute the Hessian matrix of the function
Hxx, Hxy = np.gradient(gradx)
Hyx, Hyy = np.gradient(grady)

# Compute the determinant and trace of the Hessian at each point
detH = Hxx * Hyy - Hxy * Hyx
trH = Hxx + Hyy

# Initialize arrays to store critical points
maxima = []
minima = []
saddle = []

# Checking for Gradient- 0.005 or 0.01 - Because MATPLOTLIB is not accurate 
# Discussed with sir- finite accuracy machine

if choice==4:
    slopecheck=0.01
else:
    slopecheck=0.005

# Classify each point as a maximum, minimum, or saddle point based on the sign of detH and trH
for i in range(len(x)):
    for j in range(len(y)):
        if abs(gradx[i,j]) < slopecheck and abs(grady[i,j]) < slopecheck:
            # For saddle point, we have slope as 0 and then checking Hessian
            if detH[i,j] < 0:
                saddle.append([X[i,j], Y[i,j], Z[i,j]]) # saddle point
                matrix = np.array([[Hxx[i,j],Hxy[i,j]],[Hyx[i,j],Hyy[i,j]]])
                
                # eigenVectors are returned as columns of eigen vector Matrix
                eigenValues2, eigenvectors = np.linalg.eig(matrix)

                # Current values of x and y
                cur_x= X[i,j]
                cur_y= Y[i,j]
                cnt=0
                while(cnt<=till):

                    # v1 is the eigen vector (still unknown if its pointing towards maxima or minima)
                    v1 = eigenvectors[:,0]
                    v2 = eigenvectors[:,1]
                    v3 = -v1
                    v4 = -v2
                    # Create a list of the arrays
                    arrays = [v1, v2, v3, v4]
                    p = [cur_x, cur_y]
                    #-----------------For the maxima eigen vectors---------------------
                    max_diff = -np.inf
                    max_array = None
                    # Iterate through the arrays
                    for array in arrays:
                        # Compute the diff for the current array
                        diff = f(p[0] + array[0], p[1] + array[1],choice) - f(p[0], p[1],choice)
                        # Update the maximum diff and corresponding array if necessary
                        if diff > max_diff:
                            max_diff = diff
                            max_array = array

                    # Case when v1 is max
                    if(np.array_equal(max_array, v1)):
                        # Going ahead
                        cur_x+= v1[0]/10
                        cur_y+= v1[1]/10
                        # Plotting the point for Morse Smale Complex
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        # Updating Hessian and Eigenvectors
                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))

                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)

                    # Case when v2 is max                    
                    elif(np.array_equal(max_array, v2)):
                        # Going ahead
                        cur_x+= v2[0]/10
                        cur_y+= v2[1]/10
                        # Plotting for morse smale complex
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        # Updating Hessian and Eigenvectors
                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))
                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v2[0]], [v2[1]], Z[ix,iy])

                    # Case when v3 is max
                    elif(np.array_equal(max_array, v3)):
                        # Going ahead / updating value of current point
                        cur_x+= v3[0]/10
                        cur_y+= v3[1]/10
                        # Plotting for morse smale complex
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        # Updating Hessian and Eigenvectors
                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))

                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v3[0]], [v3[1]], Z[ix,iy])

                    # Case when v4 is max
                    elif(np.array_equal(max_array, v4)):
                        # Going ahead / updating value of current point
                        cur_x+= v4[0]/10
                        cur_y+= v4[1]/10
                        # Plotting for morse smale complex
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        # Updating Hessian and Eigenvectors
                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))
                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v4[0]], [v4[1]], Z[ix,iy])

                    cnt+=1
                    
                # NOW WE WILL DO FOR MINIMA (same logic as above so repeated comments not added)
                
                # Resetting current points
                cur_x= X[i,j]
                cur_y= Y[i,j]
                cnt=0
                while(cnt<=till):

                    # v1 is the eigen vector (still unknown if its pointing towards maxima or minima)
                    v1 = eigenvectors[:,0]
                    v2 = eigenvectors[:,1]
                    v3 = -v1
                    v4 = -v2
                    # Create a list of the arrays
                    arrays = [v1, v2, v3, v4]
                    p = [cur_x, cur_y]
                    #-----------------For the minima eigen vectors---------------------
                    min_diff = np.inf
                    min_array = None
                    # Iterate through the arrays
                    for array in arrays:
                        # Compute the diff for the current array
                        diff = f(p[0] + array[0], p[1] + array[1],choice) - f(p[0], p[1],choice)
                        # Update the maximum diff and corresponding array if necessary
                        if diff < min_diff:
                            min_diff = diff
                            min_array = array

                    if(np.array_equal(min_array, v1)):
                        
                        cur_x+= v1[0]/10
                        cur_y+= v1[1]/10
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))

                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v1[0]], [v1[1]], Z[ix,iy])

                        
                    elif(np.array_equal(min_array, v2)):
                        
                        cur_x+= v2[0]/10
                        cur_y+= v2[1]/10
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))
                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v2[0]], [v2[1]], Z[ix,iy])


                    elif(np.array_equal(min_array, v3)):

                        cur_x+= v3[0]/10
                        cur_y+= v3[1]/10
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))

                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v3[0]], [v3[1]], Z[ix,iy])

                    elif(np.array_equal(min_array, v4)):

                        cur_x+= v4[0]/10
                        cur_y+= v4[1]/10
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))
                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v4[0]], [v4[1]], Z[ix,iy])

                    cnt+=1

                # LOGIC FOR 2nd MAXIMA AND 2nd MINIMA
                # Since the 2nd maxima and 2nd minima are opposite the the previous eigen vectors
                #, we simply reverse it for the first iteration and then continue finding the eigen-vectors
                # In this way, we can build the morse smale complex

                # NOW WE WILL DO FOR 2nd Maxima (opposite to first maxima) (same logic as above so repeated comments not added)
                
                # Resetting current points
                cur_y= Y[i,j]
                cur_x= X[i,j]
                matrix = np.array([[Hxx[i,j],Hxy[i,j]],[Hyx[i,j],Hyy[i,j]]])
                
                # eigenVectors are returned as columns of eigen vector Matrix
                eigenValues2, eigenvectors = np.linalg.eig(matrix)
                cnt=0
                while(cnt<=till):
                    # print(cnt)
                    # print(cur_x, cur_y)

                    # v1 is the eigen vector (still unknown if its pointing towards maxima or minima)
                    v1 = eigenvectors[:,0]
                    v2 = eigenvectors[:,1]
                    v3 = -v1
                    v4 = -v2
                    # Create a list of the arrays
                    arrays = [v1, v2, v3, v4]
                    p = [cur_x, cur_y]
                    #-----------------For the anti-maxima eigen vectors---------------------
                    max_diff = -np.inf
                    max_array = None
                    # Iterate through the arrays
                    for array in arrays:
                        # Compute the diff for the current array
                        diff = f(p[0] + array[0], p[1] + array[1],choice) - f(p[0], p[1],choice)
                        # Update the maximum diff and corresponding array if necessary
                        if diff > max_diff:
                            max_diff = diff
                            max_array = array

                    if(np.array_equal(max_array, v1)):
                        
                        if(cnt == 0):
                            cur_x+= v3[0]/10
                            cur_y+= v3[1]/10
                        else:
                            cur_x+= v1[0]/10
                            cur_y+= v1[1]/10

                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))

                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v1[0]], [v1[1]], Z[ix,iy])

                        
                    elif(np.array_equal(max_array, v2)):
                        
                        if(cnt == 0):
                            cur_x+= v4[0]/10
                            cur_y+= v4[1]/10
                        else:
                            cur_x+= v2[0]/10
                            cur_y+= v2[1]/10
                        
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))
                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v2[0]], [v2[1]], Z[ix,iy])


                    elif(np.array_equal(max_array, v3)):

                        if(cnt == 0):
                            cur_x+= v1[0]/10
                            cur_y+= v1[1]/10
                        else:
                            cur_x+= v3[0]/10
                            cur_y+= v3[1]/10
                        
                        
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))

                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v3[0]], [v3[1]], Z[ix,iy])

                    elif(np.array_equal(max_array, v4)):

                        if(cnt == 0):
                            cur_x+= v2[0]/10
                            cur_y+= v2[1]/10
                        else:
                            cur_x+= v4[0]/10
                            cur_y+= v4[1]/10
                        
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))
                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v4[0]], [v4[1]], Z[ix,iy])

                    cnt+=1

                # NOW WE WILL DO FOR 2nd Minima (opposite to first minima() (same logic as above so repeated comments not added)
                
                # Resetting current points
                cur_x= X[i,j]
                cur_y= Y[i,j]
                matrix = np.array([[Hxx[i,j],Hxy[i,j]],[Hyx[i,j],Hyy[i,j]]])
                
                # eigenVectors are returned as columns of eigen vector Matrix
                eigenValues2, eigenvectors = np.linalg.eig(matrix)
                cnt=0
                while(cnt<=till):

                    # v1 is the eigen vector (still unknown if its pointing towards maxima or minima)
                    v1 = eigenvectors[:,0]
                    v2 = eigenvectors[:,1]
                    v3 = -v1
                    v4 = -v2
                    # Create a list of the arrays
                    arrays = [v1, v2, v3, v4]
                    p = [cur_x, cur_y]
                    #-----------------For the anti-minima eigen vectors---------------------
                    min_diff = np.inf
                    min_array = None
                    # Iterate through the arrays
                    for array in arrays:
                        # Compute the diff for the current array
                        diff = f(p[0] + array[0], p[1] + array[1],choice) - f(p[0], p[1],choice)
                        # Update the maximum diff and corresponding array if necessary
                        if diff < min_diff:
                            min_diff = diff
                            min_array = array

                    if(np.array_equal(min_array, v1)):

                        if(cnt == 0):
                            cur_x+= v3[0]/10
                            cur_y+= v3[1]/10
                        else:
                            cur_x+= v1[0]/10
                            cur_y+= v1[1]/10
                        
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))

                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v1[0]], [v1[1]], Z[ix,iy])

                        
                    elif(np.array_equal(min_array, v2)):
                        
                        if(cnt == 0):
                            cur_x+= v4[0]/10
                            cur_y+= v4[1]/10
                        else:
                            cur_x+= v2[0]/10
                            cur_y+= v2[1]/10
                        
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))
                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v2[0]], [v2[1]], Z[ix,iy])


                    elif(np.array_equal(min_array, v3)):

                        if(cnt == 0):
                            cur_x+= v1[0]/10
                            cur_y+= v1[1]/10
                        else:
                            cur_x+= v3[0]/10
                            cur_y+= v3[1]/10
                        
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))

                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v3[0]], [v3[1]], Z[ix,iy])

                    elif(np.array_equal(min_array, v4)):

                        if(cnt == 0):
                            cur_x+= v2[0]/10
                            cur_y+= v2[1]/10
                        else:
                            cur_x+= v4[0]/10
                            cur_y+= v4[1]/10
                        
                        ax.scatter(cur_x, cur_y, f(cur_x, cur_y,choice))

                        ix = np.argmin(np.abs(x - cur_x))
                        iy = np.argmin(np.abs(y - cur_y))
                        matrix = np.array([[Hxx[ix,iy],Hxy[ix,iy]],[Hyx[ix,iy],Hyy[ix,iy]]])
                        # eigenVectors are returned as columns of eigen vector Matrix
                        eigenValues2, eigenvectors = np.linalg.eig(matrix)
                        # ax.quiver(X[ix,iy], Y[ix,iy], Z[ix,iy], [v4[0]], [v4[1]], Z[ix,iy])

                    cnt+=1

                
            elif detH[i,j] > 0 and trH[i,j] > 0:
                minima.append([X[i,j], Y[i,j], Z[i,j]]) # minimum
            elif detH[i,j] > 0 and trH[i,j] < 0:
                maxima.append([X[i,j], Y[i,j], Z[i,j]]) # maximum

#NORMAL ARRAYS
maximas = maxima
minimas = minima
saddles = saddle

# Convert arrays to numpy arrays for easier manipulation
maxima = np.array(maxima)
minima = np.array(minima)
saddle = np.array(saddle)

# Plot critical points
ax.scatter(maxima[:,0], maxima[:,1], maxima[:,2], color='b', s=100, label='Maximum')
ax.scatter(minima[:,0], minima[:,1], minima[:,2], color='g', s=100, label='Minimum')
ax.scatter(saddle[:,0], saddle[:,1], saddle[:,2], color='r', s=100, label='Saddle')

# Add legend
ax.legend()

# Define the rotation angles
# angles = range(0, 360, 5)  # Change the step size (5 degrees in this example) as desired

# # Define the update function for the animation
# def update(angle):
#     ax.view_init(elev=75, azim=angle)  # Change the elevation (elev) and azimuth (azim) angles as desired

# # Create the animation
# animation = FuncAnimation(fig, update, frames=angles, interval=1)  # Adjust the interval (in milliseconds) as desired

# Show the plot
plt.show()

