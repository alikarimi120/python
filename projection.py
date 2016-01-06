import cv2
import cv2.cv as cv
import numpy as np
import math
import matplotlib.pyplot as plt

def quatmult(q1,q2):
  #quaternion multiplication
  out = [0,0,0,0]
  out[0] = (q1[0]*q2[0]) - (q1[1]*q2[1]) - (q1[2]*q2[2]) - (q1[3]*q2[3])
  out[1] = (q1[0]*q2[1]) + (q1[1]*q2[0]) + (q1[2]*q2[3]) - (q1[3]*q2[2])
  out[2] = (q1[0]*q2[2]) - (q1[1]*q2[3]) + (q1[2]*q2[0]) + (q1[3]*q2[1])
  out[3] = (q1[0]*q2[3]) + (q1[1]*q2[2]) - (q2[2]*q2[1]) + (q1[3]*q2[0])
  return out

def degtorad(deg):
  #Converts degrees to radians
  rad = ((math.pi)/180)*deg
  return rad

def rotation_quaternion(axis, angle):
  #axis is rotation unit vector axis
  #angle is angle of rotation in degrees
  rotation_quat = [0, 0, 0, 0]
  angle_rad = degtorad(angle/2)
  rotation_quat[0] = math.cos(angle_rad)
  rotation_quat[1] = math.sin(angle_rad)*axis[0]
  rotation_quat[2] = math.sin(angle_rad)*axis[1]
  rotation_quat[3] = math.sin(angle_rad)*axis[2]
  return rotation_quat

def rotate(point, axis, angle):
  #input is a input 3d point
  #axis is the rotation unit vector axis
  #angle is angle of rotation in degress
  input_quat = [0, 0, 0, 0]
  input_quat[1:] = point
  rotation_quat = rotation_quaternion(axis, angle)
  rotation_conj = [0, 0, 0, 0]
  rotation_conj[0] = rotation_quat[0]
  rotation_conj[1] = -rotation_quat[1]
  rotation_conj[2] = -rotation_quat[2]
  rotation_conj[3] = -rotation_quat[3]
  rotated_point = quatmult(quatmult(rotation_quat,input_quat),rotation_conj)
  return rotated_point

def quat2rot(q):
  #Returns rotational matrix of input quaternion
  rot_matrix = np.zeros([3,3])
  rot_matrix[0][0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
  rot_matrix[0][1] = 2*(q[1]*q[2] - q[0]*q[3])
  rot_matrix[0][2] = 2*(q[1]*q[3] + q[0]*q[2])
  rot_matrix[1][0] = 2*(q[1]*q[2] + q[0]*q[3])
  rot_matrix[1][1] = q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2
  rot_matrix[1][2] = 2*(q[2]*q[3] - q[0]*q[1])
  rot_matrix[2][0] = 2*(q[1]*q[3] - q[0]*q[2])
  rot_matrix[2][1] = 2*(q[2]*q[3] + q[0]*q[1])
  rot_matrix[2][2] = q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2
  return np.matrix(rot_matrix)

def orthographic_proj(point, camera_position, camera_axes):
  point_camera_frame = np.matrix(point-camera_position)
  camera_x_axis = np.matrix(camera_axes[0]).transpose()
  camera_y_axis = np.matrix(camera_axes[1]).transpose()
  projected_point = [0, 0]
  projected_point[0] = (point_camera_frame*camera_x_axis).item(0)
  projected_point[1] = (point_camera_frame*camera_y_axis).item(0)
  return projected_point

def perspective_proj(point, camera_position, camera_axes):
  point_camera_frame = np.matrix(point-camera_position)
  camera_x_axis = np.matrix(camera_axes[0]).transpose()
  camera_y_axis = np.matrix(camera_axes[1]).transpose()
  camera_optical_axis = np.matrix(camera_axes[2]).transpose()
  projected_point = [0, 0]
  projected_point[0] = ((point_camera_frame * camera_x_axis).item(0))/((point_camera_frame * camera_optical_axis).item(0))
  projected_point[1] = ((point_camera_frame * camera_y_axis).item(0))/((point_camera_frame * camera_optical_axis).item(0))
  return projected_point

def pts_set_2():
  #Function to create a wireframe with triangle
  def create_intermediate_points(pt1, pt2, granularity):
    new_pts = []
    vector = np.array([(x[0] - x[1]) for x in zip(pt1, pt2)])
    return [(np.array(pt2) + (vector * (float(i)/granularity))) for i in range(1, granularity)]

  pts = []
  granularity = 20

  # Create cube wireframe
  pts.extend([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], \
              [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])

  pts.extend(create_intermediate_points([-1, -1, 1], [1, -1, 1], granularity))
  pts.extend(create_intermediate_points([1, -1, 1], [1, 1, 1], granularity))
  pts.extend(create_intermediate_points([1, 1, 1], [-1, 1, 1], granularity))
  pts.extend(create_intermediate_points([-1, 1, 1], [-1, -1, 1], granularity))

  pts.extend(create_intermediate_points([-1, -1, -1], [1, -1, -1], granularity))
  pts.extend(create_intermediate_points([1, -1, -1], [1, 1, -1], granularity))
  pts.extend(create_intermediate_points([1, 1, -1], [-1, 1, -1], granularity))
  pts.extend(create_intermediate_points([-1, 1, -1], [-1, -1, -1], granularity))

  pts.extend(create_intermediate_points([1, 1, 1], [1, 1, -1], granularity))
  pts.extend(create_intermediate_points([1, -1, 1], [1, -1, -1], granularity))
  pts.extend(create_intermediate_points([-1, -1, 1], [-1, -1, -1], granularity))
  pts.extend(create_intermediate_points([-1, 1, 1], [-1, 1, -1], granularity))

  # Create triangle wireframe
  pts.extend([[-0.5, -0.5, -1], [0.5, -0.5, -1], [0, 0.5, -1]])
  pts.extend(create_intermediate_points([-0.5, -0.5, -1], [0.5, -0.5, -1], granularity))
  pts.extend(create_intermediate_points([0.5, -0.5, -1], [0, 0.5, -1], granularity))
  pts.extend(create_intermediate_points([0, 0.5, -1], [-0.5, -0.5, -1], granularity))

  return np.array(pts)

def main():
  pts = np.zeros([11,3])
  pts[0,:] = [-1,-1,-1]
  pts[1,:] = [1,-1,-1]
  pts[2,:] = [1,1,-1]
  pts[3,:] = [-1,1,-1]
  pts[4,:] = [-1,-1,1]
  pts[5,:] = [1,-1,1]
  pts[6,:] = [1,1,1]
  pts[7,:] = [-1,1,1]
  pts[8,:] = [-0.5,-0.5,-1]
  pts[9,:] = [0.5,-0.5,-1]
  pts[10,:] = [0,0.5,-1]
  pts = pts_set_2()

  camera_frames = []
  camera_frames.append([0,0,-5])
  camera_orientations = []
  camera_orientations.append(np.identity(3))
  rotation_quat = rotation_quaternion([0,1,0], 30)
  for i in range(1,4):
    #Calcuate the 4 positions and rotations of the camera
    camera_frames.append(rotate(camera_frames[i-1], [0,1,0], -30)[1:])
    camera_orientations.append(quat2rot(rotation_quat) * np.matrix(camera_orientations[i-1]))
  op_figure = plt.figure(1)
  pp_figure = plt.figure(2)
  for (frame, orientation, i) in zip(camera_frames, camera_orientations, range(1,5)):
    op = []
    pp = []
    for pt in pts:
      op.append(orthographic_proj(pt, frame, orientation))
      pp.append(perspective_proj(pt, frame, orientation))
    plt.figure(1)
    plt.subplot(2,2,i)
    plt.margins(0.1,0.1)
    plt.title("Frame {0}".format(i))
    plt.plot([x[0] for x in op],[y[1] for y in op], 'bo', markersize=2)
    plt.figure(2)
    plt.subplot(2,2,i)
    plt.margins(0.1,0.1)
    plt.title("Frame {0}".format(i))
    plt.plot([x[0] for x in pp],[y[1] for y in pp], 'bo', markersize=2)
  op_figure.suptitle("Orthographic Projection")
  op_figure.savefig('op.png')
  pp_figure.suptitle("Perspective Projection")
  pp_figure.savefig('pp.png')


if __name__ == '__main__':
  main()