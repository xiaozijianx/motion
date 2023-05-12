'''
Transformation matrix from WORLD to LOCAL frame

https://stackoverflow.com/questions/56166088/how-to-find-affine-transformation-matrix-between-two-sets-of-3d-points
DIM + 1 pairs of points
For 3d: 4 pairs of points that are not on the same face
'''
import numpy as np

def calcHomogeneousMatrix(ins, out):
  # cal culations
  l = len(ins)
  B = np.vstack([np.transpose(ins), np.ones(l)])
  D = 1.0 / np.linalg.det(B)
  entry = lambda r,d: np.linalg.det(np.delete(np.vstack([r, B]), (d+1), axis=0))
  M = [[(-1)**i * D * entry(R, i) for i in range(l)] for R in np.transpose(out)]
  A, t = np.hsplit(np.array(M), [l-1])
  t = np.transpose(t)[0]
  # output
  print("Affine transformation matrix:\n", A)
  print("Affine transformation translation vector:\n", t)
  
  # ############## unittests ###############
  print("TESTING:")
  for p, P in zip(np.array(ins), np.array(out)):
    image_p = np.dot(A, p) + t
    result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
    print(p, " mapped to: ", image_p, " ; expected: ", P, result)

  p = [1070.626, 771.101, -1456.376]
  image_p = np.dot(A, p) + t
  print(image_p)
  # ########################################
  return A, t


def transformHomogeneous(x):
  A = np.array([[ 1.00061956e-01,  1.82995294e-03, -3.01522338e-03],
                [ 3.41856654e-05,  1.00258480e-01, -2.13580600e-04],
                [-1.56972487e-03, -2.82733068e-03,  9.75262468e-02]])
  t = np.array([114.30024029, -75.50785398, 144.36331394])
  return np.dot(A, x) + t


if __name__ == "__main__":
  #################################
  # coordinate in WORLD frame
  ins = np.array([[-1200.55,	750.393,	-1477.82], 
                      [1012.11,	747.093,	-2672.74], 
                      [-1257.76,	1904.87,	-2675.71], 
                      [-1237.6,	747.783,	-2708.93]])
    
  # coordinate in LOCAL frame
  out = np.array([[0, 0, 0], 
                  [225, 0, -120], 
                  [0, 116, -120], 
                  [0, 0, -120]])
  #################################
  A, t = calcHomogeneousMatrix(ins, out)
  y = transformHomogeneous([1070.626, 771.101, -1456.376])
  print(y)
