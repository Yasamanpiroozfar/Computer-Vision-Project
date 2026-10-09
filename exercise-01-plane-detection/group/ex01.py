import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter, median_filter
import scipy.ndimage as ndi

from pathlib import Path

data04 = loadmat(Path(__file__).resolve().parent / 'data' / 'example4kinect.mat')

A04=data04['amplitudes4']
D04=data04['distances4']
PC04=data04['cloud4']
X04=PC04[:,:,0]
Y04=PC04[:,:,1]
Z04=PC04[:,:,2]


valid_mask = Z04 != 0
points = PC04[valid_mask]


#eine Ebene wird durch die Normalenvektor n und den Abstand d von der Ursprungsebene definiert.

def plane_from_3_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-8:
        return None, None
    d = np.dot(n, p1)
    return n, d

# die Funktion compute_inliers berechnet die Abstände der Punkte von der Ebene und gibt eine Maske zurück, 
# die angibt, welche Punkte innerhalb des Schwellenwerts liegen.

def compute_inliers(points, plane, threshold):
    n, d = plane
    distances = np.abs(np.dot(points, n) - d) / np.linalg.norm(n)
    return distances < threshold

# die Funktion ransac_plane implementiert den RANSAC-Algorithmus, um die beste Ebene zu finden, die die meisten Inlier-Punkte enthält.

def ransac_plane(points, threshold=2, max_iterations=3):
    max_inliers = None
    best_model = None
    best_count = 0

    n_points = points.shape[0]

    for i in range(max_iterations):
        idx = np.random.choice(n_points, size=3, replace=False)
        p1, p2, p3 = points[idx]

        model = plane_from_3_points(p1, p2, p3)

        if model is None:
            continue

        inlier_mask = compute_inliers(points, model, threshold)
        count = np.sum(inlier_mask)

        if count > best_count:
            best_count = count
            best_model = model
            max_inliers = inlier_mask

    return best_model, max_inliers, best_count


#Bodenebene erkennen mit RANSAC

boden_model, boden_inliers_bool, boden_count = ransac_plane(points, threshold=0.008, max_iterations=1000)

print("Boden inliers:", boden_count)

boden_mask = np.zeros(PC04.shape[:2], dtype=bool)
boden_mask[valid_mask] = boden_inliers_bool


# die morphologischen Operationen helfen, die Bodenmaske zu verbessern, indem sie kleine Löcher füllen und kleine Objekte entfernen, die nicht zum Boden gehören.
boden_mask_filtered = ndi.binary_closing(boden_mask, structure=np.ones((5, 5)))
boden_mask_filtered = ndi.binary_opening(boden_mask_filtered, structure=np.ones((5, 5)))

# uniform_filter habe ich auch probiert, aber es war nicht so gut wie die morphologischen Operationen, da sie die Form der Bodenmaske besser erhalten.
boden_uniform = ndi.uniform_filter(boden_mask, size=5) > 0.05


# hier möchte ich die Punkte extrahieren, die nicht zum Boden gehören, um sie für die RANSAC-Box-Top-Erkennung zu verwenden. 
nicht_boden_mask = valid_mask & (~boden_mask_filtered)
nicht_boden_points = PC04[nicht_boden_mask]

print("Non-boden points:", nicht_boden_points.shape)


# Box-Top-Ebene erkennen mit RANSAC
box_model, box_inliers_bool, box_count = ransac_plane(nicht_boden_points, threshold=0.008, max_iterations=1000)

print("Box top inliers:", box_count)

box_mask = np.zeros(PC04.shape[:2], dtype=bool)
box_mask[nicht_boden_mask] = box_inliers_bool

# hier möchte ich die größte verbundene Komponente in der Box-Top-Maske extrahieren, um sicherzustellen, dass wir nur die Hauptfläche des Box-Tops betrachten.
labeled, num_labels = ndi.label(box_mask)
print("Number of labels:", num_labels)
sizes = ndi.sum(box_mask, labeled, index=np.arange(1, num_labels + 1))
if len(sizes) > 0:
    largest_label = np.argmax(sizes) + 1
    box_top_mask = labeled == largest_label
else:
    box_top_mask = box_mask


# Abstand zwischen den Ebenen berechnen
def plane_distance(model1, model2):
    n1, d1 = model1
    n2, d2 = model2

    n1_unit = n1 / np.linalg.norm(n1)
    n2_unit = n2 / np.linalg.norm(n2)

    if np.dot(n1_unit, n2_unit) < 0:
        n2_unit = -n2_unit
        d2 = -d2

    return abs(d2 / np.linalg.norm(n2) - d1 / np.linalg.norm(n1))


height = plane_distance(boden_model, box_model)
print("Estimated box height:", height)

# hier will ich die Ecken der Box-Top-Maske finden, um die Länge und Breite der Box zu schätzen.
box_pixels = np.argwhere(box_top_mask) # (row, col) Indizes der Pixel, die zum Box-Top gehören
center = box_pixels.mean(axis=0) # das ist wie schwerpunkt der Box-Top-Pixel.
box_pixels_centered = box_pixels - center # PCA / Hauptachsen der Boxmaske
U, S, Vt = np.linalg.svd(box_pixels_centered, full_matrices=False)
axes = Vt

# Pixel auf die Hauptachsen projizieren
projected = box_pixels_centered @ axes.T

min_axis1, max_axis1 = np.percentile(projected[:, 0], [2, 98])
min_axis2, max_axis2 = np.percentile(projected[:, 1], [2, 98])


'''min_axis1, max_axis1 = projected[:, 0].min(), projected[:, 0].max()
min_axis2, max_axis2 = projected[:, 1].min(), projected[:, 1].max()'''

# Eckpunkte im lokalen Koordinatensystem
corners_projected = np.array([
    [min_axis1, min_axis2],
    [max_axis1, min_axis2],
    [max_axis1, max_axis2],
    [min_axis1, max_axis2],
])

# zurück in Bildkoordinaten
corners = corners_projected @ axes + center
corners_pixels = np.round(corners).astype(int)

# 3D-Koordinaten der Ecken
corners_PC = PC04[corners_pixels[:, 0], corners_pixels[:, 1]]

edge_lengths = [
    np.linalg.norm(corners_PC[1] - corners_PC[0]),
    np.linalg.norm(corners_PC[2] - corners_PC[1]),
    np.linalg.norm(corners_PC[3] - corners_PC[2]),
    np.linalg.norm(corners_PC[0] - corners_PC[3]),
]

length = max(edge_lengths)
width = min(edge_lengths)

print("Corner pixels:")
print(corners_pixels)
print("Corner 3D points:")
print(corners_PC)
print("Estimated box length:", length)
print("Estimated box width:", width)



#Visualizations
plt.figure()
plt.imshow(A04, cmap="gray")
plt.title("Amplitude")
plt.colorbar()

plt.figure()
plt.imshow(D04)
plt.title("Distance")
plt.colorbar()

plt.figure()
plt.imshow(boden_mask)
plt.title("Boden mask raw")
plt.colorbar()

plt.figure()
plt.imshow(boden_mask_filtered)
plt.title("Boden mask filtered")
plt.colorbar()


plt.figure()
plt.imshow(boden_uniform)
plt.title("Boden mask uniform")
plt.colorbar()


plt.figure()
plt.imshow(box_top_mask)
plt.title("Largest box top component")
plt.colorbar()


# Simple 2D-Visualisierung der Punktwolke mit den erkannten Boden- und Box-Top-Punkten
vis = np.zeros((*PC04.shape[:2], 3))
vis[:] = [0, 0, 0]
vis[boden_mask_filtered] = [0.5, 1.0, 0.3]
vis[box_top_mask] = [1.0, 0.0, 0.0]
rest_mask = valid_mask & (~boden_mask_filtered) & (~box_top_mask)
vis[rest_mask] = [0.0, 0.0, 0.7]


if len(box_pixels) >= 4:
    for r, c in corners_pixels:
        r0, r1 = max(0, r-2), min(vis.shape[0], r+3)
        c0, c1 = max(0, c-2), min(vis.shape[1], c+3)
        vis[r0:r1, c0:c1] = [0.0, 1.0, 1.0]

plt.figure()
plt.imshow(vis)
plt.title("Visualization of floor and box top")
plt.axis("off")
plt.show()



# 3D-Visualisierung der Punktwolke mit den erkannten Boden- und Box-Top-Punkten 
boden_points = PC04[boden_mask_filtered]
box_points = PC04[box_top_mask]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(boden_points[::10, 0], boden_points[::10, 1], boden_points[::10, 2], s=1, label="boden")
ax.scatter(box_points[:, 0], box_points[:, 1], box_points[:, 2], s=3, label="box top")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()