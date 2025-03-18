import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import copy
from PIL import Image

def is_positive(x):
    return x > 0


def get_y_parabolic(factors, x, x_center, y_center):
    return round((factors[0] * (x - x_center) * (x - x_center) + factors[1] * (x - x_center) + factors[2] + y_center))


def start_points(factors, is_vertical, rect_top_left_x, rect_top_left_y, rect_width, rect_height):
    if not is_vertical:
        rect_top_left_x, rect_top_left_y = rect_top_left_y, rect_top_left_x
        rect_width, rect_height = rect_height, rect_width
    x = rect_top_left_x
    i = 0
    tmp = []
    while x <= rect_top_left_x + rect_width // 2:
        y = get_y_parabolic(factors, x, rect_top_left_x, rect_top_left_y)
        tmp.append((x, y))
        x += round(rect_width / 8)
        i += 1
    x = rect_top_left_x + rect_width // 2
    y = get_y_parabolic(factors, x, rect_top_left_x, rect_top_left_y)
    tmp.append((x, y))
    i += 1
    for i in range(len(tmp) - 1, -1, -1):
        x = (rect_top_left_x + rect_width // 2) * 2 - tmp[i][0]
        y = tmp[i][1]
        tmp.append((x, y))
        i += 1

    ans = []
    if not is_vertical:
        for i in range(0, len(tmp)):
            ans.append((tmp[i][1], tmp[i][0]))
    else:
        ans = tmp

    return ans


def get_factors(rect_width, rect_height):
    is_vertical = False
    ir = np.random.randint(4)
    if ir == 0:
        PointsX = [0, rect_width // 2, rect_width]
        PointsY = [rect_height, 0, rect_height]
    elif ir == 1:
        PointsX = [0, rect_width // 2, rect_width]
        PointsY = [0, rect_height, 0]
    elif ir == 2:
        PointsY = [0, rect_height // 2, rect_height]
        PointsX = [rect_width, 0, rect_width]
    elif ir == 3:
        PointsY = [0, rect_height // 2, rect_height]
        PointsX = [0, rect_width, 0]

    if ir in (1, 0):
        factors = [[x ** 2, x, 1] for x in PointsX]
        res = np.linalg.solve(factors, PointsY)
        is_vertical = True
    else:
        factors = [[x ** 2, x, 1] for x in PointsY]
        res = np.linalg.solve(factors, PointsX)
    return res, is_vertical


def convert_to_homogeneous(x, y):
    coords = []
    for i in range(len(x)):
        coords.append([x[i], y[i], 1])
    coords = np.array(coords)
    return coords.T


def rotation_coords(x, y, x0, y0, phi):
    xt, yt = transpose_coords(x, y, -x0, -y0)
    coords = np.array([[np.cos(phi), -np.sin(phi), 0],
                       [np.sin(phi), np.cos(phi), 0],
                       [0, 0, 1]]) @ convert_to_homogeneous(xt, yt)
    xt, yt = transpose_coords(coords[0], coords[1], x0, y0)
    return xt, yt


def scale_coords(x, y, x0, y0, alpha, beta):
    xt, yt = transpose_coords(x, y, -x0, -y0)

    coords = np.array([[alpha, 0, 0],
                       [0, beta, 0],
                       [0, 0, 1]]) @ convert_to_homogeneous(xt, yt)
    xt, yt = transpose_coords(coords[0], coords[1], x0, y0)

    return xt, yt


def transpose_coords(x, y, lamb, mu):
    Hcoords = convert_to_homogeneous(x, y)
    result_coords = np.array([[1, 0, lamb],
                              [0, 1, mu],
                              [0, 0, 1]]) @ Hcoords
    return np.round(result_coords[0]).astype(int), np.round(result_coords[1]).astype(int)


def __Bresenheam_yx(plot, x_start, y_start, x_end, y_end, clr1, clr2, width, vertices=None):
    if vertices is None:
        vertices = []
    dx = x_end - x_start
    dy = y_end - y_start
    s = 0
    y = y_start
    dcolor = (clr2 - clr1) / dx
    for x in range(x_start, x_end + 1, 1):
        if s >= dx:
            y += 1
            s -= 2 * dx
        elif s <= -dx:
            y -= 1
            s += 2 * dx
        vertices.append((float(x), float(y), *tuple(clr1)))
        for i in range(-width // 2, width // 2 + 1):
            for j in range(-width // 2, width // 2 + 1):
                plot[y + i, x + j] = clr1
        clr1 = clr1 + dcolor
        s += 2 * (dy)
    return vertices


def __Bresenheam_xy(plot, x_start, y_start, x_end, y_end, clr1, clr2, width, vertices=None):
    if vertices is None:
        vertices = []
    dx = x_end - x_start
    dy = y_end - y_start
    if dy == 0:
        return vertices
    s = 0
    x = x_start
    dcolor = (clr2 - clr1) / dy
    for y in range(y_start, y_end + 1, 1):
        if s >= dy:
            x += 1
            s -= 2 * dy
        elif s <= -dy:
            x -= 1
            s += 2 * dy
        vertices.append((int(x), int(y), *tuple(clr1)))
        for i in range(-width // 2, width // 2 + 1):
            for j in range(-width // 2, width // 2 + 1):
                plot[y + i, x + j] = clr1
        clr1 = clr1 + dcolor
        s += 2 * (dx)
    return vertices


def Bresenham_colour(plot, ctmp, clr1, clr2, width):
    vertices_contour = []
    for i in range(len(ctmp)):
        clr1, clr2 = clr2, clr1
        dx = ctmp[(i + 1) % (len(ctmp))][0] - ctmp[i][0]
        dy = ctmp[(i + 1) % (len(ctmp))][1] - ctmp[i][1]
        if np.abs(dx) > np.abs(dy):
            if dx > 0:
                vertices_contour.append(__Bresenheam_yx(plot, ctmp[i][0], ctmp[i][1], ctmp[(i + 1) % (len(ctmp))][0], ctmp[(i + 1) % (len(ctmp))][1], clr1, clr2, width))
            else:
                vertices_contour.append(__Bresenheam_yx(plot, ctmp[(i + 1) % (len(ctmp))][0], ctmp[(i + 1) % (len(ctmp))][1], ctmp[i][0], ctmp[i][1], clr2, clr1, width))
        else:
            if dy > 0:
                vertices_contour.append(__Bresenheam_xy(plot, ctmp[i][0], ctmp[i][1], ctmp[(i + 1) % (len(ctmp))][0], ctmp[(i + 1) % (len(ctmp))][1], clr1, clr2, width))
            else:
                vertices_contour.append(__Bresenheam_xy(plot, ctmp[(i + 1) % (len(ctmp))][0], ctmp[(i + 1) % (len(ctmp))][1], ctmp[i][0], ctmp[i][1], clr2, clr1, width))

    return vertices_contour


def fill_polygon_no_interpolation(plot, pixels):
    for i in range(0, len(pixels), 2):
        left_vertex = pixels[i]
        right_vertex = pixels[i + 1]
        for x in range(int(left_vertex[0]), int(right_vertex[0])):
            plot[int(left_vertex[1]), x] = pixels[0][2:]



def fill_polygon_w_interpolation(plot, pixels):
    for i in range(0, len(pixels), 2):
        left_vertex = pixels[i]
        right_vertex = pixels[i + 1]
        color1 = np.array(left_vertex[2:])
        color2 = np.array(right_vertex[2:])
        dcolor = (color2 - color1) / (- left_vertex[0] + right_vertex[0])
        k = 0
        for x in range(int(left_vertex[0]), int(right_vertex[0])):
            plot[int(left_vertex[1]), x] = color1 + dcolor * k
            k += 1


def sort_pixels_by_y(all_vertexes):
    all_vertexes = all_vertexes[np.argsort(all_vertexes[:, 1])]
    vertex_sorted_by_y = np.array([])
    for y in range(int(all_vertexes[0][1]), int(all_vertexes[-1][1])):
        tmparr = []
        for j in range(len(all_vertexes)):
            if all_vertexes[j][1] == y:
                tmparr.append(all_vertexes[j])
        tmparr = np.array(tmparr)
        if (len(tmparr)) == 0:
            continue
        tmparr = tmparr[np.argsort(tmparr[:, 0])]
        if len(vertex_sorted_by_y) == 0:
            vertex_sorted_by_y = np.array(tmparr[0])
        else:
            vertex_sorted_by_y = np.vstack((vertex_sorted_by_y, tmparr[0]))
        vertex_sorted_by_y = np.vstack((vertex_sorted_by_y, tmparr[-1]))
    return vertex_sorted_by_y


def rotate_image(image, angle, width, height):
    tmp_coords = np.array([[i, j] for i in range(len(image)) for j in range(len(image[0]))])
    rotated_coords_x, rotated_coords_y = rotation_coords(tmp_coords[:, 0], tmp_coords[:, 1], width // 2, height // 2, angle)
    image_array = np.zeros_like(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            x = int(rotated_coords_x[i * len(image[0]) + j])
            y = int(rotated_coords_y[i * len(image[0]) + j])
            if 0 <= x < len(image) and 0 <= y < len(image[0]):
                image_array[i][j] = image[x][y]

    return image_array

def stretch_image_from_array(image_array, new_width, new_height):
    image = Image.fromarray(image_array)
    stretched_image = image.resize((new_width, new_height))
    return np.array(stretched_image)


def fill_polygon_with_texture(plot, pixels, texture, rect_top_left_x, rect_top_left_y):
    for i in range(0, len(pixels), 2):
        left_vertex = pixels[i]
        right_vertex = pixels[i + 1]
        for x in range(int(left_vertex[0]), int(right_vertex[0])):
            plot[int(left_vertex[1]), x] = texture[int(left_vertex[1]) - rect_top_left_y, x - rect_top_left_x]


def get_new_coords_height_width_rectangle(rect_top_left_x, rect_top_left_y, rect_width, rect_height, angle):
    corners_coords = np.array([[rect_top_left_x, rect_top_left_y], [rect_top_left_x + rect_width, rect_top_left_y], [rect_top_left_x, rect_top_left_y + rect_height], [rect_top_left_x + rect_width,
                                                                                                                            rect_top_left_y + rect_height]])
    rotated_corners_coords_x, rotated_corners_coords_y = rotation_coords(corners_coords[:, 0], corners_coords[:, 1], rect_top_left_x + rect_width // 2, rect_top_left_y + rect_height // 2, angle)
    rotated_corners_coords = []
    for i in range(len(corners_coords)):
        rotated_corners_coords.append([rotated_corners_coords_x[i], rotated_corners_coords_y[i]])
    rotated_corners_coords = np.array(rotated_corners_coords)
    min_x_point = rotated_corners_coords[np.argmin(rotated_corners_coords[:, 0])][0]
    max_x_point = rotated_corners_coords[np.argmax(rotated_corners_coords[:, 0])][0]
    min_y_point = rotated_corners_coords[np.argmin(rotated_corners_coords[:, 1])][1]
    max_y_point = rotated_corners_coords[np.argmax(rotated_corners_coords[:, 1])][1]
    return min_x_point, min_y_point, max_x_point-min_x_point,max_y_point-min_y_point

def fill_polygon_with_texture_repeat(plot, pixels, texture, rect_top_left_x, rect_top_left_y, tx, ty):
    tx +=1
    ty +=1
    print(np.shape(texture))
    new_texture = stretch_image_from_array(texture, np.shape(texture)[1]//tx, np.shape(texture)[0]//ty)
    print(np.shape(new_texture)[1])
    for i in range(0, len(pixels), 2):
        left_vertex = pixels[i]
        right_vertex = pixels[i + 1]
        for x in range(int(left_vertex[0]), int(right_vertex[0])):
            print((int(left_vertex[1]) - rect_top_left_y))
            print((x - rect_top_left_x) % np.shape(new_texture)[0])
            plot[int(left_vertex[1]), x] = new_texture[((int(left_vertex[1]) - rect_top_left_y)) % np.shape(new_texture)[0], (x - rect_top_left_x) % np.shape(new_texture)[1]]



height = 200
width = 200
background_colour = (255, 255, 255)

v_base = np.full((height, width, 3), background_colour, dtype="uint8")
v_filled_wo_color_interpolation = np.full((height, width, 3), background_colour, dtype="uint8")
v_filled_w_color_interpolation = np.full((height, width, 3), background_colour, dtype="uint8")
v_scaled = np.full((height, width, 3), background_colour, dtype="uint8")

colour = (np.random.randint(0, 255, size=3))
rect_height = np.random.randint(80, 120)
rect_width = np.random.randint(80, 120)
rect_top_left_x = np.random.randint(30, width - rect_width - 30)
rect_top_left_y = np.random.randint(30, height - rect_height - 30)
clr = (np.random.randint(0, 255, size=3))
angle = np.random.random() * np.pi * 2

bias_x = - (rect_top_left_x + rect_width // 2) + (width // 2)
bias_y = - (rect_top_left_y + rect_height // 2) + (height // 2)

segments_to_Bresenham = start_points(*get_factors(rect_width, rect_height), rect_top_left_x, rect_top_left_y, rect_width, rect_height)
segments_to_Bresenham = np.array(segments_to_Bresenham)


all_vertexes = Bresenham_colour(v_base, segments_to_Bresenham, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 1)
all_vertexes = np.array(list([elem[i] for elem in all_vertexes for i in range(len(elem))]))
vertex_sorted_by_y = sort_pixels_by_y(all_vertexes)


image_path = r'C:\Users\artyo\Downloads\1.jpg'
image = Image.open(image_path)
image_array = np.array(image)
formated_image = stretch_image_from_array(image_array, rect_width, rect_height)
fill_polygon_with_texture(v_base, vertex_sorted_by_y, formated_image, rect_top_left_x, rect_top_left_y)


plt.suptitle(f"Начальное изображение")
plt.imshow(v_base)
plt.show()

v_rotated = np.full((height, width, 3), background_colour, dtype="uint8")
v_transposed =  np.full((height, width, 3), background_colour, dtype="uint8")
v_ =  np.full((height, width, 3), background_colour, dtype="uint8")
rotated_image = rotate_image(formated_image, angle, width, height)
segments_to_Bresenham_x_rotated, segments_to_Bresenham_y_rotated = rotation_coords(segments_to_Bresenham[:,0],segments_to_Bresenham[:,1], rect_top_left_x + rect_width // 2, rect_top_left_y +
                                                                                   rect_height // 2, angle)
segments_to_Bresenham_rotated = []
for i in range(len(segments_to_Bresenham_x_rotated)):
    segments_to_Bresenham_rotated.append([segments_to_Bresenham_x_rotated[i], segments_to_Bresenham_y_rotated[i]])
all_vertexes1 = Bresenham_colour(v_rotated, segments_to_Bresenham_rotated, np.array((0,0,0)), np.array((0,0,0)), 1)
all_vertexes1 = np.array(list([elem[i] for elem in all_vertexes1 for i in range(len(elem))]))
vertex_sorted_by_y1 = sort_pixels_by_y(all_vertexes1)
rect_rotated_top_left_x, rect_rotated_top_left_y, rect_rotated_width, rect_rotated_height = get_new_coords_height_width_rectangle(rect_top_left_x, rect_top_left_y, rect_width,  rect_height, angle)
formated_image_rotated = stretch_image_from_array(formated_image, rect_rotated_width, rect_rotated_height)
fill_polygon_with_texture(v_rotated, vertex_sorted_by_y1, formated_image_rotated, rect_rotated_top_left_x, rect_rotated_top_left_y)
num = "{:.2f}".format(angle)
plt.suptitle(f"Поворот на {num} радиан")
plt.imshow(v_rotated)
plt.show()

segments_to_Bresenham_x_transposed, segments_to_Bresenham_y_transposed = transpose_coords(segments_to_Bresenham[:,0],segments_to_Bresenham[:,1] , bias_x, bias_y)
segments_to_Bresenham_transposed = []
for i in range(len(segments_to_Bresenham_x_transposed)):
    segments_to_Bresenham_transposed.append([segments_to_Bresenham_x_transposed[i], segments_to_Bresenham_y_transposed[i]])
all_vertexes2 = Bresenham_colour(v_transposed, segments_to_Bresenham_transposed, np.array((0,0,0)), np.array((0,0,0)), 3)
all_vertexes2 = np.array(list([elem[i] for elem in all_vertexes2 for i in range(len(elem))]))
vertex_sorted_by_y2 = sort_pixels_by_y(all_vertexes2)
fill_polygon_with_texture(v_transposed, vertex_sorted_by_y2, formated_image, rect_top_left_x + bias_x, rect_top_left_y+bias_y)
plt.suptitle(f"Смещенеие на x = {bias_x}, на y = {bias_y}")
plt.imshow(v_transposed)
plt.show()

tx = 4
ty = 2

v_res = np.full((height, width, 3), background_colour, dtype="uint8")


fill_polygon_with_texture_repeat(v_res, vertex_sorted_by_y, formated_image, rect_top_left_x, rect_top_left_y, tx, ty)
plt.imshow(v_res)
plt.show()