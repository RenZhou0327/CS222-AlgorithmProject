import cv2
import numpy as np
from GenerateMask.GenerateContours import readPoints

dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]


def combine_contours(pic_list):
    if len(pic_list) == 0:
        raise Exception("picture path list can not be empty!")
    img0 = cv2.imread(pic_list[0], 0)
    input_size = img0.shape
    com_contour = np.zeros(input_size)
    com_contour += (255 - img0)
    for pic_path in pic_list[1:]:
        img = cv2.imread(pic_path, 0)
        com_contour += (255 - img)
    com_contour = com_contour.clip(0, 255)
    com_contour = 255 - com_contour
    # print(com_contour.max(), com_contour.min())
    # cv2.imshow("new", com_contour)
    # cv2.waitKey(0)
    return com_contour, input_size


def get_cross_lines(matrix, row, col):
    nrow, ncol = matrix.shape
    pixels_cnt = 0
    lines_cnt = 0
    tmp = np.count_nonzero(matrix[0:row, col])
    if tmp >= 3:
        lines_cnt += 1
    pixels_cnt += tmp
    tmp = np.count_nonzero(matrix[row + 1: nrow, col])
    if tmp >= 3:
        lines_cnt += 1
    pixels_cnt += tmp
    tmp = np.count_nonzero(matrix[row, 0:col])
    if tmp >= 3:
        lines_cnt += 1
    pixels_cnt += tmp
    tmp = np.count_nonzero(matrix[row, col + 1: ncol])
    if tmp >= 3:
        lines_cnt += 1
    pixels_cnt += tmp
    i = row - 1
    j = col - 1

    tmp = 0
    while i >= 0 and j >= 0:
        if matrix[i][j] != 0:
            tmp += 1
        i -= 1
        j -= 1
    if tmp >= 3:
        lines_cnt += 1
    pixels_cnt += tmp
    i = row + 1
    j = col + 1
    tmp = 0
    while i < nrow and j < ncol:
        if matrix[i][j] != 0:
            tmp += 1
        i += 1
        j += 1
    if tmp >= 3:
        lines_cnt += 1
    pixels_cnt += tmp
    i = row - 1
    j = col + 1
    tmp = 0
    while i >= 0 and j < ncol:
        if matrix[i][j] != 0:
            tmp += 1
        i -= 1
        j += 1
    if tmp >= 3:
        lines_cnt += 1
    pixels_cnt += tmp
    i = row + 1
    j = col - 1
    tmp = 0
    while i < nrow and j >= 0:
        if matrix[i][j] != 0:
            tmp += 1
        i += 1
        j -= 1
    if tmp >= 3:
        lines_cnt += 1
    pixels_cnt += tmp
    return lines_cnt, pixels_cnt


def get_pixels_score(lines, pixels, row, col, input_size, total_num):
    if lines <= 4:
        return 0.0
    ALPHA = 0.8
    BIAS = -6
    center = [input_size[0] // 2, input_size[1] // 2]
    center_weight = 0.5 - (np.abs(row - center[0]) + np.abs(col - center[1])) / (input_size[0] + input_size[1])
    pixels_weight = pixels / total_num
    # print(center_weight, pixels_weight)
    bias_weight = ALPHA * center_weight + (1 - ALPHA) * pixels_weight
    lines_weight = 5 + (lines - 5) * 0.8
    x = lines_weight + bias_weight + BIAS
    score = ((1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x)) + 1) * 0.5
    return score


def complete_contours(contour, input_size):
    bin_contour = 255 - contour
    bin_contour = bin_contour.astype(np.uint8)
    _, bin_contour = cv2.threshold(bin_contour, 127, 255, cv2.THRESH_BINARY)
    pixels_num = np.count_nonzero(bin_contour)
    # print(bin)
    # cv2.imshow("new", bin_contour) # 黑底
    # cv2.waitKey(0)
    tmp = np.pad(bin_contour, ((1, 1), (1, 1)), "constant", constant_values=(0, 0))  # 黑底 0
    # cv2.imshow("xx", tmp)
    # cv2.waitKey(0)
    # file = open("test.txt", "w")
    for row in range(input_size[0]):
        for col in range(input_size[1]):
            if bin_contour[row][col] == 0:
                lines, pixels = get_cross_lines(tmp, row + 1, col + 1)
                # print(lines, pixels)
                score = get_pixels_score(lines, pixels, row, col, input_size, pixels_num)
                # print(row, col, score)
                if np.random.uniform() < score:
                    bin_contour[row][col] = 255
    #         file.write(str(np.round(2 ** lines, 1)) + '\t')
    #     file.write('\n')
    # file.close()
    # cv2.imwrite("demo_picture/lwx_heristic.jpg", 255 - bin_contour)
    # cv2.imshow("xxxx", bin_contour)
    # cv2.waitKey(0)
    # exit(0)
    return bin_contour


def dfs(i, j, x, y, img_region):
    # print(x, y)
    # x: 113, y:246
    stack_list = []
    stack_list.append((i, j))
    while len(stack_list) != 0:
        x0, y0 = stack_list.pop()
        for dir in dirs:
            i = x0 + dir[0]
            j = y0 + dir[1]
            if i < 0 or j < 0 or i >= x or j >= y:
                continue
            if img_region[i][j] != 0:
                continue
            img_region[i][j] = -1
            stack_list.append((i, j))
    return img_region


def dfs_max_area(i, j, x, y, img_region, fill_num):
    area = 0
    stack_list = []
    stack_list.append((i, j))
    while len(stack_list) != 0:
        x0, y0 = stack_list.pop()
        area += 1
        for dir in dirs:
            i = x0 + dir[0]
            j = y0 + dir[1]
            if i < 0 or j < 0 or i >= x or j >= y:
                continue
            if img_region[i][j] != 255:
                continue
            img_region[i][j] = fill_num
            stack_list.append((i, j))
    return img_region, area


def generate_mask(img, fig_path):
    img_ = np.pad(img, ((1, 1), (1, 1)), constant_values=(0, 0))
    x, y = img_.shape
    img_region = np.asarray(img_, dtype=np.int)
    img_region = dfs(0, 0, x, y, img_region)
    img_region[img_region >= 0] = 255
    img_region[img_region < 0] = 0
    last_idx = 0
    max_area = 0
    reserve_idx = 0
    for i in range(x):
        for j in range(y):
            if img_region[i][j] == 255:
                img_region, area = dfs_max_area(i, j, x, y, img_region, last_idx + 1)
                last_idx += 1
                if area > max_area:
                    max_area = area
                    reserve_idx = last_idx
    # print(max_area, reserve_idx)
    # print(img_region)
    img_region[img_region != reserve_idx] = 0
    img_region[img_region == reserve_idx] = 255
    new_img = np.asarray(255 - img_region[1:-1, 1:-1], dtype=np.uint8)
    mask_path = fig_path.replace(".", "_cut_mask.")
    cv2.imwrite(mask_path, new_img)
    cv2.imshow("xxx", new_img)
    cv2.waitKey(0)
    return new_img


def write_full_mask(local_mask, shape, point1, point2, fig_path):
    # cv2.imshow("new", local_mask)
    # cv2.waitKey(0)
    mask_img = np.pad(local_mask, ((point1[1], shape[0] - point2[1]), (point1[0], shape[1] - point2[0])),
                      constant_values=(255, 255))
    print(mask_img.shape)
    # cv2.imshow("new", mask_img)
    # cv2.waitKey(0)
    mask_path = fig_path.replace(".", "_mask.")
    cv2.imwrite(mask_path, mask_img)


def test_get_cross_lines():
    M = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 1, 1, 1, 2, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         ]
    M = np.array(M)
    print(get_cross_lines(M, 6, 4))


def test_get_pixel_score():
    lines = 7
    row = 5
    col = 2
    input_size = (10, 10)
    pixels = 15
    total_pixel = 100
    score = get_pixels_score(lines, pixels, row, col, input_size, total_pixel)


if __name__ == '__main__':
    fig_path, point1, point2 = readPoints()
    name = fig_path.split('/')[1].split('.')[0]
    path_list = ["test_demo/tmp_pics/" + name + "_tmp" + str(i) + ".jpg" for i in range(7, 10)]
    fig_contour, fig_shape = combine_contours(path_list)
    # cv2.imshow('new', fig_contour)
    # cv2.waitKey(0)
    # exit(0)
    full_contour = complete_contours(fig_contour, fig_shape)
    local_mask = generate_mask(full_contour, fig_path)
    # exit(0)
    full_img = cv2.imread(fig_path)
    print(full_img.shape)
    full_img_shape = full_img.shape[:2]
    print(full_img_shape)
    # point1, point2 = [449, 211], [517, 294]
    # point1, point2 = [270, 385], [454, 529]
    # point1, point2 = [425, 569], [510, 671]
    write_full_mask(local_mask, full_img_shape, point1, point2, fig_path)
    # get_pixels_score(5, 10, 0, 0, (10, 10))
