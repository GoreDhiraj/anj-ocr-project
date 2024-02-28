import subprocess
# from google.colab import files
import fnmatch
import io
import operator
import os
# from IPython.display import Image
import os.path
import shutil
import subprocess
# import imutils
from math import *
from PIL import Image as PILImage

import cv2
# from google.colab import files
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage


exception = []
# dir = '/content/temporary_files/'
new_dir = 'content/final_line_segmentation/'
rotate_line = os.path.join("content", "Rotated_line_by_HaughLine_Affine", "")
rotate_line_Dskew = os.path.join("content", "DSkew", "")
rotate_line_Haughline = os.path.join("content", "HaughLine_Affine", "")
cwd = os.getcwd()


def line_sort(lines):
    sort_lines = {}
    for line in lines:
        img_lb = line.split('.')[0]
        lb = [i for i in img_lb.split('_')]
        lb = [int(i) if i.isdigit() else i for i in lb]
        sort_lines[tuple(lb)] = line

    sorted_lines = sorted(sort_lines.items(), key=lambda x: (tuple(map(str, x[0])), x[1]))
    new_lines = [line for _, line in sorted_lines]
    return new_lines


def show_transitions(f_name):
    filename = f_name
    crop_line = os.listdir(filename)
    crop_lines = line_sort(crop_line)
    print(crop_lines)
    j = 1
    for i in crop_lines:
        print("Line no:", j)
        print("Image name:: ", i)
        im = filename + "/" + i
        crop_img = cv2.imread(im)

        width = crop_img.shape[1]
        height = crop_img.shape[0]
        print("Width, height: ", width, height)

        # plot_fig(crop_img)
        j = j + 1


def show_transitions_by_comparing(filename, gt_source_path, name1, name2):
    crop_line = os.listdir(filename)
    crop_lines = line_sort(crop_line)
    print("Cropped Line images: ", crop_lines)
    print()
    j = 1
    for i in crop_lines:
        print("Line no:", j)
        print("Image name:: ", i)
        im = filename + "/" + i
        crop_img = cv2.imread(im)

        width = crop_img.shape[1]
        height = crop_img.shape[0]

        # print("Annotated Line or GroundTruth ->")
        print(name1)
        # img_gd = gt_line_img_dir + i
        img_gd = gt_source_path + i
        img_gd1 = cv2.imread(img_gd)
        # plot_fig(img_gd1)

        # print("Predicted Line ->")
        print(name2)
        # plot_fig(crop_img)
        print()
        j = j + 1


def draw_BB(img_path, label_path, flag):
    img = cv2.imread(img_path)
    dh, dw, _ = img.shape

    lb = open(label_path, 'r')
    data = lb.readlines()
    lb.close()

    for dt in data:
        if flag == 0:
            _, x, y, w, h = map(float, dt.split(' '))
        else:
            _, x, y, w, h, conf = map(float, dt.split(' '))
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        if flag == 0:
            cv2.rectangle(img, (l, t), (r, b), (0, 250, 0), 2)
        else:
            cv2.rectangle(img, (l, t), (r, b), (0, 0, 250), 2)

    # plot_fig(img)
    return img


def yolo_detection(img_path, img_size, conf):
    weights_path = 'content/model/line_model_best.pt'
    script_path = 'content/yolov5/detect.py'
    command = [
        "python",
        script_path,
        "--weights",
        weights_path,
        "--img",
        str(img_size),
        "--conf",
        str(conf),
        "--source",
        img_path,
        "--save-conf",
        "--save-txt"
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error: {e}")


class Line_sort:
    def __init__(self, txt_files, txt_loc, sort_label, flag):
        self.txt_files = txt_files
        self.txt_loc = txt_loc
        self.sort_label = sort_label
        self.flag = flag
        self.read_file()

    def read_file(self):
        files = self.txt_files
        # os.mkdir('/content/sorted_line_after_1st_detection')
        # os.mkdir(self.sort_label)
        for file in files:
            txt_file = []
            file_loc = self.txt_loc + file

            with open(file_loc, 'r', encoding='utf-8', errors='ignore') as lines:
                for line in lines:
                    token = line.split()

                    _, x, y, w, h, conf = map(float, line.split(' '))
                    # print("width -> ",w)
                    # print("confidence -> ",conf)
                    if self.flag == 0:  # 1st line detection lavel
                        if w > 0.50 and conf < 0.50:
                            continue
                        else:
                            txt_file.append(token)
                    else:  # Word detection lavel
                        # if w > 0.50:
                        #   continue
                        # else:
                        txt_file.append(token)

            if self.flag == 0:  # 1st line detection lavel
                sorted_txt_file = sorted(txt_file, key=operator.itemgetter(2))
            else:  # Word detection lavel
                sorted_txt_file = sorted(txt_file, key=operator.itemgetter(1))

            # lenght = len(sorted_txt_file[0])
            self.file_write(sorted_txt_file, file)

    def file_write(self, txt_file, file_name):
        # loc = '/content/sorted_line_after_1st_detection/'+file_name
        loc = self.sort_label + file_name
        with open(loc, 'w') as f:
            c = 0
            for line in txt_file:
                for l in line:
                    c += 1
                    if c == len(line):
                        f.write('%s' % l)
                    else:
                        f.write('%s ' % l)
                f.write("\n")
                c = 0


def sort_detection_label(txt_loc, sort_label, flag):
    txt_files = os.listdir(txt_loc)
    obj = Line_sort(txt_files, txt_loc, sort_label, flag)


# text file and image sorting
def images_and_txtfile_sort(images, txt_file):
    txt = []
    image = []
    for item in txt_file:
        ch = ""
        for c in item:
            if c == "_":
                break
            ch += c
        txt.append(ch)

    txt.sort(key=int)
    sorted_txtfiles = []
    sorted_images = []

    for i in range(len(txt)):
        for ele, ele2 in zip(txt_file, images):
            st = ""
            st2 = ""
            for ch in ele:
                if ch == "_":
                    break
                st += ch
            if st == txt[i]:
                sorted_txtfiles.append(ele)
            for ch in ele2:
                if ch == "_":
                    break
                st2 += ch
            if st2 == txt[i]:
                sorted_images.append(ele2)
    return sorted_images, sorted_txtfiles


def line_segmantation(path, image_loc, txt_loc, images, txt_file):
    # Assuming line_sort function sorts based on some criteria
    image = line_sort(images)
    txt_files = line_sort(txt_file)

    for image, txt in zip(image, txt_files):
        # Extract image name without extension
        image_name = os.path.splitext(image)[0]

        # Create folder named "segment"
        new_folder = "segment"
        new_folder_loc = os.path.join(path, new_folder)

        # Create subfolder in "segment" based on image name
        os.makedirs(new_folder_loc, exist_ok=True)

        # Current image location
        current_image = os.path.join(image_loc, image)
        img = cv2.imread(current_image)
        dh, dw, _ = img.shape

        # Current txt file location
        current_txt = os.path.join(txt_loc, txt)
        fl = open(current_txt, 'r')
        data = fl.readlines()
        fl.close()

        k = 1
        for dt in data:
            _, x, y, w, h, conf = map(float, dt.split(' '))
            if 0.50 < w < 0.80:
                x = 0.5
                w = 1.0

            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)

            crop = img[t:b, l:r]
            output_filename = "{}_{}.jpg".format(image_name, k)
            output_path = os.path.join(new_folder_loc, output_filename)
            cv2.imwrite(output_path, crop)
            k += 1


def take_valid_img(images):
    image = []
    valid_img_ext = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
    for img in images:
        try:
            ext = img.split('.')[1]
            if ext not in valid_img_ext:
                continue
            else:
                image.append(img)
        except:
            continue
    return image


def line_segmentation(img_path, sorted_label):
    current_directory = img_path
    # print(current_directory)
    current_directory1 = sorted_label
    # print(current_directory1)
    images = []  # take for all images
    txt = []  # take for all txt files

    # images
    # take all files in list in current directory location
    current_directory_files = os.listdir(current_directory)
    # print(current_directory_files)
    if not current_directory_files:
        print("{} folder is empty!")

    # lines
    # take all files in list in current directory1 location
    current_directory_1_files = os.listdir(current_directory1)
    if not current_directory_1_files:
        print("Line folder is empty of {} !")

    txt_pattern = "*.txt"
    # take all only image in images list from current directory files list
    images1 = [entry for entry in current_directory_files]
    images = take_valid_img(images1)

    # take all only txt file in txt list from current directory1 files list
    txt = [entry for entry in current_directory_1_files if fnmatch.fnmatch(entry, txt_pattern)]

    # os.mkdir('/content/croped_line_after_1st_detection')
    curr_path = os.getcwd()
    path = os.path.join(curr_path, "content", "croped_line_after_1st_detection", "")
    # calling the line segmantation function
    line_segmantation(path, current_directory, current_directory1, images, txt)
    print("Successful line segmented in the {} folder".format(path))


class ImgCorrect():
    def __init__(self, img):
        self.img = img
        self.h, self.w, self.channel = self.img.shape
        # print("Original images h & w -> | w: ",self.w, "| h: ",self.h)
        if self.w <= self.h:
            self.scale = 700 / self.w
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.scale = 700 / self.h
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        # print("Scaled image:")
        # plot_fig(self.img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def img_lines(self):
        # print("Gray Image:")
        # plot_fig(self.gray)
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # print("Inverse Binary:")
        # plot_fig(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # rectangular structure
        # print("Kernel for dialation:")
        # print(kernel)
        binary = cv2.dilate(binary, kernel)  # dilate
        # print("Dialated Binary:")
        # plot_fig(binary)
        edges = cv2.Canny(binary, 50, 200)
        # print("Canny edged detection:")
        # plot_fig(edges)

        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        # print(self.lines)
        if self.lines is None:
            print("Line segment not found")
            return None

        lines1 = self.lines[:, 0, :]  # Extract as 2D
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # print("Probabilistic Hough Lines:")
        # plot_fig(imglines)
        # return imglines

    def search_lines(self):
        lines = self.lines[:, 0, :]  # extract as 2D

        number_inexist_k = 0
        sum_pos_k45 = number_pos_k45 = 0
        sum_pos_k90 = number_pos_k90 = 0
        sum_neg_k45 = number_neg_k45 = 0
        sum_neg_k90 = number_neg_k90 = 0
        sum_zero_k = number_zero_k = 0

        for x in lines:
            if x[2] == x[0]:
                number_inexist_k += 1
                continue
            # print(degrees(atan((x[3] - x[1]) / (x[2] - x[0]))), "pos:", x[0], x[1], x[2], x[3], "Slope:",(x[3] - x[1]) / (x[2] - x[0]))
            degree = degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            # print("Degree or Slope of detected lines : ",degree)
            if 0 < degree < 45:
                number_pos_k45 += 1
                sum_pos_k45 += degree
            if 45 <= degree < 90:
                number_pos_k90 += 1
                sum_pos_k90 += degree
            if -45 < degree < 0:
                number_neg_k45 += 1
                sum_neg_k45 += degree
            if -90 < degree <= -45:
                number_neg_k90 += 1
                sum_neg_k90 += degree
            if x[3] == x[1]:
                number_zero_k += 1

        max_number = max(number_inexist_k, number_pos_k45, number_pos_k90, number_neg_k45, number_neg_k90,
                         number_zero_k)
        # print("Num of lines in different Degree range ->")
        # print("Not a Line: ",number_inexist_k, "| 0 to 45: ",number_pos_k45, "| 45 to 90: ",number_pos_k90, "| -45 to 0: ",number_neg_k45, "| -90 to -45: ",number_neg_k90, "| Line where y1 equals y2 :",number_zero_k)

        if max_number == number_inexist_k:
            return 90
        if max_number == number_pos_k45:
            return sum_pos_k45 / number_pos_k45
        if max_number == number_pos_k90:
            return sum_pos_k90 / number_pos_k90
        if max_number == number_neg_k45:
            return sum_neg_k45 / number_neg_k45
        if max_number == number_neg_k90:
            return sum_neg_k90 / number_neg_k90
        if max_number == number_zero_k:
            return 0

    def rotate_image(self, degree):
        """
        Positive angle counterclockwise rotation
        :param degree:
        :return:
        """
        # print("degree:", degree)
        if -45 <= degree <= 0:
            degree = degree  # #negative angle clockwise
        if -90 <= degree < -45:
            degree = 90 + degree  # positive angle counterclockwise
        if 0 < degree <= 45:
            degree = degree  # positive angle counterclockwise
        if 45 < degree < 90:
            degree = degree - 90  # negative angle clockwise
        print("DSkew angle: ", degree)

        # degree = degree - 90
        height, width = self.img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(
            cos(radians(degree))))  # This formula refers to the previous content
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        # print("Height :",height)
        # print("Width :",width)
        # print("HeightNew :",heightNew)
        # print("WidthNew :",widthNew)

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # rotate degree counterclockwise
        # print("Mat Rotation (Before): ",matRotation)
        matRotation[0, 2] += (widthNew - width) / 2
        # Because after rotation, the origin of the coordinate system is the upper left corner of the new image, so it needs to be converted according to the original image
        matRotation[1, 2] += (heightNew - height) / 2
        # print("Mat Rotation (After): ",matRotation)

        # Affine transformation, the background color is filled with white
        imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        # Padding
        pad_image_rotate = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(0, 255, 0))
        # plot_fig(pad_image_rotate)

        return imgRotation


def dskew(line_path, img):
    img_loc = line_path + img
    im = cv2.imread(img_loc)

    # Padding
    bg_color = [255, 255, 255]
    pad_img = cv2.copyMakeBorder(im, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=bg_color)

    imgcorrect = ImgCorrect(pad_img)
    lines_img = imgcorrect.img_lines()
    # print(type(lines_img))

    if lines_img is None:
        rotate = imgcorrect.rotate_image(0)
    else:
        degree = imgcorrect.search_lines()
        rotate = imgcorrect.rotate_image(degree)

    return rotate


# Degree conversion
def DegreeTrans(theta):
    res = theta / np.pi * 180
    # print(res)
    return res


# Rotate the image degree counterclockwise (original size)
def rotateImage(src, degree):
    # The center of rotation is the center of the image
    h, w = src.shape[:2]
    # Calculate the two-dimensional rotating affine transformation matrix
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # print("Rotate Matrix: ")
    # print(RotateMatrix)

    # Affine transformation, the background color is filled with GREEN so that the rotation can be easily understood
    rotate1 = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(0, 255, 0))
    # plot_fig(rotate1)
    # Affine transformation, the background color is filled with white
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))

    # Padding
    bg_color = [255, 255, 255]
    pad_image_rotate = cv2.copyMakeBorder(rotate, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=bg_color)

    return pad_image_rotate


# Calculate angle by Hough transform
def CalcDegree(srcImage, canny_img):
    lineimage = srcImage.copy()
    lineimg = srcImage.copy()
    # Detect straight lines by Hough transform
    # The fourth parameter is the threshold, the greater the threshold, the higher the detection accuracy
    try:
        lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 200)
        # print("HoughLines: ")
        # cv2_imshow(lines)
        # Due to different images, the threshold is not easy to set, because the threshold is set too high, so that the line cannot be detected, the threshold is too low, the line is too much, the speed is very slow
        theta_sum = 0
        rho_sum = 0
        sum_x1 = sum_x2 = sum_y1 = sum_y2 = 0
        # Draw each line segment in turn
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                # print("theta:", theta, " rho:", rho)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 1000 * (-b)))
                y1 = int(round(y0 + 1000 * a))
                x2 = int(round(x0 - 1000 * (-b)))
                y2 = int(round(y0 - 1000 * a))
                # print("a: ",a, " b: ",b, " x0: ",x0, " y0: ",y0, " x1: ",x1, " y1: ",y1, " x2: ",x2, " y2: ",y2)
                # Only select the smallest angle as the rotation angle
                sum_x1 += x1
                sum_x2 += x2
                sum_y1 += y1
                sum_y2 += y2
                rho_sum += rho
                theta_sum += theta
                cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

        # print("HoughLines: ")
        # plot_fig(lineimage)
        print()

        pt1 = (sum_x1 // len(lines), sum_y1 // len(lines))
        pt2 = (sum_x2 // len(lines), sum_y2 // len(lines))

        # print("Sum of thetas: ",theta_sum)
        # print("lines: ",lines)
        average = theta_sum / len(lines)
        # print("Avg. Theta: ",average)
        angle = DegreeTrans(average) - 90
        # print("Avg. Angle: ",angle)
        print("Skewed Angle: ", angle)
        average_rho = rho_sum / len(lines)
        # print("Avg. rho: ",average_rho)

        # print('Draw best fit line with full:')
        # h, w = lineimg.shape[:2]
        # pt2 = (w,h)
        # print("Cordinates of the best fit line: ",pt1,pt2)
        cv2.line(lineimg, pt1, pt2, (0, 0, 255), 2)
        # plot_fig(lineimg)

        return angle
    except:
        angle = 0.0
        return angle


def ready_for_rotate(line_path, img):
    print()
    print("Image :: ", img)
    img_loc = line_path + img
    image = cv2.imread(img_loc)

    org_width = image.shape[1]
    org_height = image.shape[0]

    img1 = image
    im_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # print("Gray Image: ")
    # plot_fig(im_gray)

    edges = cv2.Canny(im_gray, 50, 150, apertureSize=3)
    # print("Canny Image: ")
    # plot_fig(edges)

    degree = CalcDegree(image, edges)

    if degree == 0.0:
        rotate = dskew(line_path, img)
        # print("Rotated Image by DSkew: ")
        # plot_fig(rotate)
        print()

        filename1 = rotate_line_Dskew + img
        cv2.imwrite(filename1, rotate)
        filename = rotate_line + img
        cv2.imwrite(filename, rotate)
    else:
        rotate = rotateImage(image, degree)
        # print("Rotated Image by Haughline Affine transform: ")
        # plot_fig(rotate)
        print()

        filename2 = rotate_line_Haughline + img
        cv2.imwrite(filename2, rotate)
        filename = rotate_line + img
        cv2.imwrite(filename, rotate)


def rotate_lines(first_detection):
    line_path = first_detection
    line_dir = line_sort(os.listdir(line_path))
    print(line_dir)

    for img in line_dir:
        ready_for_rotate(line_path, img)


def find_undetected_images(img, label):
    undetected_images_path = []

    # img_path = "/content/Rotated_line_by_HaughLine_Affine/"
    img_path = img
    # detect_lb_path = "/content/yolov5/runs/detect/exp2/labels/"
    detect_lb_path = label
    undetect_img_path = os.path.join("content", "final_line_segmentation", "")

    def take_valid_img(images):
        image = []
        valid_img_ext = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
        for img in images:
            try:
                ext = img.split('.')[1]
                if ext not in valid_img_ext:
                    continue
                else:
                    image.append(img)
            except:
                continue
        return image

    img1 = os.listdir(img_path)
    img = take_valid_img(img1)
    detect_lb = os.listdir(detect_lb_path)

    def find_undetect_img(img, detect_lb):
        img_lb = [im.split('.')[0] for im in img]
        dt_lb = [dt.split('.')[0] for dt in detect_lb]
        undt_lb = list(set(img_lb).difference(dt_lb))
        undetect_img = []
        detect_img = []
        for lb in undt_lb:
            for im in img:
                im_lb = im.split('.')[0]
                if lb == im_lb:
                    undetect_img.append(im)
                else:
                    detect_img.append(im)
        print("Undetect image: ", undetect_img)
        write_image(undetect_img)

    def write_image(undt_img):
        for im in undt_img:
            filename = undetect_img_path + im
            img = cv2.imread(img_path + im)
            cv2.imwrite(filename, img)
            undetected_images_path.append(filename)
            # print(undetected_images_path)

    find_undetect_img(img, detect_lb)


def crop_image(bb_data, destination, image, img_lb, dh, dw):
    x = float(bb_data[1])
    y = float(bb_data[2])
    w = float(bb_data[3])
    h = float(bb_data[4])

    # x = 0.5
    # w  = 1.0
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    crop = image[t:b, l:r]
    filename = destination + img_lb
    cv2.imwrite(filename, crop)
    print("Segmented successfully!\n")


def line_segmantation_2(img, img_path, label, label_path, segmented_img_path):
    dir = segmented_img_path
    print("Image path -> ", img_path)
    img1 = cv2.imread(img_path)
    dh, dw, _ = img1.shape
    txt_lb = open(label_path, 'r')
    txt_lb_data = txt_lb.readlines()
    txt_lb.close()
    img_name = img

    max_w = 0
    data1 = []
    for line in txt_lb_data:
        token = line.split()
        data1.append(token)

    if len(data1) == 1:
        bb_data = data1[0]
        wdth = float(bb_data[3])
        if wdth > 0.4:
            crop_image(bb_data, dir, img1, img_name, dh, dw, )
        else:
            filename = dir + img_name
            cv2.imwrite(filename, img1)
    elif len(data1) == 2:
        bb_data1 = data1[0]
        bb_data2 = data1[1]
        w1 = float(bb_data1[3])
        w2 = float(bb_data2[3])
        c1 = float(bb_data1[5])
        c2 = float(bb_data2[5])
        if w1 <= 0.5 and w2 <= 0.5:
            if c1 >= 0.8 and c2 >= 0.8:
                sorted_bb_data = sorted(data1, key=operator.itemgetter(5))
                bb_data = sorted_bb_data[-1]
                crop_image(bb_data, dir, img1, img_name, dh, dw, )
            else:
                filename = dir + img_name
                cv2.imwrite(filename, img1)
        else:
            sorted_bb_data = sorted(data1, key=operator.itemgetter(3))
            bb_data = sorted_bb_data[-1]
            crop_image(bb_data, dir, img1, img_name, dh, dw, )
    elif len(data1) == 3:
        sorted_bb_data = sorted(data1, key=operator.itemgetter(2))
        bb_data = sorted_bb_data[1]
        crop_image(bb_data, dir, img1, img_name, dh, dw, )
    else:
        sorted_bb_data = sorted(data1, key=operator.itemgetter(3))
        bb_data = sorted_bb_data[-1]
        crop_image(bb_data, dir, img1, img_name, dh, dw, )


def filter_list(list):
    list1 = []
    for i in list:
        if i.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', 'JPG', 'JPEG', 'PNG')):
            list1.append(i)
    return list1


def copy_files(path, files):
    for i in files:
        path1 = path + i
        print(path1)
        img = cv2.imread(path1)
    # cv2_imshow(img)


def trim_original_image(rotate, org_w, org_h, crop_amount=60):
    org_width = org_w
    org_height = org_h

    img1 = rotate
    width = img1.shape[1]
    height = img1.shape[0]
    print("Original height -> ", org_height)
    print("Original width -> ", org_width)

    if height <= 2 * crop_amount or width <= 2 * crop_amount:
        print("Warning: Image size is too small for specified crop amount.")
        return img1  # Return the original image without cropping

    start_row = crop_amount
    end_row = height - crop_amount

    start_col = crop_amount
    end_col = width - crop_amount

    img_new = img1[start_row:end_row, start_col:end_col]

    width1 = img_new.shape[1]
    height1 = img_new.shape[0]
    print("New height -> ", height1)
    print("New width -> ", width1)

    return img_new


def remove_contents_of_last_folder(path):
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def word_segmentation(line_images, word_labels):
    line_img = os.listdir(line_images)
    word_label = os.listdir(word_labels)
    print(line_img)
    print(word_label)

    for i in word_label:
        if i == "163_19_11.txt":
            print("Yes")
        for j in line_img:
            if j == "163_19_11.jpg":
                print("Yes")
            fn_i = i.split(".")
            fn_j = j.split(".")
            if fn_i[0] == fn_j[0]:
                # print("yes")
                dir = os.path.join(cwd, "content", "final_word_segmentation", fn_i[0])

                os.mkdir(dir)

                img = cv2.imread(line_images + j)
                dh, dw, _ = img.shape
                txt_lb = open(word_labels + i, 'r')
                txt_lb_data = txt_lb.readlines()
                txt_lb.close()
                img_lb = fn_i[0]

                k = 1
                for dt in txt_lb_data:
                    # _, x, y, w, h = map(float, dt.split(' '))
                    _, x, y, w, h, conf = map(float, dt.split(' '))
                    l = int((x - w / 2) * dw)
                    r = int((x + w / 2) * dw)
                    t = int((y - h / 2) * dh)
                    b = int((y + h / 2) * dh)



                    crop = img[t:b, l:r]
                    cv2.imwrite("{}/{}_{}.jpg".format(dir, img_lb, k), crop)
                    k += 1


def yolo_word_detection(img_path, img_size, conf):
    weights_path = 'content/model/word_model_best.pt'
    script_path = 'content/yolov5/detect.py'
    img_path = 'content/final_line_segmentation'
    command = [
        "python",
        script_path,
        "--weights",
        weights_path,
        "--img",
        str(img_size),
        "--conf",
        str(conf),
        "--source",
        img_path,
        "--save-conf",
        "--save-txt"
    ]

    try:
        subprocess.run(command, check=True)
        print("YOLOv5 detection completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during YOLOv5 detection: {e}")





def main():
    # remove_previous_saves()
    current_path = os.getcwd()
    # # print(current_path)
    # #

    image_folder_path = os.path.join(current_path, "content", "uploaded_image", "")
    destination_folder = os.path.join(current_path, "content", "temporary_files", "")

    for file in os.listdir(image_folder_path):
        filename, extension = os.path.splitext(file)
        if extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            src_path = os.path.join(image_folder_path, file)
            dst_path = os.path.join(destination_folder, "1" + extension)
            shutil.copy(src_path, dst_path)
            print(f'File {file} has been copied to {dst_path}')

    # First detection
    directory = os.path.join(current_path, "content", "temporary_files")
    img_path = directory
    img_size = 640
    conf = 0.30
    yolo_detection(img_path, img_size, conf)
    print("first Yolo detection is done !")
    #

    # Sorting of labels
    txt_loc = os.path.join(current_path, "content", "yolov5", "runs", "detect", "exp", "labels", "")
    print(txt_loc)
    new_sort_label = os.path.join(current_path, "content", "sorted_line_after_1st_detection", "")
    flag = 0
    sort_detection_label(txt_loc, new_sort_label, flag)
    print("Sorting of labels done !")

    #  Line Segmentation after 1st Detection...
    img_path = os.path.join(current_path, "content", "temporary_files", "")
    sorted_label = os.path.join(current_path, "content", "sorted_line_after_1st_detection", "")
    print(img_path, sorted_label)
    line_segmentation(img_path, sorted_label)
    #
    # #Rotate Lines

    first_detection = os.path.join(current_path, "content", "croped_line_after_1st_detection", "segment", "")
    rotate_lines(first_detection)

    # Second yolo detection ..
    rotated_img_path = os.path.join(current_path, "content", "Rotated_line_by_HaughLine_Affine", "")
    img_size = 640
    conf = 0.50
    yolo_detection(rotated_img_path, img_size, conf)

    # Second Segmentation of lines ..

    target_label_path = os.path.join(current_path, 'content', 'yolov5', 'runs', 'detect', 'exp2', 'labels', '')
    target_image_path = os.path.join(current_path, "content", "Rotated_line_by_HaughLine_Affine", "")
    target_image = os.listdir(target_image_path)
    target_label = os.listdir(target_label_path)
    for i in target_image:
        for j in target_label:
            # Split filenames to extract base names'
            fn_i = os.path.splitext(i)[0]
            fn_j = os.path.splitext(j)[0]

            # Check if the base names match
            if fn_i == fn_j:
                img_path = os.path.join(target_image_path, i)
                sorted_label_path = os.path.join(target_label_path, j)
                line_segmantation_2(i, img_path, j, sorted_label_path, new_dir)

    # # Finding Undetected Image
    find_undetected_images(target_image_path, target_label_path)

    final_line_segment = os.path.join(current_path, "content", "final_line_segmentation", "")
    rotate_line_Dskew = os.path.join(current_path, "content", "Dskew", "")

    dskew_img_list = os.listdir(rotate_line_Dskew)
    print(dskew_img_list)

    for i in dskew_img_list:
        print("Target image -> ", i)
        temp = os.path.join(final_line_segment, i)
        print("Path -> ", temp)

        img1 = cv2.imread(temp)
        if img1 is None:
            print(f"Error: Could not read image {temp}")
            continue  # Skip to the next iteration if the image cannot be read

        height, width, channels = img1.shape

        temp2 = os.path.join(rotate_line_Dskew, i)
        img2 = cv2.imread(temp2)
        if img2 is None:
            print(f"Error: Could not read image {temp2}")
            continue  # Skip to the next iteration if the image cannot be read

        height2, width2, channels = img2.shape

        print(f"Original Image Dimensions: {height} x {width}")
        print(f"Rotated Image Dimensions: {height2} x {width2}")

        if height >= height2:
            temp3 = trim_original_image(img1, width, height)
            cv2.imwrite(temp, temp3)
            print("Image trimmed and saved.")
        else:
            print("Skipping trimming as the original image is smaller.")
        print()

    img_path = 'content/final_line_segmentation'
    yolo_word_detection(img_path, img_size=640, conf=0.40)

    txt_loc = os.path.join(cwd, "content", "yolov5", "runs", "detect", "exp3", "labels", "")
    new_sort_label = os.path.join(cwd, "content", "sorted_Word_detection", "")
    flag = 1
    sort_detection_label(txt_loc, new_sort_label, flag)

    # Word Segmentation...
    word_labels = os.path.join(cwd, "content", "sorted_Word_detection", "")
    line_images = os.path.join(cwd, "content", "final_line_segmentation", "")
    final_word_dir = os.path.join(cwd, "content", "final_word_segmentation", "")
    word_segmentation(line_images, word_labels)



    # Create a Workbook and add a worksheet
    workbook = Workbook()
    sheet = workbook.active

    # List all subdirectories in the specified directory
    image_directory = './content/final_word_segmentation/'
    subdirectories = sorted([d for d in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, d))])

    # Create a DataFrame with image paths
    df = pd.DataFrame(columns=['images'])

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(image_directory, subdirectory)
        w_images = os.listdir(subdirectory_path)
        images = sorted(w_images, key=len)
        images = [os.path.join(subdirectory_path, i) for i in images]
        df = pd.concat([df, pd.DataFrame({'images': images})], ignore_index=True)

    print(df.head())

    # Add images to the Excel sheet
    for index, row in df.iterrows():
        image_path = row['images']
        img = PILImage.open(image_path)

        # Set the dimensions for the images in the Excel sheet
        image_height = img.size[1] / 96  # Image height in inches (assuming 96 DPI)
        image_width = img.size[0] / 96  # Image width in inches

        # Resize the image
        img = img.resize((int(image_width * 96), int(image_height * 96)))  # 96 DPI

        # Convert the image data to bytes
        img_bytesio = io.BytesIO()
        img.save(img_bytesio, format='PNG')  # You can change the format as needed

        excel_image = ExcelImage(img_bytesio)

        # Set the dimensions of the image in the Excel sheet
        sheet.column_dimensions[f'A'].width = image_width * 8  # Adjust the column width
        sheet.row_dimensions[index + 1].height = image_height * 72  # Adjust the row height (assuming 72 points per inch)


    # Save the Excel workbook
    excel_file_path = './content/Excel/Words.xlsx'
    workbook.save(excel_file_path)



if __name__ == "__main__":
    main()
