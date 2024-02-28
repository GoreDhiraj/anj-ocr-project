import os.path
import re

import streamlit as st
from natsort import natsorted
from paddleocr import PaddleOCR

use_gpu = True
from Segmentation import *

current_wd = os.getcwd()
txt_loc = os.path.join(current_wd, "content", "yolov5", "runs", "detect", "exp", "labels", "")
new_sort_label = os.path.join(current_wd, "content", "sorted_line_after_1st_detection", "")
sorted_label = os.path.join(current_wd, "content", "sorted_line_after_1st_detection", "")
first_detection = os.path.join(current_wd, "content", "croped_line_after_1st_detection", "segment", "")
rotated_img_path = os.path.join(current_wd, "content", "Rotated_line_by_HaughLine_Affine", "")
target_label_path = os.path.join(current_wd, 'content', 'yolov5', 'runs', 'detect', 'exp2', 'labels', '')
input_folder = os.path.join(current_wd, "content", "final_word_segmentation", "")

uploaded_file = st.file_uploader('Upload Images', accept_multiple_files=False,
                                 type=['jpg', 'jpeg', 'png', 'gif', 'bmp'])
destination_folder = os.path.join(current_wd, "content", "temporary_files", "")
if uploaded_file:
    filename, extension = os.path.splitext(uploaded_file.name)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    if extension.lower() in valid_extensions:
        dst_path = os.path.join(destination_folder, "1" + extension)
        with open(dst_path, 'wb') as f:
            f.write(uploaded_file.read())
        print(f'File {uploaded_file.name} has been uploaded to {dst_path}')
        st.success("Image Was Uploaded Succesfully !")
        output_txt_path = os.path.join(current_wd, f"{filename}_recognized_text.txt")  # Set output text file name



def remove_contents_of_last_folder(path):
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def remove_previous_saves():
    curr_dir = os.getcwd()

    p1 = os.path.join(curr_dir, "content", "temporary_files", "")
    p2 = os.path.join(curr_dir, "content", "croped_line_after_1st_detection", "segment", "")
    p3 = os.path.join(curr_dir, "content", "sorted_line_after_1st_detection", "")
    p4 = os.path.join(curr_dir, "content", "yolov5", "runs", "detect", "")
    p5 = os.path.join(curr_dir, "content", "final_line_segmentation", "")
    p6 = os.path.join(curr_dir, "content", "Rotated_line_by_HaughLine_Affine", "")
    p7 = os.path.join(curr_dir, "content", "DSkew", "")
    p8 = os.path.join(curr_dir, "content", "HaughLine_Affine", "")
    p9 = os.path.join(curr_dir, "doc", "Results", "")
    p10 = os.path.join(curr_dir, "content", "sorted_Word_detection", "")
    p11 = os.path.join(curr_dir, "content", "final_word_segmentation", "")
    p12 = os.path.join(curr_dir, "content", "uploaded_image", "")
    # Remove contents of the last folder in each path
    remove_contents_of_last_folder(p1)
    remove_contents_of_last_folder(p2)
    remove_contents_of_last_folder(p3)
    remove_contents_of_last_folder(p4)
    remove_contents_of_last_folder(p5)
    remove_contents_of_last_folder(p6)
    remove_contents_of_last_folder(p7)
    remove_contents_of_last_folder(p8)
    remove_contents_of_last_folder(p9)
    remove_contents_of_last_folder(p10)
    remove_contents_of_last_folder(p11)


def sort_dirs(dirs):
    return natsorted(dirs, key=lambda x: [int(s) for s in re.findall(r'\d+', x)])


def main():
    img_path = os.path.join(current_wd, "content", "temporary_files", "")
    yolo_detection(img_path, img_size=640, conf=0.30)
    txt_loc1 = os.path.join(current_wd, "content", "yolov5", "runs", "detect", "exp", "labels", "")
    new_sort_label1 = os.path.join(current_wd, "content", "sorted_line_after_1st_detection", "")
    sort_detection_label(txt_loc1, new_sort_label1, flag=0)
    line_segmentation(img_path, sorted_label)
    rotate_lines(first_detection)
    yolo_detection(rotated_img_path, img_size=640, conf=0.50)
    target_label_path = os.path.join(current_wd, 'content', 'yolov5', 'runs', 'detect', 'exp2', 'labels', '')
    target_image_path = os.path.join(current_wd, "content", "Rotated_line_by_HaughLine_Affine", "")
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
    find_undetected_images(target_image_path, target_label_path)

    final_line_segment = os.path.join(current_wd, "content", "final_line_segmentation", "")
    rotate_line_Dskew = os.path.join(current_wd, "content", "Dskew", "")

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
    st.success("Your lines are now Segmented using YOLO")

    img_path = 'content/final_line_segmentation'
    yolo_word_detection(img_path, img_size=640, conf=0.40)

    txt_loc2 = os.path.join(cwd, "content", "yolov5", "runs", "detect", "exp3", "labels", "")
    new_sort_label2 = os.path.join(cwd, "content", "sorted_Word_detection", "")
    flag = 1
    sort_detection_label(txt_loc2, new_sort_label2, flag)

    # Word Segmentation...
    word_labels = os.path.join(cwd, "content", "sorted_Word_detection", "")
    line_images = os.path.join(cwd, "content", "final_line_segmentation", "")
    # final_word_dir = os.path.join(cwd, "content", "final_word_segmentation", "")
    word_segmentation(line_images, word_labels)


def perform_ocr():
    ocr = PaddleOCR(det_model_dir='./inference/Multilingual_PP-OCRv3_det_infer/',
                    rec_model_dir='./inference/svtr_2024/',
                    rec_char_dict_path='./multilingual_dict.txt',
                    show_log=True,
                    cls=False,
                    vis_font_path='/doc/fonts/shruti.ttf',
                    use_gpu=True)

    # Iterate over directories
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        for dir_name in sort_dirs(os.listdir(input_folder)):
            dir_path = os.path.join(input_folder, dir_name)
            if os.path.isdir(dir_path):
                # Sort images within the directory
                image_paths = natsorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

                # Concatenate OCR results for all images in the directory
                folder_result = []
                for img_name in image_paths:
                    img_path = os.path.join(dir_path, img_name)
                    result = ocr.ocr(img_path, cls=False)
                    txts = [line[1][0] for line in result[0]]
                    folder_result.extend(txts)

                # Write concatenated result to output file for this folder
                txt_file.write(" ".join(folder_result) + "\n")
                print("Text saved for:", dir_path)

    print("Text saved to:", output_txt_path)


st.button("Perform Segmentation ! ", on_click=main)
st.button("Perform OCR", on_click=perform_ocr)
st.button("Clear Previous saves ", on_click=remove_previous_saves)
