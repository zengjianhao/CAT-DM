import cv2
import sys
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

input_path = sys.argv[1]
mask_path = sys.argv[2]

category = input_path.split('/')[-1]

def get_img_agnostic_dresses(im, im_parse, pose_data):
    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 1).astype(np.float32) +
                    (parse_array == 2).astype(np.float32) +
                    (parse_array == 3).astype(np.float32) +
                    (parse_array == 11).astype(np.float32) +
                    (parse_array == 17).astype(np.float32))
    parse_lower = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32) +
                    (parse_array == 8).astype(np.float32))
    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[11] - pose_data[8])
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

    r = int(length_a / 16) + 1
    for i in [8, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 8]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 11, 8]], 'gray', 'gray')

    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')


    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'gray', width=r*12)
    for i in [8, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*10, pointy-r*6, pointx+r*10, pointy+r*6), 'gray', 'gray')
    for i in [9, 10, 12, 13]:
        if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*20)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*10, pointy-r*6, pointx+r*10, pointy+r*6), 'gray', 'gray')
    
    for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
        mask_arm = Image.new('L', (768, 1024), 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
        mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    for parse_id, pose_ids in [(9, [11, 12, 13]), (10, [8, 9, 10])]:
        mask_leg = Image.new('L', (768, 1024), 'white')
        mask_leg_draw = ImageDraw.Draw(mask_leg)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_leg_draw.ellipse((pointx-r*10, pointy-r*6, pointx+r*10, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
                continue
            mask_leg_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_leg_draw.ellipse((pointx-r*10, pointy-r*6, pointx+r*10, pointy+r*6), 'black', 'black')
        mask_leg_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')
        parse_leg = (np.array(mask_leg) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_leg * 255), 'L'))
    
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    
    return agnostic

def get_img_agnostic_upper_body(im, im_parse, pose_data):
    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 1).astype(np.float32) +
                    (parse_array == 2).astype(np.float32) +
                    (parse_array == 3).astype(np.float32) +
                    (parse_array == 11).astype(np.float32) +
                    (parse_array == 17).astype(np.float32))
    parse_lower = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 8).astype(np.float32) +
                    (parse_array == 9).astype(np.float32) +
                    (parse_array == 10).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))

    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[11] - pose_data[8])
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

    r = int(length_a / 16) + 1

    for i in [8, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 8]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 11, 8]], 'gray', 'gray')

    
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
        mask_arm = Image.new('L', (768, 1024), 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
        mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    return agnostic

def get_img_agnostic_lower_body(im, im_parse, pose_data):
    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 1).astype(np.float32) +
                    (parse_array == 2).astype(np.float32) +
                    (parse_array == 3).astype(np.float32) +
                    (parse_array == 11).astype(np.float32) +
                    (parse_array == 17).astype(np.float32))
    parse_upper = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32) +
                    (parse_array == 16).astype(np.float32))
    parse_lower = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32) +
                    (parse_array == 8).astype(np.float32))
    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[11] - pose_data[8])
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'gray', width=r*12)
    for i in [8, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*10, pointy-r*6, pointx+r*10, pointy+r*6), 'gray', 'gray')
    for i in [9, 10, 12, 13]:
        if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*20)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*10, pointy-r*6, pointx+r*10, pointy+r*6), 'gray', 'gray')
    parse_lower = parse_lower.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(parse_lower)
    agnostic_draw.rectangle([x, y, x+w, y+h], fill='gray', outline='gray')
    for parse_id, pose_ids in [(9, [11, 12, 13]), (10, [8, 9, 10])]:
        mask_leg = Image.new('L', (768, 1024), 'white')
        mask_leg_draw = ImageDraw.Draw(mask_leg)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_leg_draw.ellipse((pointx-r*10, pointy-r*6, pointx+r*10, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
                continue
            mask_leg_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_leg_draw.ellipse((pointx-r*10, pointy-r*6, pointx+r*10, pointy+r*6), 'black', 'black')
        mask_leg_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')
        parse_leg = (np.array(mask_leg) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_leg * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    return agnostic

os.makedirs(mask_path, exist_ok=True)

for im_name in tqdm(os.listdir(osp.join(input_path, 'images'))):
    if im_name.endswith("1.jpg"):
        continue
    pose_name = im_name.replace('0.jpg', '2.json')
    try:
        with open(osp.join(input_path, 'keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data[:, :2] * 2
    except IndexError:
        print(pose_name)
        continue
    
    label_name = im_name.replace('0.jpg', '4.png')
    im_label = Image.open(osp.join(input_path, 'label_maps', label_name)).convert('P')
    im = np.ones((1024,768))
    im = Image.fromarray(np.uint8(im * 255), 'L')
    if category == "dresses":
        agnostic = get_img_agnostic_dresses(im, im_label, pose_data)
    if category == "upper_body":
        agnostic = get_img_agnostic_upper_body(im, im_label, pose_data)
    if category == "lower_body":
        agnostic = get_img_agnostic_lower_body(im, im_label, pose_data)

    agnostic = np.array(agnostic)
    agnostic = (agnostic <= 254).astype(int)
    agnostic = agnostic * 255.0
    agnostic = Image.fromarray(np.uint8(agnostic)).convert('L')
    agnostic.save(osp.join(mask_path, im_name.replace('.jpg', '.png')))
    
print("Done")