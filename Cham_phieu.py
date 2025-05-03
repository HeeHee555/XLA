# import các thư viện cần thiết
import imutils
import numpy as np
import cv2
from math import ceil
from collections import defaultdict
from scipy import stats
import os
import glob
import json


if True:
    Xsbd = []
    sbd = [24,33]
    for i in range(6):
        for j in range(10):
            Xsbd.append([sbd[0]+i*47,sbd[1]+j*66])

    Xmdt = []
    mdt = [24,33]
    for i in range(3):
        for j in range(10):
            Xmdt.append([mdt[0]+i*47,mdt[1]+j*66])
        
    Xp1 = []
    p1 = [118,94]
    for i in range(10):
        for j in range(4):
            Xp1.append([p1[0]+j*94,p1[1]+i*56])

    Xp2 = []
    p2 = [117,157]
    for i in range(2):
        for j in range(4): 
            for k in range(2):
                Xp2.append([p2[0]+i*188+k*94,p2[1]+j*54])

    Xp3 = []
    p3 = [110,180]
    for x in range(6):
        for i in range(4):
            for j in range(12):
                Xp3.append([p3[0]+342*x+i*58,p3[1]+j*57])

def is_bubble_filled(bubble_roi, threshold=0.3, avg = 100):
    if len(bubble_roi.shape) == 3:
        bubble_roi = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(bubble_roi, avg, 255, cv2.THRESH_BINARY_INV)
    total_pixels = thresh.size
    eroded = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    filled_pixels = cv2.countNonZero(eroded)
    fill_ratio = filled_pixels / total_pixels
    return fill_ratio > threshold

def thresholding(img, avg = 100):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, avg, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    cv2.imshow('thresh', eroded)
    cv2.waitKey(0)
    return eroded

def preprocess_image(image):
    img = cv2.imread(image, 0)
    clahe = cv2.createCLAHE(clipLimit= 4, tileGridSize=(4, 4))
    img_c = clahe.apply(img)
    blurred = cv2.fastNlMeansDenoising(img_c, h=50, templateWindowSize=7, searchWindowSize=21)
    thresh = cv2.adaptiveThreshold( blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 3)
    return thresh, img

def find_part(thresh, img):
    y, x = thresh.shape
    info = thresh[:int(y/2.5), int(x/1.5):]
    pt1 = thresh[int(y/3.3):int(y/1.6)]
    pt2 = thresh[int(y/2):int(y/1.3)]
    pt3 = thresh[int(y/1.6):]
    img_info = img[:int(y/2.5), int(x/1.5):]
    img_pt1 = img[int(y/3.3):int(y/1.6)]
    img_pt2 = img[int(y/2):int(y/1.3)]
    img_pt3 = img[int(y/1.6):]
    return info, pt1, pt2, pt3, img_info, img_pt1, img_pt2, img_pt3

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_quad(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2)
        return order_points(quad)
    else:
        x, y, w, h = cv2.boundingRect(c)
        quad = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
        return order_points(quad)

def deskew(img, pts, width, height):
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def homography():
    template = cv2.imread('THPT2025.png', 0)
    img = cv2.imread(gb_sheet, 0)
    # denoised = cv2.fastNlMeansDenoising(img, h=20)
    Padding = 300
    color = [255, 255, 255]
    scanned = cv2.copyMakeBorder(img, Padding, Padding,Padding,Padding, cv2.BORDER_CONSTANT, value=color)
    
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(template, None) 
    kp2, des2 = akaze.detectAndCompute(scanned, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
  
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h, w = template.shape
    aligned = cv2.warpPerspective(scanned, H, (w, h))
    info = []
    info.append(aligned[425:1085,1775:2050])
    info.append(aligned[425:1085,2130:2270])
    pt1 = []
    for i in range(4): pt1.append(aligned[1220:1855, 200+530*i:645+530*i])
    pt2 = []
    for i in range(4): pt2.append(aligned[1945:2315, 200+530*i:645+530*i])
    pt3 = []
    pt3.append(aligned[2440:3250,200:2235])

    return info, pt1, pt2, pt3

def get_contours(thresh):
    height, width = thresh.shape[:2]
    area = height * width
    c = []
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        if cv2.contourArea(i) > area / 20:
            c.append(i)
    c = sorted(c, key=lambda c: cv2.boundingRect(c)[0])
    return c

def get_info(thresh, part):
    w = 280
    h = 660
    c = get_contours(thresh)
    if len(c) != 2:
        global homograph
        homograph = homography()
        c = homograph[0]
        box = c[0]
        box2 = c[1]
    else:
        pts = get_quad(c[0])
        box = deskew(part, pts, w,h)
        pts2 = get_quad(c[1])
        box2 = deskew(part, pts2, int(w/2),h)
    
    std_dev = np.std(box)
    avg = np.mean(box) - 0 - std_dev
    SBD = ''
    for idx,i in enumerate(Xsbd):
        roi = box[i[1]-18:i[1]+18, i[0]-18:i[0]+18]
        if is_bubble_filled(roi, 0.7, avg):
            cv2.circle(box, tuple(i), radius=18, color=(0, 255, 0), thickness=2)
            SBD += str(idx%10)
    if len(box.shape) == 2: box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
    y, x, z = box.shape
    output[425:425+y,1775:1775+x] = box
    std_dev = np.std(box2)
    avg = np.mean(box2) - std_dev
    MDT = ''
    thresholding(box2, avg)
    for i in range(8,0,1):
        MDT = ''
        threshold = 0.1*i
        box2x = box2.copy()
    for idx,i in enumerate(Xmdt):
        roi = box2[i[1]-18:i[1]+18, i[0]-18:i[0]+18]
        if is_bubble_filled(roi, 0.1, avg):
            cv2.circle(box2, tuple(i), radius=18, color=(0, 255, 0), thickness=2)
            MDT += str(idx%10)
    if len(box2.shape) == 2: box2 = cv2.cvtColor(box2, cv2.COLOR_GRAY2BGR)
    y, x, z = box2.shape
    output[425:425+y,2130:2130+x] = box2
    return SBD, MDT

def get_part_1(thresh, part):
    c = get_contours(thresh)
    if len(c) != 4:
        if 'homograph' not in globals():
            global homograph
            homograph = homography()
        b = homograph[1]
    else:
        b = []
        for i in c:
            pts = get_quad(i)
            box = deskew(part, pts, 460, 640)
            b.append(box)

    
    P1 = [[] for _ in range(40)]
    for i, box in enumerate(b):
        std_dev = np.std(box)
        avg = np.mean(box) - 0 - std_dev
        thresholding(box, avg)
        if len(box.shape) == 2: box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR) 
        for idx,ii in enumerate(Xp1):
            bubble_roi = box[ii[1]-18:ii[1]+18, ii[0]-18:ii[0]+18]
            if is_bubble_filled(bubble_roi, 0.5, avg):
                cv2.circle(box, tuple(ii), radius=18, color=(0, 255, 0), thickness=2)
                P1[(idx//4)+10*i] = [idx%4]     
        y, x, z = box.shape
        output[1220:1220+y,200+530*i:200+x+530*i] = box
            
    return P1

def get_part_2(thresh, part):
    c = get_contours(thresh)
    if len(c) != 4:
        if 'homograph' not in globals():
            global homograph
            homograph = homography()
        b = homograph[2]
    else:
        b = []
        for i in c:
            pts = get_quad(i)
            box = deskew(part, pts, 460, 360)
            b.append(box)
    P2 = [[] for _ in range(32)]
    for i, box in enumerate(b):
        std_dev = np.std(box)
        avg = np.mean(box) - 0 - std_dev
        thresholding(box, avg)
        if len(box.shape) == 2: box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
        for idx, ii in enumerate(Xp2):
            bubble_roi = box[ii[1]-18:ii[1]+18, ii[0]-18:ii[0]+18]
            if is_bubble_filled(bubble_roi,0.5, avg):
                cv2.circle(box, tuple(ii), radius=18, color=(0, 255, 0), thickness=2)
                P2[(idx//2)+8*i] = [(idx+1)%2]
        y, x, z = box.shape
        output[1945:1945+y,200+530*i:200+x+530*i] = box
    return P2

def get_part_3(thresh, part):
    c = get_contours(thresh)
    if len(c) != 1 :
        if 'homograph' not in globals():
            global homograph
            homograph = homography()
        b = homograph[3]
    else:
        b = []
        for i in c:
            pts = get_quad(i)
            box = deskew(part, pts, 2060, 860)
            b.append(box)
    P3 = ['' for _ in range(6)]
    no_bubble = [1, 12, 24, 36, 37, 
                 49, 60, 72, 84, 85, 
                 97, 108, 120, 132, 133, 
                 145, 156, 168, 180, 181,
                 193, 204, 216, 228, 229,
                 241, 252, 264, 276, 277]
    for box in b:
        std_dev = np.std(box)
        avg = np.mean(box) - 0 - std_dev
        thresholding(box, avg)
        for idx, ii in enumerate(Xp3): 
            bubble_roi = box[ii[1]-18:ii[1]+18, ii[0]-18:ii[0]+18]
            if is_bubble_filled(bubble_roi,0.7,avg) and idx not in no_bubble:
                    cv2.circle(box, tuple(ii), radius=18, color=(0, 255, 0), thickness=2)
                    if idx % 12 == 0:
                        P3[(idx//48)] += "-"
                    elif idx % 12 == 1:
                        P3[(idx//48)] += ","
                    else:
                        P3[(idx//48)] += str((idx % 12)-2)
        if len(box.shape) == 2: box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
        y, x, z = box.shape
        output[2440:2440+y,200:200+x] = box
    return P3

def cham_phieu(sheet):
    global gb_sheet
    gb_sheet = sheet

    global output
    output = cv2.imread('Output.jpg')

    basename = os.path.basename(sheet)
    json_name = os.path.splitext(basename)[0] + '.json'
    original_path = os.path.join('PhieuQG', basename)
    output_path = os.path.join('Output', basename)
    json_path = os.path.join('Json', json_name)

    os.system('clear')
    if 'homograph' in globals():
        del globals()['homograph']
    thresh, img = preprocess_image(sheet)
    part = find_part(thresh, img)
                    
    info = part[0]
    info_img = part[4]
    pt1 = part[1]
    pt1_img = part[5]
    pt2 = part[2]
    pt2_img = part[6]
    pt3 = part[3]
    pt3_img = part[7]

    sbd, mdt = get_info(info, info_img)
    fc = get_part_1(pt1, pt1_img)
    tf = get_part_2(pt2, pt2_img)
    dg = get_part_3(pt3, pt3_img)
    
    result = {
        "org": original_path,
        "out": output_path,
        "warn": "",
        "err": [],
        "res": {
            "fc": {str(i+1): fc[i] if i < len(fc) else [] for i in range(40)},
            "tf": {str(i+1): tf[i] if i < len(tf) else [] for i in range(32)},
            "dg": {str(i+1): dg[i] if i < len(dg) else "" for i in range(6)},
        },
        "sbd": sbd,
        "mdt": mdt
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    filename = os.path.basename(sheet)
    cv2.imwrite(f'Output/{filename}', output)
    cv2.imshow("Output", output)
    cv2.waitKey(0)


cham_phieu('PhieuQG/PhieuQG.0124.jpg')

# folder_path = 'PhieuQG'
# all_files = glob.glob(f'{folder_path}/*')
# for jpg_file in all_files:
#     cham_phieu(jpg_file)
