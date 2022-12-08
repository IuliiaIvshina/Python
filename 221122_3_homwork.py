from glob import glob

import cv2 as cv
import numpy as np
import os

# чтнеие рисунков 
def load_images(mask):
    images = {}
    for path in glob(mask):
        print(path)
        img = cv.imread(path)
        images[path] = {"img": img}
    return images

# создание базы контрольных точек
def detect_features(images):
    detector = cv.SIFT_create()
    for path, image in images.items():
        grayscale = cv.cvtColor(image["img"], cv.COLOR_BGR2GRAY)
        keypoints, descriptors = detector.detectAndCompute(grayscale, None)
        images[path]["keypoints"] = keypoints
        images[path]["descriptors"] = descriptors
        print(path, len(keypoints), "keypoints")

# сравнение кнтрольных точек
def find_matches(images, threshold, min_match_count):
    pairs = {}
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    for path1 in images.keys():
        for path2 in images.keys():
            if path1 >= path2:
                continue
            all_matches = matcher.knnMatch(images[path1]["descriptors"], images[path2]["descriptors"], 2)
            good_matches = []
            for m, n in all_matches:
                if m.distance < threshold * n.distance:
                    good_matches.append(m)
            print("pair", path1, path2, len(good_matches), "matches")
            if len(good_matches) < min_match_count:
                continue
            pairs[path1 + "|" + path2] = good_matches
    return pairs

# изображение сходных точек
def show_similar_pair(images, filenames, matches):
    path1, path2 = filenames.split("|")
    image1 = images[path1]
    image2 = images[path2]
    img_pair = np.empty((max(image1["img"].shape[0], image2["img"].shape[0]), image1["img"].shape[1] + image2["img"].shape[1], 3), dtype=np.uint8)
    cv.drawMatches(image1["img"], image1["keypoints"], image2["img"], image2["keypoints"], matches, img_pair, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    window_header = filenames.replace("|", " == ", 1)
    cv.imshow(window_header, img_pair)
    cv.waitKey()




img_dict = load_images("*.jpg")
# print(img_dict)
detect_features(img_dict)
similar = find_matches(img_dict, 0.7, 100)
for pair, good in similar.items():
    # print(pair)
    # print(good)
    show_similar_pair(img_dict, pair, good)
    print(len(pair))
    numer_Str_pair = pair.find('|')
    print(numer_Str_pair)
    delete_pair = pair[(numer_Str_pair+1):len(pair)]
    print(delete_pair)
    os.remove(delete_pair)


#def compare_kp_des_w_obj(obj_a,obj_b):
#    kp_a, des_a = obj_a
#    kp_b, des_b = obj_b
#
#    mathers = [ m for n in Flann.knnMatch(des_a,des_b,k=2)]
 
#    good = []
#    for m,n in matches:
#        if m.distance < 0.7*n.distance:
#             good.append(m)
#
#    good = [] if len(good) < MIN_MATCH_COUNT else good
#
#    return good


# print(mem)


# -------------------------
# db_dir = 'E:\ПНИПУ\Python\study\Python\picture\SDali.jpg'
# img = cv.imread('SDali.jpg', cv.IMREAD_GRAYSCALE)
# cv.imshow('Dali', img)
# cv.waitKey(0)

# orb_a = cv.ORB_create()
# kp_a = orb_a.detect(img,None)
# kp_a, des_a = orb_a.compute(img, kp_a)

# img_b = cv.imread('grayDali.jpg', cv.IMREAD_GRAYSCALE)
# cv.imshow('grayDali', img_b)
# cv.waitKey(0)

# orb_b = cv.ORB_create()
# kp_b = orb_b.detect(img_b,None)
# kp_b, des_b = orb_b.compute(img_b, kp_b)


# matches = [ m for m in Flann.knnMatch(des_a,des_b,k=2)]
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
# --------------------------------------



# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()



# cv.imshow('orb', orb)
# cv.waitKey(0)

# db_dir = 'e:\ПНИПУ\Python\study\Python\picture\*'



# def init(db_dir):
#     mem = {}

#     for cat in glob(db_dir):
#         cat_name = cat.split('/')[-1]
#         mem.update({cat_name:{}})
#         for img_name in glob ('%s' % cat):
#             img = cv.imread(img_name,0)
#             kp, des = SIFT.detectAndCompute(img, None)
#             img_key = img_name.split('/')[-1]
#             h,w = img.shape
#             mem[cat_name].update({
#                 img_key:{
#                     'keypoints': [kp, des],
#                     'width': w,
#                     'height': h,
#                 }
#             })
#         return mem

# def recognize(img, mem):
#     def compare_kp_des_w_obj(obj_a,obj_b):
#         kp_a, des_a = obj_a
#         kp_b, des_b = obj_b

#         mathers = [ m for n in Flann.knnMatch(des_a,des_b,k=2)]

#         good = []
#         for m,n in matches:
#             if m.distance < 0.7*n.distance:
#                 good.append(m)

#         good = [] if len(good) < MIN_MATCH_COUNT else good

#         return good

# def calc_cat_weight(cat, kp_a, des_a):
#     v = 0
#     for obj in mem[cat]:
#         kp_b, des_b = mem[cat][obj]
#         v += compare_kp_des_w_obj((kp_a, des_a),(kp_b, des_b))

#     return v

# def f_res():
#     res = {}
#     for cat in mem:
#         key_points = calc_cat_weight(cat, kp_shot, des_shot)
#         position = clac_obj_position(key_points),
#         res.update(
#             {
#                 cat:{
#                     'weight': len(key_points),
#                     'position': position,
#                 }
#             }
#         )
#     return(res)

# if __name__ == '__main__':
#     db = init(db_dir)
#     cv.imread('SDali.jpg', cv.IMREAD_GRAYSCALE)
#     cv.imshow('Dali', img)
#     cv.waitKey(0)
#     variants = recognize(img, db)
#     print(variants)

