# import os, glob
# for filename in glob.glob('*.txt'):
#    with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
#       # do your stuff

import cv2 as cv
import  glob
for filename in glob.glob('*.jpg'):
    img = cv.imread(filename)
    cv.imshow('Dali', img)
    cv.waitKey(0)
    # print(len(filename))
    str = (filename[0:(len(filename)-4)] + '.png')
    cv.imwrite(str, img)