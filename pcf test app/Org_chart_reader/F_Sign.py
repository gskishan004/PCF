
# coding: utf-8

# In[1]:

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys


# In[2]:

def findSimilarityBW(doc,sign):

    
    MIN_MATCH_COUNT = 10
    
    img1 = cv2.imread(doc,0)          # document
    img2 = cv2.imread(sign,0)      # signature template to match

    # Initiate SIFT detector
    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)



    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (4,40,60), # draw matches 
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    #plt.imshow(img3, 'gray'),plt.show()
    
    return len(good),img3


def verifySign(doc):

	#doc = "docs/test.jpg"
	signsDir = "signs"

	# doc =  sys.argv[1]
	# signDir = sys.argv[2]


	# In[10]:

	signsList = []
	import os
	for file in os.listdir(signsDir):
	    if file.endswith(".jpg"):
	        signsList.append(os.path.join(file))


	# In[11]:

	min = 0
	bestMatchImg = ""
	for s in signsList:
	    sPath= signsDir+"/"+s
	    print sPath
	    per,img= findSimilarityBW(doc,sPath)
	    if (per>min):
	        min = per
	        bestMatchImg = sPath
	        
	    print per,"% - Match with : ",s

	print "-----------------------------------------------------------------------\n\n"
	if min>54:
		print "Signature found in the Document ",doc," and it matches with ",bestMatchImg 
		per,img= findSimilarityBW(doc,bestMatchImg)
		plt.imshow(img, 'gray'),plt.show()
		return True

	else:
		return False





