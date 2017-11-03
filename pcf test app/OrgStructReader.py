
# coding: utf-8
'''
             *     ,MMM8&&&.            *
                  MMMM88&&&&&    .
                 MMMM88&&&&&&&
     *           MMM88&&&&&&&&
                 MMM88&&&&&&&&
                 'MMM88&&&&&&'
                   'MMM8&&&'      *    
          |\___/|     /\___/\
          )     (     )    ~( .              '
         =\     /=   =\~    /=
           )===(       ) ~ (
          /     \     /     \
          |     |     ) ~   (
         /       \   /     ~ \
         \       /   \~     ~/
  ____/\_/\__  _/_/\_/\__~__/_/\_/\_/\_/\_/\_
  |  |  |  |( (  |  |  | ))  |  |  |  |  |  |     
  |  |  |  | ) ) |  |  |//|  |  |  |  |  |  |
  |  |  |  |(_(  |  |  (( |  |  |  |  |  |  |     
  |  |  |  |  |  |  |  |\)|  |  |  |  |  |  |
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |   


Things to do:
    > green line - away from rect - crop verts - get text from orig image
                                        > empty rectangles - ignore
                                        > if dup text - ignore
    
    
    
    do differnt img processing for text and differnt for image and use the 1st one for ocr and second for rectangle
    
    Alternative to tesseract ? IBM datacap
    
    Logic for case where two sibling boxes can be connected
    Logic for case where parent is there on the left most side instead of top 
    
'''

def main(inp1, inp2):

            import numpy as np
            import cv2
            import PIL
            from scipy import misc
            from PIL import Image
            from pytesseract import image_to_string 
            import pytesseract
            import argparse
            import re
            from GVision import gvision_ocr_text
            from GVision import gvision_ocr_perc
            from F_PATHS import find_paths
            import sys
            import math
            import os
            import shutil
            from F_Sign import verifySign 


            # In[2]:

            dict_centre_vert= dict()
            dict_centre_rect= dict()
            dict_vfc_rect = dict()
            dict_vert_text = dict()
            dict_bvert_perc = dict()
            dict_centre_text = dict()
            dict_perc_rectCentre = dict()

            centres= []


            #image_source = raw_input("Enter the image name :")
            #clientName = raw_input("Enter the client name :")

            image_source = inp1
            clientName = inp2

            import ntpath
            out_file_name = os.path.splitext(ntpath.basename(image_source))[0]
            dir = 'output/'+ out_file_name
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)

            f_out = dir+ "/out"
            f = open(f_out, 'w')
            img = cv2.imread(image_source)
            orig = img.copy()


            # In[5]:

            def angle_cos(p0, p1, p2):
                d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
                return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


            # In[6]:

            #finding area of the org img

            height, width, channels = img.shape
            figArea = height*width

            minArea = figArea * 0.0014
            maxArea = figArea * 0.6
            maxCenLen = int(min([height,width]) * 0.022) 

            #print "minArea ", minArea
            #print "maxArea ", maxArea
            #print "maxCenLen", maxCenLen


            # In[7]:

            def find_rects(img):
                img_org = cv2.GaussianBlur(img, (5, 5), 0)
                rects = []

                img_neg = cv2.bitwise_not(img_org)

                #getting RGB from img and so when we take just R, in a way it will be gray scale (0-255)
                #aka r,g,b = cv2.split(img)
                for i in range (0,2):
                    if i == 0 :
                        img = img_org
                    elif i == 1 :
                        img = img_neg

                    for gray in cv2.split(img):

                        #thresh from 0 - 255 with increments of 26
                        for thrs in xrange(0, 255, 26):
                            if thrs == 0:
                                bin = cv2.Canny(gray, 0, 50, apertureSize=5)

                                #dilating to thicken the edges for edge detection
                                bin = cv2.dilate(bin, None)

                            else:

                                #retval will be same as thrs unless until Otsu’s Binarizatio is usd
                                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)

                            #hierar = [Next, Previous, First_Child, Parent] 
                            # can later use it to remove rectangle inside another by checking hierarchy[0,i,3], where i != -1
                            _,contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                            for cnt in contours:
                                cnt_len = cv2.arcLength(cnt, True)
                                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                                cnt_area = cv2.contourArea(cnt)
                                if len(cnt) == 4 and cnt_area > minArea and cnt_area < maxArea and cv2.isContourConvex(cnt):

                                    #removing 1 outer list
                                    cnt = cnt.reshape(-1, 2)

                                    #the angles should roughly be 90, cos(90) = 0
                                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                                    if max_cos < 0.1:
                                        rects.append(cnt)
                return rects


            # In[8]:

            def r2centresDiff( (cX,cY), centres):
                for i in range(-maxCenLen,maxCenLen):
                    for j in range(-maxCenLen,maxCenLen):
                        if(cX+i,cY+j) in centres:
                            return False

                return True


            # In[9]:

            def find_minMax(rect):
                minX = min([rect[0][0],rect[1][0],rect[2][0],rect[3][0]])
                maxX = max([rect[0][0],rect[1][0],rect[2][0],rect[3][0]])

                minY = min([rect[0][1],rect[1][1],rect[2][1],rect[3][1]])
                maxY = max([rect[0][1],rect[1][1],rect[2][1],rect[3][1]])

                return (minX, maxX, minY, maxY)



            # In[10]:

            def find_unique_rects(rects):
                uniq_rects=[]


                for rect in rects:
                    minX, maxX, minY, maxY = find_minMax(rect)

                    cX = (minX + maxX)/2
                    cY = (minY + maxY)/2

                    if ( r2centresDiff((cX,cY),centres)):
                        centre = (cX,cY)
                        centres.append(centre)
                        uniq_rects.append(rect)
                        dict_centre_rect[centre] = rect

                return uniq_rects


            # In[11]:

            def isRhombus(rect):


                dist=[]
                for i in range (0, len(rect)):
                    x1 = rect[i%4][0]
                    y1 = rect[i%4][1]

                    x2= rect[(i+1)%4][0]
                    y2= rect [(i+1)%4][1]
                    dist.append(math.sqrt((x2 - x1)**2 +  (y2 - y1)**2))

                if abs(dist[0] + dist[2] - dist[1] - dist[3]) <= 5:
                    print "-----------------------"
                    print "Found a rhombus"
                    print dist[0], dist[1], dist[2], dist [3]
                    print "-----------------------"
                    return True

                return False


            # In[12]:

            def removeEmptyRect(vfc, choice):
                uniq = []
                print "Doing OCR ....... Finding Text"
                for vert in vfc:
                    minX, maxX, minY, maxY = vert
                    text= ocr(vert, choice)
                    #s = text.splitlines()
                    if u'' == text:
                        #print "Found blank rectangle, hence removing it"
                        cropped_img = orig[minY-1:maxY+1, minX-1:maxX+1] # Crop from y1:y2, x1:x2
                        magnified_img = misc.imresize(cropped_img, 50)
                        gray_img = cv2.cvtColor(magnified_img, cv2.COLOR_BGR2GRAY)
                        th_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                        #cv2.imwrite("emptyrect.png",th_img)

                    else:
                        print text
                        s = str(text)
                        stext = s.replace('\n', '')
                        cX = (minX + maxX)/2
                        cY = (minY + maxY)/2
                        centre = (cX, cY)
                        dict_centre_text[(cX, cY)] = stext
                        dict_vert_text[vert] = stext
                        uniq.append(vert)
                        #f.write("%s\n" % stext)
                return uniq


            # In[13]:

            def ocr(vert, choice):


                minX, maxX, minY, maxY = vert
                rect = dict_vfc_rect[(minX, maxX, minY, maxY)]


                #making the borders of the rhombus white

                orig2 = cv2.imread(image_source)

                for i in range (0, len(rect)):
                    x1 = rect[i%4][0]
                    y1 = rect[i%4][1]

                    x2= rect[(i+1)%4][0]
                    y2= rect[(i+1)%4][1]

                    cv2.line(orig2,(x1,y1),(x2,y2),(255,255,255),20)

                #if rhombus/diamond dont rotate
                if isRhombus(rect):

                    cropped_img = orig2[ minY+1:maxY-1, minX+1:maxX-1]
                    cv2.imwrite("cropped2.png",cropped_img)

                #else rotate if rectangle
                else: 
                    transformImg(rect)

                if choice == "gtesseract":
                    orig3 = cv2.imread("cropped2.png")
                    height, width, channels= orig3.shape
                    #cropped_img = orig3[10:height-10, 10:width-10] # Crop from y1:y2, x1:x2
                    cropped_img = orig3.copy()

                    magnified_img = misc.imresize(cropped_img, 50)
                    gray_img = cv2.cvtColor(magnified_img, cv2.COLOR_BGR2GRAY)
                    th_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                    magnified_img = misc.imresize(th_img, 500)

                    kernel = np.ones((2,2), np.uint8)
                    th_img = cv2.dilate(magnified_img, kernel, iterations=4)

                    cv2.imwrite("cropped.png",th_img)
                    text= image_to_string(Image.open('cropped.png'))


                elif choice == "gvision":

                    or_im = cv2.imread("cropped2.png")
                    mag_im = misc.imresize(or_im, 500)
                    cv2.imwrite("cropped2.png",mag_im)
                    text = gvision_ocr_text(["cropped2.png"])


                else:
                    print "INVALID CHOICE"

                return text


            # In[14]:

            gray_img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            th_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            #cv2.imwrite("12345.png",th_img)


            # In[15]:

            def removeRectInsideOther(rects):

                vertForCropping= []
                vertToIgnore=[]
                uniq_rects = []

                combinationList = []
                for x in range(0, len(rects)):
                    combinationList.append(x)

                for r in rects:
                    vertForCropping.append(find_minMax(r))


                a = iter(combinationList)
                import itertools
                combinations=[]
                combinations.extend(itertools.combinations(a, 2))


                for c in combinations:
                    minX1, maxX1, minY1, maxY1 = find_minMax(rects[c[0]])
                    minX2, maxX2, minY2, maxY2 = find_minMax(rects[c[1]])

                    #second rectangle lies inside first = use 1st
                    if minX2 > minX1 and minY2 > minY1 and maxX2 < maxX1 and maxY2 < maxY1:
                        vertToIgnore.append((minX2, maxX2, minY2, maxY2))



                    #opposite to first condition
                    elif minX2 < minX1 and minY2 < minY1 and maxX2 > maxX1 and maxY2 > maxY1:
                         vertToIgnore.append((minX1, maxX1, minY1, maxY1))



                uniq_vertForCropping = [x for x in vertForCropping if x not in vertToIgnore]


                return uniq_vertForCropping


            # In[16]:

            from transform import four_point_transform

            def transformImg(rect):
                import os
                import time

                image = cv2.imread(image_source)
                pts = np.array(eval("[(rect[0][0],  rect[0][1]), (rect[1][0], rect[1][1]), (rect[2][0], rect[2][1]), (rect[3][0],rect[3][1])]"), dtype = "float32")

                warped = four_point_transform(image, pts)
                cv2.imwrite("cropped2.png", warped)





            # In[17]:

            def makeVFCRect_Dict(rects):
                for r in rects:
                    minX, maxX, minY, maxY = find_minMax(r)
                    dict_vfc_rect[(minX, maxX, minY, maxY)]= r



            # In[18]:

            def getClientCentre(clientName):
                minY = 1000
                clientCentre = ""
                for centre in dict_centre_text.keys():
                    name = dict_centre_text[centre]
                    if str(clientName) == str(name):
                        print "matched ", name
                        cY = centre[1]
                        if cY < minY:
                            minY = cY
                            clientCentre = centre

                return clientCentre


            # In[19]:


            def makeCentreVFC_Dict(uniq):
                uniq_centres= []
                for vfc in uniq:
                    minX, maxX, minY, maxY = vfc
                    cX = (minX + maxX)/2
                    cY = (minY + maxY)/2

                    centre = (cX,cY)
                    dict_centre_vert[centre] = [vfc]
                    uniq_centres.append(centre)
                return uniq_centres



            # In[20]:

            def percRelations(uniq_centres):

                for d in dict_vert_perc.keys():
                    min_dist = 100000000000000000
                    perc = dict_vert_perc[d]
                    pcX = (d[0] + d[4])/2
                    pcY = (d[1] + d[5])/2
                    rc = ""
                    for rect_centre in uniq_centres:
                        rcX = rect_centre[0]
                        rcY = rect_centre[1]
                        dist = math.sqrt((rcX - pcX)**2 +  (rcY - pcY)**2)
                        if dist < min_dist:
                            min_dist = dist
                            rc = (rcX, rcY)

                    import unicodedata
                    perc = unicodedata.normalize('NFKD', perc).encode('ascii','ignore')
                    perc.replace("'", '')
                    dict_perc_rectCentre[rc] = perc

                #print dict_perc_rectCentre

                f.write('\n')
                for rect_centre in dict_perc_rectCentre.keys():
                    perc = dict_perc_rectCentre[rect_centre]
                    rect_text = dict_centre_text[rect_centre]
                    print "\nPercentage ", perc, " is associated with ", rect_text
                    f.write("\nPercentage %s is associated with %s" % (perc, rect_text))


            # In[21]:

            import sys
            import win32api

            if verifySign(image_source):
                print "signature"
                # win32api.MessageBox(0, 'Signature Matched', 'Success')

            else:
                return 0
                # win32api.MessageBox(0, 'Signature Not Matched', 'Warning')


            rects = find_rects(img)

            uniq_rects = find_unique_rects(rects)

            makeVFCRect_Dict(uniq_rects)

            vfc = removeRectInsideOther(uniq_rects)


            print "............ Start finding rectangles ..............."
            # uniq_vfc = removeEmptyRect(vfc, "gtesseract")
            uniq_vfc = removeEmptyRect(vfc, "gvision")
            # print "after removing empty ",len(uniq_vfc)

            uniq_centres = makeCentreVFC_Dict(uniq_vfc)

            print "\nTotal number of entities found = ", len(uniq_vfc)
            f.write("\nTotal number of entitites found = %s" % len(uniq_vfc))


            out_file_name = os.path.splitext(ntpath.basename(image_source))[0]


            img = cv2.imread(image_source)
            for vert in uniq_vfc:
                cv2.rectangle(img,(vert[0]-2,vert[2]-2),(vert[1]+2,vert[3]+2),(0,0,255),3)
            s = dir + "/text.jpg"
            cv2.imwrite(s,img)

            print "\nDoing OCR ....... Finding Percentages %"

            img = cv2.imread(image_source)
            dict_vert_perc = gvision_ocr_perc(image_source)
            for d in dict_vert_perc.keys():
                cv2.rectangle(img,(d[0]-2,d[1]-2),(d[4]+2,d[5]+2),(21,213,55),3)
            s = dir + "/perc.jpg"
            cv2.imwrite(s,img)




            # In[22]:

            #print dict_centre_text
            #clientName  = "HSBC Holdings plc"
            clientCentre = getClientCentre(clientName)


            # In[23]:

            percRelations(uniq_centres)


            # In[24]:

            parentChildDict = find_paths(uniq_vfc,dict_centre_vert,uniq_centres,image_source, clientCentre, dict_centre_text,f)

            return 1


