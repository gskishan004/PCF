
# coding: utf-8

# In[1]:

import numpy as np
import cv2
import PIL
from scipy import misc
from PIL import Image
from pytesseract import image_to_string 
import pytesseract
import argparse
from Queue import Queue


# In[2]:

parentChildDict=dict()
direct_paths=[]
hasParentList=[]
image_path = ""
dict_centre_vert=dict()
dict_centres_path=dict()


# In[3]:

def negImg(vertForCropping, image_path):#Converting the img to negative and removing text
    image_source = image_path
    gray_img = cv2.imread(image_source)
    
    rect,thresh = cv2.threshold(gray_img,127,255,0)
    newImg = cv2.bitwise_not(thresh)
   
    for vert in vertForCropping:
        #print "whitened rect vert ", vert
        cv2.rectangle(newImg,(vert[0],vert[2]),(vert[1],vert[3]),(255,255,255),-1)

   

    cv2.imwrite("123neg.png",newImg)
    #cv2.imshow("sdd", newImg)
    return newImg


# In[4]:

# making all the other rectangles black except the start and end

def iswhite(value):
    #if value == (255,255,255):
    #   return True
    if value != (0,0,0) and value != (127,127,127):
        return True
  
    return False

def getNewVertForCropping(vertForCropping, rect1_centre,  rect2_centre, dict_centre_vert, image_path):
    newVertForCropping=[]
    
    newImg = negImg(vertForCropping, image_path)

    
    
    rect1_x1,  rect1_x2, rect1_y1, rect1_y2 = dict_centre_vert[rect1_centre][0]
    
    rect2_x1,  rect2_x2, rect2_y1,rect2_y2 = dict_centre_vert[rect2_centre][0]

    #rect1_x1, rect1_y1, rect1_x2, rect1_y2 = exclude_p1[0][0], exclude_p1[0][1], exclude_p1[1][0], exclude_p1[1][1]
    #rect2_x1, rect2_y1, rect2_x2, rect2_y2 = exclude_p2[0][0], exclude_p2[0][1], exclude_p2[1][0], exclude_p2[1][1]
  
    
    for vert in vertForCropping:

        #this box which is being considered leave it white
        if vert[0] == rect1_x1 and vert[1] ==rect1_x2 and vert[2] ==rect1_y1 and vert[3] ==rect1_y2:
            pass
        elif vert[0] == rect2_x1 and vert[1] ==rect2_x2 and vert[2] ==rect2_y1 and vert[3] ==rect2_y2:
            pass
        else:
            #make all other boxes black
            cv2.rectangle(newImg,(vert[0]-6,vert[2]-6),(vert[1]+6,vert[3]+6),(0,0,0),-1)
            cv2.imwrite('blackRectNewNeg.png', newImg)
            #print "image written"
            
    
    return True


# In[5]:

#code for finding the path b/w start and the end


def getadjacent(n):
    x,y = n
    return [(x-1,y),(x,y-1),(x+1,y),(x,y+1),(x-4,y),(x,y-4),(x+4,y),(x,y+4)]

def BFS(start, end, pixels):

    queue = Queue()
    queue.put([start]) # Wrapping the start tuple in a list

    while not queue.empty():
        
        path = queue.get() 
        #print path
        pixel = path[-1]

        if pixel == end:
            return path

        for adjacent in getadjacent(pixel):
            x,y = adjacent
            
            if iswhite(pixels[x,y]):
                pixels[x,y] = (127,127,127) # see note
                new_path = list(path)
                new_path.append(adjacent)
                queue.put(new_path)

    #print "Queue has been exhausted. No answer was found."



    # invoke: python mazesolver.py <mazefile> <outputfile>[.jpg|.png|etc.]


# In[6]:



def ifPathExists(vertForCropping, rect1_centre, rect2_centre, dict_centre_vert, image_path, dict_centre_text):
    getNewVertForCropping(vertForCropping, rect1_centre, rect2_centre, dict_centre_vert, image_path)
    base_img = Image.open("blackRectNewNeg.png")
    base_pixels = base_img.load()

    path = BFS(rect1_centre, rect2_centre, base_pixels)
    
    path_img = Image.open("blackRectNewNeg.png")
    path_pixels = path_img.load()
    
    
    try:
        for position in path:
            x,y = position
            path_pixels[x,y] = (255,0,0) # red
        
        
        #print 'Path found between' , rect1_centre, ' and ', rect2_centre
        if setParentChild([rect1_centre,rect2_centre]):
            path_img.save('path.png')
            
            #setting dict_centres_path
            if (rect1_centre[1]>rect2_centre[1]):
                dict_centres_path[(rect1_centre,rect2_centre)] = path
            else:
                dict_centres_path[(rect2_centre,rect1_centre)] = path
            
            
            print dict_centre_text[rect1_centre],  " <----> ", dict_centre_text[rect2_centre]
            direct_paths.append([rect1_centre,rect2_centre])
            return True
        else:
            return False
    except:
        #print "Processing ...... Finding connections"
        #print 'No path found between' , rect1_centre, ' and ', rect2_centre
        return False
    


# In[7]:

#combinations = [ ((5,16),(7,-6)),((1,6), (3,-4)),((4,6), (2,8))]
def sortCombinations(combinations):
    
    initial_sort=[]
    for c in combinations:
        initial_sort.append(sorted(c, key=lambda tup: tup[1]))
    
    sorted_comb = sorted(initial_sort, key=lambda tup: tup[0][1])
   
    return sorted_comb
    
#sortCombinations(combinations)


# 

# In[8]:

def getChildren(p):
    for key in parentChildDict:
        if p == key:
            return parentChildDict[p]
    return []

#get parent of a given child

def getParent(c):
    for key in parentChildDict.keys():
        if c in parentChildDict[key]:
            hasParentList.append(c)
            return key
    return "NA"

def alreadyHasParent(c):
    if c in hasParentList:
        return True
    else:
        return False


# In[9]:

#find parent and child from direct_paths
#and putting them in dictionay



def setParentChild(dp):
    print "#####################",dp
    y0 = dp[0][1]
    y1 = dp[1][1]

    p1 = getParent(dp[0])
    p2 = getParent(dp[1])
    #print "--------",p1,p2
    
    
    if( p1 != "NA" and p2 != "NA" and p1 == p2):
        print y0, y1, "inside p1==p2"
        return False
    
    else :
    #parent = y1 lava
        #if y0>=y1 and not (p1 != "NA" and p2 != "NA"):
        if y0>=y1:
            print y1, y0, "in else loop"
            #print "PARENT     |     CHILD"
            #print dp[1],dp[0]
            if dp[1] in parentChildDict.keys():
                parentChildDict[dp[1]].append(dp[0])

            else :    
                parentChildDict[dp[1]] =[dp[0]]
            return True

        #elif y1>y0 and not (p1 != "NA" and p2 != "NA"):
        elif y1>y0:
            print y0, y1, "in else loop"
            #print "PARENT     |     CHILD"
            #print dp[0],dp[1]
            if dp[0] in parentChildDict.keys():
                parentChildDict[dp[0]].append(dp[1])

            else:    
                parentChildDict[dp[0]] =[dp[1]]
            return True
    return False


# In[10]:

# getting all the combinations

# arab.jpg
# uniq = [(375, 553, 1035, 1079), (376, 556, 974, 1016), (377, 556, 896, 954), (376, 556, 839, 878), (875, 1011, 813, 911), (82, 270, 805, 861), (614, 723, 788, 874), (375, 554, 762, 823), (875, 1011, 713, 799), (1061, 1192, 702, 753), (613, 723, 688, 774), (375, 554, 688, 748), (622, 985, 101, 169), (1244, 1373, 688, 760), (312, 512, 600, 675), (74, 274, 700, 775), (75, 274, 901, 974), (799, 1055, 600, 674), (575, 749, 601, 674), (636, 931, 276, 337)]
# dict_centre_vert = {(176, 833): [(82, 270, 805, 861)], (668, 831): [(614, 723, 788, 874)], (927, 637): [(799, 1055, 600, 674)], (466, 995): [(376, 556, 974, 1016)], (662, 637): [(575, 749, 601, 674)], (943, 756): [(875, 1011, 713, 799)], (943, 862): [(875, 1011, 813, 911)], (174, 737): [(74, 274, 700, 775)], (466, 858): [(376, 556, 839, 878)], (464, 1057): [(375, 553, 1035, 1079)], (412, 637): [(312, 512, 600, 675)], (466, 925): [(377, 556, 896, 954)], (1308, 724): [(1244, 1373, 688, 760)], (1126, 727): [(1061, 1192, 702, 753)], (668, 731): [(613, 723, 688, 774)], (803, 135): [(622, 985, 101, 169)], (464, 718): [(375, 554, 688, 748)], (174, 937): [(75, 274, 901, 974)], (464, 792): [(375, 554, 762, 823)], (783, 306): [(636, 931, 276, 337)]}
# centres = [(464, 1057), (466, 995), (466, 925), (466, 858), (943, 862), (176, 833), (668, 831), (464, 792), (943, 756), (1126, 727), (668, 731), (464, 718), (803, 135), (1308, 724), (412, 637), (174, 737), (174, 937), (927, 637), (662, 637), (783, 306)]

# complex.jpg
# uniq = [(714, 823, 795, 833), (506, 615, 716, 754), (714, 823, 713, 751), (29, 138, 675, 713), (506, 615, 634, 672), (714, 823, 632, 670), (29, 138, 614, 652), (29, 138, 553, 591), (714, 823, 552, 590), (506, 615, 552, 590), (316, 426, 552, 590), (1091, 1201, 503, 541), (880, 990, 503, 541), (28, 138, 491, 529), (316, 426, 471, 509), (1091, 1201, 422, 460), (880, 990, 422, 460), (316, 426, 391, 429), (880, 990, 260, 298), (256, 366, 260, 298), (411, 584, 139, 167), (411, 584, 28, 56), (505, 615, 798, 836), (316, 426, 634, 672), (6, 116, 260, 298), (505, 615, 470, 510), (742, 852, 390, 429), (591, 701, 390, 429), (191, 302, 390, 429), (66, 177, 390, 430), (1091, 1201, 340, 379), (880, 990, 340, 379), (755, 866, 259, 299), (629, 739, 259, 299), (504, 615, 259, 299), (379, 490, 259, 299), (131, 242, 259, 298), (1091, 1202, 573, 613)]
# dict_centre_vert = {(434, 279): [(379, 490, 259, 299)], (1146, 593): [(1091, 1202, 573, 613)], (186, 278): [(131, 242, 259, 298)], (311, 279): [(256, 366, 260, 298)], (121, 410): [(66, 177, 390, 430)], (684, 279): [(629, 739, 259, 299)], (560, 653): [(506, 615, 634, 672)], (1146, 441): [(1091, 1201, 422, 460)], (83, 572): [(29, 138, 553, 591)], (560, 490): [(505, 615, 470, 510)], (371, 571): [(316, 426, 552, 590)], (560, 571): [(506, 615, 552, 590)], (61, 279): [(6, 116, 260, 298)], (83, 510): [(28, 138, 491, 529)], (371, 653): [(316, 426, 634, 672)], (371, 490): [(316, 426, 471, 509)], (935, 359): [(880, 990, 340, 379)], (246, 409): [(191, 302, 390, 429)], (559, 279): [(504, 615, 259, 299)], (560, 735): [(506, 615, 716, 754)], (497, 42): [(411, 584, 28, 56)], (935, 441): [(880, 990, 422, 460)], (810, 279): [(755, 866, 259, 299)], (83, 633): [(29, 138, 614, 652)], (935, 522): [(880, 990, 503, 541)], (1146, 522): [(1091, 1201, 503, 541)], (797, 409): [(742, 852, 390, 429)], (935, 279): [(880, 990, 260, 298)], (768, 732): [(714, 823, 713, 751)], (560, 817): [(505, 615, 798, 836)], (768, 571): [(714, 823, 552, 590)], (768, 651): [(714, 823, 632, 670)], (768, 814): [(714, 823, 795, 833)], (371, 410): [(316, 426, 391, 429)], (497, 153): [(411, 584, 139, 167)], (83, 694): [(29, 138, 675, 713)], (1146, 359): [(1091, 1201, 340, 379)], (646, 409): [(591, 701, 390, 429)]}
# centres = [(768, 814), (560, 735), (768, 732), (83, 694), (560, 653), (768, 651), (83, 633), (83, 572), (768, 571), (560, 571), (371, 571), (1146, 522), (935, 522), (83, 510), (371, 490), (1146, 441), (935, 441), (371, 410), (935, 279), (311, 279), (497, 153), (497, 42), (560, 817), (371, 653), (61, 279), (560, 490), (797, 409), (646, 409), (246, 409), (121, 410), (1146, 359), (935, 359), (810, 279), (684, 279), (559, 279), (434, 279), (186, 278), (1146, 593)]
 
# hsbc.jpg
# uniq = [(145, 223, 146, 172), (127, 242, 208, 235), (159, 308, 13, 41), (249, 380, 208, 236), (363, 466, 145, 172), (229, 360, 146, 172), (9, 140, 146, 172), (160, 306, 67, 91)]
# dict_centre_vert = {(74, 159): [(9, 140, 146, 172)], (233, 27): [(159, 308, 13, 41)], (184, 221): [(127, 242, 208, 235)], (233, 79): [(160, 306, 67, 91)], (294, 159): [(229, 360, 146, 172)], (314, 222): [(249, 380, 208, 236)], (414, 158): [(363, 466, 145, 172)], (184, 159): [(145, 223, 146, 172)]}
# centres = [(184, 159), (184, 221), (233, 27), (314, 222), (414, 158), (294, 159), (74, 159), (233, 79)]
# vertForCropping = uniq

#------------------------
def find_paths(vertForCropping,dict_centre_vert,centres,image_path, clientCentre, dict_centre_text):
    a = iter(centres)
    import itertools
    combinations=[]
    combinations.extend(itertools.combinations(a, 2))

    # print "-------------------Combinations--------------------"
    # print combinations

    sorted_comb = sortCombinations(combinations)

    # print "---------------Sorted Combinations-----------------"
    # print sorted_comb


    #finding if path is possible b/w all the combinations of rectnagles

    num_of_direct_paths = 0

    print ('\n')
    print ".... Finding relationships between entities ...... "
    
    for c in sorted_comb:

        if (ifPathExists( vertForCropping, c[0], c[1], dict_centre_vert, image_path, dict_centre_text)):
            num_of_direct_paths = num_of_direct_paths + 1

    print ('\n')
    print "Total number of connections = ", num_of_direct_paths
    
    drawHierarchyLines(image_path)
    
    print ('\n')
    print "Number of levels excluding client = ", noOfLvls(clientCentre, centres)-1
    
    print "DCP-",dict_centres_path
    
    return parentChildDict


# In[11]:

def drawHierarchyLines(image_path):
    im = cv2.imread(image_path)
    x = 7
    for k in parentChildDict.keys():

        children = getChildren(k)
        for child in children:
            cv2.line(im,(k[0],k[1]),(child[0],child[1]),((271-x)%255,(5+x)%255,(127+x)%255),10)
        x = x + 13
    cv2.imwrite("output.png",im)


# In[12]:

def noOfLvls(centreOfClient,centres):
    max_lvl = 0
    
    for centre in centres:
        lvl = 1
        curr = centre
        while (getParent(curr) != "NA"): 
            lvl = lvl + 1
            curr = getParent(curr) 
        if lvl > max_lvl:
            max_lvl = lvl
 
    return max_lvl


# In[ ]:



