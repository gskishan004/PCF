
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
import math


# In[2]:


parentChildDict=dict()
nodeParentsDict=dict()
direct_paths=[]
hasParentList=[]
image_path = ""
dict_centre_vert=dict()
dict_centres_path=()
dict_ftext_text=dict()
dict_nodes_path= dict()

consider_again=[]


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


def getadjacent1(n):
    x,y = n
    return [(x-1,y),(x,y-1),(x+1,y),(x,y+1),(x-4,y),(x,y-4),(x+4,y),(x,y+4)]
    #return [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]

def getadjacent2(n):
    x,y = n
    return [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]

def BFS(start, end, pixels, image_path, multiHeaded= False):
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    
    queue = Queue()
    queue.put([start]) # Wrapping the start tuple in a list

    while not queue.empty():
        
        path = queue.get() 
        #print path
        pixel = path[-1]

        if pixel == end:
            return path
        
        if not multiHeaded: 
            for adjacent in getadjacent1(pixel):
                x,y = adjacent

                if x<width and x>0 and y<height and y>0:
                    if iswhite(pixels[x,y]):
                        pixels[x,y] = (127,127,127) # see note
                        new_path = list(path)
                        new_path.append(adjacent)
                        queue.put(new_path)
                else:
                    pass
                
        else:
            for adjacent in getadjacent2(pixel):
                x,y = adjacent

                if x<width and x>0 and y<height and y>0:
                    if iswhite(pixels[x,y]):
                        pixels[x,y] = (127,127,127) # see note
                        new_path = list(path)
                        new_path.append(adjacent)
                        queue.put(new_path)
                else:
                    pass

    #print "Queue has been exhausted. No answer was found."
    return False


    # invoke: python mazesolver.py <mazefile> <outputfile>[.jpg|.png|etc.]


# In[6]:




def ifPathExists(vertForCropping, rect1_centre, rect2_centre, dict_centre_vert, image_path, dict_centre_text):
    getNewVertForCropping(vertForCropping, rect1_centre, rect2_centre, dict_centre_vert, image_path)
    base_img = Image.open("blackRectNewNeg.png")
    base_pixels = base_img.load()

    path = BFS(rect1_centre, rect2_centre, base_pixels, image_path)
    
    path_img = Image.open("blackRectNewNeg.png")
    path_pixels = path_img.load()
    
    
    try:
        path_coords=[]
        for position in path:
            x,y = position
            path_pixels[x,y] = (255,0,0) # red
            path_coords.append((x,y))
        
        
        y_rect1  = rect1_centre[1]
        y_rect2  = rect2_centre[1]
        #store all found paths
        #rect1 is parent
        if y_rect1 < y_rect2:
            dict_nodes_path[(rect1_centre,rect2_centre)] = path_coords
            direct_paths.append([rect1_centre,rect2_centre])
        else: #rect2 is parent
            dict_nodes_path[(rect2_centre,rect1_centre)] = path_coords
            direct_paths.append([rect2_centre,rect1_centre])
                
         
        #print 'Path found between' , rect1_centre, ' and ', rect2_centre
        
        # path invalidation rules ..... store only the true paths
        if setParentChild_new([rect1_centre,rect2_centre], vertForCropping, dict_centre_vert, image_path):
            path_img.save('path.png')
            
            text1 = dict_centre_text[rect1_centre]
            text2 = dict_centre_text[rect2_centre]
            #print "\n", text1,  " <----> ", text2
            #f.write("\n%s <----> %s" % (dict_centre_text[rect1_centre], dict_centre_text[rect2_centre]))
            if text1 in dict_ftext_text.keys():
                dict_ftext_text[text1].append(text2)
            else:
                dict_ftext_text[text1] = [text2]
            
            
            print "Path found between ", dict_centre_text[rect1_centre]," and ", dict_centre_text[rect2_centre]
            return True
        else:
            return False
    except:
        #print "Processing ...... Finding connections"
        #print 'No path found between' , rect1_centre, ' and ', rect2_centre
        return False
    


# In[7]:



def findDistance(point1,point2):
    x1,y1 = point1
    x2,y2 = point2
    
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return (dist)

def isIntersecting(path1,path2):

    minDist = 10000000
    finalP1= []
    finalP2= []
    finalP1.append(sorted(path1, key=lambda tup: tup[1]))
    finalP2.append(sorted(path2, key=lambda tup: tup[1]))
    
    if (len(finalP1[0]) < len(finalP2[0])):
        for i in range (0,len(finalP1[0])):
            for j in range(0,len(finalP2[0])):
                currDist = findDistance(finalP1[0][i],finalP2[0][j])
                
                if (currDist < minDist ):
                    minDist = currDist
    
    else:
        for i in range (0,len(finalP2[0])):
            for j in range(0,len(finalP1[0])):
                currDist = findDistance(finalP2[0][i],finalP1[0][j])
                
                if (currDist < minDist ):
                    minDist = currDist
        
    #print "minDistance -------------------- ",minDist
    if minDist <5:
        return True
    
    return False


# In[8]:


# for children having the exact same parent
def checkPathsAgain(dp, vertForCropping, dict_centre_vert, image_path):
    
    count = 0
    #print "................. INSIDE CHECK PATHS AGAIN ..........."
    #print "Checking for ", dp

    common_parent = getParent(dp[0]) #or dp[1]
    #print "common parent of " , dp[0], " and ", dp[1], " is ", common_parent
    
    path1 = dict_nodes_path[(common_parent[0],dp[0])]
    path2 = dict_nodes_path[(common_parent[0],dp[1])]
    #print " @@@@@@@ ", len(path1), len(path2)
    
    #delete the path coordinates which are inside the rectangle
    #start comparing the intersection only from the boundaries of the boxes
    bottom_right_vert1  = dict_centre_vert[dp[0]]
    bottom_right_vert2  = dict_centre_vert[dp[1]]
    
        
    for coord in path1:
        y = coord[1]
        if y <= bottom_right_vert1[0][3]:
            path1.remove(coord)
    
    
    for coord in path2:
        y = coord[1]
        if y <= bottom_right_vert2[0][3]:
            path2.remove(coord)
    
    
    #if paths with common parent are overlapping, ie - the case of hsbc.jpg  
    #blacken the previously found paths and try to find if a new path exists between the 2 children
    
    #if len(list(set(path1).intersection(set(path2)))) > 0:
    if isIntersecting(path1,path2):
        
        #print "child paths are intersecting"
        child1 = dp[0]
        child2 = dp[1]
        getNewVertForCropping(vertForCropping, child1, child2, dict_centre_vert, image_path)
        b_img = cv2.imread("blackRectNewNeg.png")
        
        
        
        #blacken the paths
        for x,y in path1:
            cv2.circle(b_img,(x,y),5, (0,0,0), -1)  
        
        for x,y in path2:
            cv2.circle(b_img,(x,y),5, (0,0,0), -1)  
        
        #whiten the centres
        cv2.circle(b_img,(dp[0][0],dp[0][1]),7, (255,255,255), -1) 
        cv2.circle(b_img,(dp[1][0],dp[1][1]),7, (255,255,255), -1)
        
        cv2.imwrite("temp.jpg",b_img)
#         p = cv2.imread("temp.jpg" )
#         cv2.imshow("temp.jpg", p)
#         cv2.waitKey(0)
        
        
        base_img = Image.open("temp.jpg")
        base_pixels = base_img.load()
        path = BFS(child1, child2, base_pixels, image_path, True)

        path_img = Image.open("temp.jpg")
        path_pixels = path_img.load()

            
        y0 = dp[0][1]
        y1 = dp[1][1]
        
        #print "y0 = " ,y0
        #print "y1 = " ,y1
    
        try:
            for position in path:
                x,y = position
                path_pixels[x,y] = (255,0,0) # red
                path_img.save('path.png')

            if y0>y1:
                #print "y1 is parent"
                if dp[0] not in nodeParentsDict.keys():
                    nodeParentsDict[dp[0]] = [dp[1]]
                else:
                    nodeParentsDict[dp[0]].append(dp[1])

                if dp[1] not in parentChildDict.keys():
                    parentChildDict[dp[1]] = [dp[0]]
                else:
                    parentChildDict[dp[1]].append(dp[0])


            #y0 is parent
            else:
                #print "y0 is parent"
                if dp[1] not in nodeParentsDict.keys():
                    nodeParentsDict[dp[1]] = [dp[0]]
                else:
                    nodeParentsDict[dp[1]].append(dp[0])

                if dp[0] not in parentChildDict.keys():
                    parentChildDict[dp[0]] = [dp[1]]
                else:
                    parentChildDict[dp[0]].append(dp[1])

#             p = cv2.imread("path.png")
#             cv2.imshow("path.png", p)
#             cv2.waitKey(0)
            return True
        except:
            #print "No second path found b/w children"
            return False
    
    #else when paths with common parent are not overlapping, ie - the case of ac55.jpg
    else:
        #print "child paths are not overlapping"
        y0 = dp[0][1]
        y1 = dp[1][1]
        
        #print "y0 = " ,y0
        #print "y1 = " ,y1
        if y0>y1:
                #print "y1 is parent"
                if dp[0] not in nodeParentsDict.keys():
                    nodeParentsDict[dp[0]] = [dp[1]]
                else:
                    nodeParentsDict[dp[0]].append(dp[1])

                if dp[1] not in parentChildDict.keys():
                    parentChildDict[dp[1]] = [dp[0]]
                else:
                    parentChildDict[dp[1]].append(dp[0])


        #y0 is parent
        else:
            #print "y0 is parent"
            if dp[1] not in nodeParentsDict.keys():
                nodeParentsDict[dp[1]] = [dp[0]]
            else:
                nodeParentsDict[dp[1]].append(dp[0])

            if dp[0] not in parentChildDict.keys():
                parentChildDict[dp[0]] = [dp[1]]
            else:
                parentChildDict[dp[0]].append(dp[1])
        return True
        
    


    


# In[9]:


def bubbleSort_x1(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            x1_1 = alist[i][0][0]
            x1_2 = alist[i+1][0][0]
            if x1_1> x1_2:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
    
    #print alist
    return alist
    

def bubbleSort_y2(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            y2_1 = alist[i][1][1]
            y2_2 = alist[i+1][1][1]
            if y2_1> y2_2:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
    
    #print alist
    return alist    
    
    
def bubbleSort_x2(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            x2_1 = alist[i][1][0]
            x2_2 = alist[i+1][1][0]
            if x2_1> x2_2:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
    
    #print alist
    return alist


# In[10]:


#combinations = [ ((5,16),(7,-6)),((1,6), (3,-4)),((4,6), (2,8))]
def sortCombinations(combinations, dict_centre_text):

    initial_sort=[]
    final_sort=[]

    # ********** INTERNAL SORT B/W Y1 AND Y2 ***********

    new_nl = []

    for c in combinations:
        l = sorted(c, key=lambda tup: tup[1])
        new_nl.append(tuple(l))


    # **************** SORT BY Y1 *****************************
    initial_sort.append(sorted(combinations, key=lambda tup: tup[0][1]))

    sorted_by_y1 = initial_sort[0]
    #print sorted_by_y1, "\n"

    
        
   
    # **************** SORT BY Y2 *****************************          

    l=[]
    sorted_by_y2=[]

    s= sorted_by_y1
    
    for j in range(0, len(s)-1):
        if s[j][0][1] == s[j+1][0][1]:
            if j< len(s)-2:
                l.append(s[j])
            else:
                l.append(s[j])
                l.append(s[j+1])
                ans = bubbleSort_y2(l)
                sorted_by_y2.extend(ans)
                l=[]
                break
        else:
            if j< len(s)-2:
                l.append(s[j])
                ans = bubbleSort_y2(l)
                sorted_by_y2.extend(ans)
                l=[]
            else:
                l.append(s[j])
                ans = bubbleSort_y2(l)
                sorted_by_y2.extend(ans)
                l=[]
                l.append(s[j+1])
                ans = bubbleSort_y2(l)
                sorted_by_y2.extend(ans)
                break
    
    
    
    #print "sorted by y2"
    #print sorted_by_y2
    
     #   **************** SORT BY X1 *****************************
    l=[]
    sorted_by_x1=[]
    
    s = sorted_by_y2
    
    for j in range(0, len(s)-1):
        if s[j][1][1] == s[j+1][1][1]:
            if j< len(s)-2:
                l.append(s[j])
            else:
                l.append(s[j])
                l.append(s[j+1])
                ans = bubbleSort_x1(l)
                sorted_by_x1.extend(ans)
                l=[]
                break

        else:
            if j< len(s)-2:
                l.append(s[j])
                ans = bubbleSort_x1(l)
                sorted_by_x1.extend(ans)
                l=[]
            else:
                l.append(s[j])
                ans = bubbleSort_x1(l)
                sorted_by_x1.extend(ans)
                l=[]
                l.append(s[j+1])
                ans = bubbleSort_x1(l)
                sorted_by_x1.extend(ans)
                break
    
    #print "sorted by x1"
    #print sorted_by_x1

    
#     for c in sorted_by_x1:
#         print dict_centre_text[c[0]], "    ", dict_centre_text[c[1]]
#         print "______________________"
        
    return sorted_by_x1
    
    # **************** SORT BY X2 *****************************

#     print "\n\n\n"

#     l=[]
#     final_sort=[]

#     s = sorted_by_y2
#     for j in range(0, len(s)-1):
#         if s[j][1][1] == s[j+1][1][1]:
#             if j< len(s)-2:
#                 l.append(s[j])
#             else:
#                 l.append(s[j])
#                 l.append(s[j+1])
#                 ans = bubbleSort_x2(l)
#                 final_sort.extend(ans)
#                 l=[]
#                 break
#         else:
#             if j< len(s)-2:
#                 l.append(s[j])
#                 ans = bubbleSort_x2(l)
#                 final_sort.extend(ans)
#                 l=[]
#             else:
#                 l.append(s[j])
#                 ans = bubbleSort_x2(l)
#                 final_sort.extend(ans)
#                 l=[]
#                 l.append(s[j+1])
#                 ans = bubbleSort_x2(l)
#                 final_sort.extend(ans)
#                 break


    
    

#     print final_sort
#     return final_sort


# In[11]:



def getChildren(p):
    for key in parentChildDict:
        if p == key:
            return parentChildDict[p]
    return []

#get parent of a given child

def getParent(node):
    if node in nodeParentsDict.keys():
        return nodeParentsDict[node]
    
    else:
        return "N"

    
    
   


# In[12]:


#find parent and child from direct_paths
#and putting them in dictionay


def compareList(list1,list2):
                
    for l1 in list1:
        for l2 in list2:
            if not abs(cmp(l1, l2)) and l1!="N" and l2!="N":
                print l1
                print l2
                print "l1 = l2"
                return True
    return True
            
def setParentChild_new(dp, vertForCropping, dict_centre_vert, image_path):
    
    y0 = dp[0][1]
    y1 = dp[1][1]

    p1 = getParent(dp[0])
    p2 = getParent(dp[1])
    #print dp[0], "-->", p1,"  ", dp[1],"-->" , p2
    
    if p1==p2 and p1!="N" and p2!="N":
        #print "both have exactly same defined parents, send to checkPathsAgain"
        ans = checkPathsAgain(dp, vertForCropping, dict_centre_vert, image_path)
        #print "return value from checkPaths is " , ans
        return ans
        
    else:
        #if compareList(p1,p2):
        # y1 is parent
        if y0>y1:
            if dp[0] not in nodeParentsDict.keys():
                nodeParentsDict[dp[0]] = [dp[1]]
            else:
                nodeParentsDict[dp[0]].append(dp[1])

            if dp[1] not in parentChildDict.keys():
                parentChildDict[dp[1]] = [dp[0]]
            else:
                parentChildDict[dp[1]].append(dp[0])


        #y0 is parent
        else:
            if dp[1] not in nodeParentsDict.keys():
                nodeParentsDict[dp[1]] = [dp[0]]
            else:
                nodeParentsDict[dp[1]].append(dp[0])

            if dp[0] not in parentChildDict.keys():
                parentChildDict[dp[0]] = [dp[1]]
            else:
                parentChildDict[dp[0]].append(dp[1])

        return True

    return False


# In[13]:


def multiHeaded(vertForCropping, dict_centre_vert, image_path, dict_centre_text):
    count2=0
    img = cv2.imread(image_path)

    for node in dict_centre_text.keys():
        #print "%%%%%%%%%%%%", node, "%%%%%%%%%%%%%"
        if getParent(node) == "N":
            #print "topmost node is : " ,node
            break
            
    
    
    children_of_topmost_node = parentChildDict[node]
    #print children_of_topmost_node
    top_y_coord = node[1]
    
    for child in children_of_topmost_node:
        y_child = child[1]
        if abs(top_y_coord - y_child) <= 11:
            #print "multiheaded with ",node, " and " ,child
             
            getNewVertForCropping(vertForCropping, node, child, dict_centre_vert, image_path)
            b_img = cv2.imread("blackRectNewNeg.png")
            
            path = dict_nodes_path[(node,child)]
            for x,y in path:
                cv2.circle(b_img,(x,y),5, (0,0,0), -1)  
            cv2.imwrite("temp.jpg",b_img)
            
            base_img = Image.open("temp.jpg")
            base_pixels = base_img.load()
            path = BFS(node, child, base_pixels, image_path, True)

            path_img = Image.open("temp.jpg")
            path_pixels = path_img.load()
               
            
            try:
                for position in path:
                    x,y = position
                    path_pixels[x,y] = (255,0,0) # red
                    path_img.save('path.png')
                #print "This paths actually  exists"
                
                
            except:
                #print "removing the incorrect path"
                children_of_topmost_node.remove(child)
                parentChildDict[node] = children_of_topmost_node
                count2 = count2 + 1
    return count2


# In[14]:


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
def find_paths(vertForCropping,dict_centre_vert,centres,image_path, clientCentre, dict_centre_text,f):
    a = iter(centres)
    import itertools
    combinations=[]
    combinations.extend(itertools.combinations(a, 2))

    # print "-------------------Combinations--------------------"
    # print combinations

    sorted_comb = sortCombinations(combinations, dict_centre_text)
    #print sorted_comb

    #finding if path is possible b/w all the combinations of rectnagles

    num_of_direct_paths = 0

    print ('\n')
    print ".... Finding relationships between entities ...... "
    
    f.write('\n')
    for c in sorted_comb:

        if (ifPathExists( vertForCropping, c[0], c[1], dict_centre_vert, image_path, dict_centre_text)):
            num_of_direct_paths = num_of_direct_paths + 1
    
    # check paths again in case of samsame parent
    #count = checkPathsAgain(vertForCropping, dict_centre_vert, image_path, dict_centre_text)
    
    count2 = multiHeaded(vertForCropping, dict_centre_vert, image_path, dict_centre_text)
    
    n = num_of_direct_paths - count2
    print "\n\nTotal number of connections = ", n
    f.write("\n\nTotal number of connections = %d" % n)
    
    fwrite_paths(f)
    drawHierarchyLines(image_path)
    
    no_of_levels = noOfLvls(parentChildDict,clientCentre)
    print "\n\nNumber of levels excluding client = ", no_of_levels
    f.write("Number of levels excluding client = %d" % no_of_levels)
    f.close()
    return parentChildDict


# In[15]:


def fwrite_paths(f):
    for key in dict_ftext_text.keys():
        l = dict_ftext_text[key]
        f.write("\n%s" %key)
        f.write("\n---------------------------------------- ")
        for item in l:
            f.write("\n%s" %item)
    
        f.write("\n\n\n")
    


# In[16]:


def drawHierarchyLines(image_path):
    import os
    import ntpath
    import shutil
    out_file_name = os.path.splitext(ntpath.basename(image_path))[0]
    s = "output/" + out_file_name + "/paths.jpg"
    
    im = cv2.imread(image_path)
    x = 25
    noOfColors = len(parentChildDict.keys())
    noOfColors = int (255 / noOfColors)
    
    for k in parentChildDict.keys():

        children = parentChildDict[k]
        for child in children:
            #cv2.line(im,(k[0],k[1]),(child[0],child[1]),((255-x)%255,(x)%255,(x)%255),10)
            cv2.line(im,(k[0],k[1]),(child[0],child[1]),((x+23)%255,(x-79)%255,(x+131)%255),10)
        x = x + noOfColors
    cv2.imwrite(s,im)


# In[17]:


# def noOfLvls(centreOfClient,centres):
#     max_lvl = 0
    
#     for centre in centres:
#         lvl = 1
#         curr = centre
#         while (getParent(curr) != "N"): 
#             lvl = lvl + 1
#             curr = getParent(curr) 
#         if lvl > max_lvl:
#             max_lvl = lvl
 
#     return max_lvl

def curr_lvl(parentChildDict, val,clientCentre):
    if val not in parentChildDict.keys():
        return 0
    
    if parentChildDict[val] == "N" or parentChildDict[val] ==clientCentre:
        return 0
    
    return 1 + max(curr_lvl(parentChildDict, val2,clientCentre) for val2 in parentChildDict[val])

def noOfLvls(parentChildDict,clientCentre):
    max_lvl = (max(curr_lvl(parentChildDict, val,clientCentre) for val in parentChildDict.keys()))
    return max_lvl


# In[18]:


# image_path = "input_images/hsbc.jpg"
# import os
# out_file_name = os.path.splitext(image_path)[0]
# print out_file_name
# f = open(out_file_name, 'w')
# clientCentre = (0,0)
# find_paths(vertForCropping,dict_centre_vert,centres,image_path, clientCentre, dict_centre_text,f)

