
# coding: utf-8

# In[20]:

import os
import sys
import glob
import subprocess
from subprocess import STDOUT, check_output

import dlib
from skimage import io

from xml.etree import ElementTree as et
import xml.dom.minidom


# In[ ]:

# code to kill the process if it takes too long
# <NOTE> the time is in Seconds

import subprocess, shlex
from threading import Timer

def kill_proc(proc, timeout):
  timeout["value"] = True
  proc.kill()

def run(cmd, timeout_sec):
  proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  timeout = {"value": False}
  timer = Timer(timeout_sec, kill_proc, [proc, timeout])
  timer.start()
  stdout, stderr = proc.communicate()
  timer.cancel()
  return proc.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"), timeout["value"]


# In[41]:

def removeUselessText(filename):
    l1 = "<?xml version='1.0' encoding='ISO-8859-1'?>"
    l2 = "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>"
    
    f = open(filename,"r+")
    d = f.readlines()
    f.seek(0)
    for i in d:
        if not (l1 in i or l2 in i):
            f.write(i)
            print i
    f.truncate()
    f.close()


# In[ ]:

# Function to serach for relevent data b/w 2 points

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    


# In[47]:

# Putting relevent data of xml2 to xml1

def combineXML(xml1,xml2):

    f = open(xml2,"r+")
    lines_to_insert = find_between( f.read(), "<images>", "</images>" )
    #print 'lines_to_insert - \n',lines_to_insert 
    f.close()

    with open(xml1, "r+") as f:
        a = [x.rstrip() for x in f]
        index = 0
        for item in a:
            if item.startswith("</images>"):
                a.insert(index, lines_to_insert) 
                break
            index += 1
        # Go to start of file and clear it
        f.seek(0)
        f.truncate()
        # Write each line back
        for line in a:
            f.write(line + "\n")


# In[82]:

# import os, os.path, sys
# import glob
# from xml.etree import ElementTree

# def runXMLComb():
#     xml_files = ['input_images_ML/training.xml','input_images_ML/trainingTemp.xml']
#     xml_element_tree = None
#     for xml_file in xml_files:
#         data = ElementTree.parse(xml_file).getroot()
#         print ElementTree.tostring(data), '\n --------------'
#         for result in data.iter('images'):
#             if xml_element_tree is None:
#                 xml_element_tree = data 
#                 insertion_point = xml_element_tree.findall("./images")[0]
#             else:
#                 insertion_point.extend(result) 
#     if xml_element_tree is not None:
#         print '-------------------'
#         print ElementTree.tostring(xml_element_tree)

# runXMLComb()


# In[84]:

import sys
from xml.etree import ElementTree as et


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class XMLCombiner(object):
    def __init__(self, filenames):
        assert len(filenames) > 0, 'No filenames!'
        # save all the roots, in order, to be processed later
        self.roots = [et.parse(f).getroot() for f in filenames]

    def combine(self):
        for r in self.roots[1:]:
            # combine each element with the first one, and update that
            self.combine_element(self.roots[0], r)
        # return the string representation
        return et.ElementTree(self.roots[0])

    def combine_element(self, one, other):
        """
        This function recursively updates either the text or the children
        of an element if another element is found in `one`, or adds it
        from `other` if not found.
        """
        # Create a mapping from tag name to element, as that's what we are fltering with
        mapping = {(el.tag, hashabledict(el.attrib)): el for el in one}
        for el in other:
            if len(el) == 0:
                # Not nested
                try:
                    # Update the text
                    mapping[(el.tag, hashabledict(el.attrib))].text = el.text
                except KeyError:
                    # An element with this name is not in the mapping
                    mapping[(el.tag, hashabledict(el.attrib))] = el
                    # Add it
                    one.append(el)
            else:
                try:
                    # Recursively process the element, and update it in the same way
                    self.combine_element(mapping[(el.tag, hashabledict(el.attrib))], el)
                except KeyError:
                    # Not in the mapping
                    mapping[(el.tag, hashabledict(el.attrib))] = el
                    # Just add it
                    one.append(el)


# image_folder = "input_images_ML/"
# xml1 = image_folder + "/training.xml"
# xml2 = image_folder + "/trainingTemp.xml" 

# r = XMLCombiner((xml1, xml2)).combine()
# print '-'*20
# print et.tostring(r.getroot())


# In[22]:

def train(image_folder,append):
    
    
    
    if append == 0:
    
        #code to open createXML for the noob user with all the required params

        cmd = 'createXML.exe -c' + image_folder + '/training.xml '+ image_folder
        run(cmd,5)

        cmd = 'createXML.exe ' + image_folder + '/training.xml'
        run(cmd,100)
        
    # <NOTE> <IN PROGRESS> include code to write new XML to the old XML and use the latter for the training 
    
    elif append == 1:
            
        #code to open createXML for the noob user with all the required params

        cmd = 'createXML.exe -c' + image_folder + '/trainingTemp.xml '+ image_folder
        run(cmd,5)
        
        
        cmd = 'createXML.exe ' + image_folder + '/trainingTemp.xml'
        run(cmd,100)
        
        dlib.hit_enter_to_continue()
        
        # doing all the magic stuff to append the new XML to the old one
        
        xml1 = image_folder + "/training.xml"
        xml2 = image_folder + "/trainingTemp.xml" 

        removeUselessText(xml1)
        removeUselessText(xml2)


        
        #combineXML(xml1,xml2)
        r = XMLCombiner((xml1, xml2)).combine()
        
        with open(xml1, "r+") as f:
            f.write(et.tostring(r.getroot()))


        #Convert the XML to better format before saving it for the training as there may be some improper indentation 

        
    
    
    # setting option in dlib

    options = dlib.simple_object_detector_training_options()

    # symmetric detector
    options.add_left_right_image_flips = True

    # SVM C parameter.larger value will lead to overfitting
    options.C = 1

    # Tell the code how many CPU cores your computer has for the fastest training.
    options.num_threads = 4
    options.be_verbose = True
    
    training_xml_path = os.path.join(image_folder, "training.xml")
    #testing_xml_path = os.path.join(image_folder, "testing.xml")
    
    # saving the detector as detector.svm with input as the xml file after doing the training 
    

    dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)
    
    # Printing the accuracy with training data

    print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
    
    # Doing the detection 
    detector = dlib.simple_object_detector("detector.svm")

    # Looking at the HOG filter the machine has learned. 
    win_det = dlib.image_window()
    win_det.set_image(detector)



# In[7]:

def predict(image_folder):
     # Doing the detection 
    detector = dlib.simple_object_detector("detector.svm")

    # Looking at the HOG filter the machine has learned. 
    win_det = dlib.image_window()
    win_det.set_image(detector)
    
    # running the detector for all the images in the folder
    print("Using ML the images for detection...")
    win = dlib.image_window()
    for f in glob.glob(os.path.join(image_folder, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)
        dets = detector(img)
        print("Number of detections: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        dlib.hit_enter_to_continue()


# In[8]:

# ................ uncomment after testing ................  

if len(sys.argv) != 3:
    print "Kindly use this format to run the code : python F_ML.py <options> <path to the image dir>"
    exit()

user_choice = sys.argv[1]
image_folder = sys.argv[2]


# In[9]:

# # ................ comment after testing ................  

# user_choice = "train"
# image_folder = "input_images_ML/"


# In[10]:

if user_choice == "train":
    train(image_folder,0)
elif user_choice == "trainAppend":
    train(image_folder,1)
elif user_choice == "predict":
    predict(image_folder)
else :
    print "Wrong parameter given as <option> please check and try again"

exit()

