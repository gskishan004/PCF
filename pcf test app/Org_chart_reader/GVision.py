
# coding: utf-8

# In[1]:

from base64 import b64encode
from os import makedirs
from os.path import join, basename
from sys import argv
import json
import requests
import re

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
api_key = "AIzaSyB-ToYyW9OqyposPWp8dgtpi1lqwTwb5p8"
dict_vert_perc=dict()

def make_image_data_list(image_filenames):
    """
    image_filenames is a list of filename strings
    Returns a list of dicts formatted as the Vision API
        needs them to be
    """
    img_requests = []
    for imgname in image_filenames:
        with open(imgname, 'rb') as f:
            ctxt = b64encode(f.read()).decode()
            img_requests.append({
                    'image': {'content': ctxt},
                    'features': [{
                        'type': 'TEXT_DETECTION',
                        'maxResults': 1
                    }]
            })
    return img_requests

def make_image_data(image_filenames):
    """Returns the image data lists as bytes"""
    imgdict = make_image_data_list(image_filenames)
    return json.dumps({"requests": imgdict }).encode()


def request_ocr(api_key, image_filenames):
    response = requests.post(ENDPOINT_URL,
                             data=make_image_data(image_filenames),
                             params={'key': api_key},
                             headers={'Content-Type': 'application/json'})
    return response


def gvision_ocr_perc(image_source):
    vertlist=[]
    image_filenames = [image_source]
    import json

    response = request_ocr(api_key, image_filenames)
    if response.status_code != 200 or response.json().get('error'):
        print(response.text)
    else:
        for idx, resp in enumerate(response.json()['responses']):
            # save to JSON file
            #imgname = image_filenames[idx]
            #jpath = join(basename(imgname) + '.json')
            parsed= json.dumps(resp, indent=2)
            data = json.loads(parsed)
            
            
            #print "formatted json"
            #print (data)

            for i in range (0, len(data['textAnnotations'])):
                r=[]
                perc = data['textAnnotations'][i]['description']
                
                if bool(re.match('[\d/-]+%', perc)) or bool(re.match('[\d/-]', perc)) : 
                    bounding_vert = data['textAnnotations'][i]['boundingPoly']
                    #dict_vert_perc[bounding_vert]=perc
    
                    dictofverts= bounding_vert['vertices']
                    
                    for l in dictofverts:
                        point = []
                        for coord,value in l.iteritems():
                            r.append(value)
                        
                    y1,x1,y2,x2,y3,x3,y4,x4 = r
                    dict_vert_perc[(x1,y1,x2,y2,x3,y3,x4,y4)] = perc
#                     import cv2
#                     img = cv2.imread("try.jpg")
#                     print img
#                     cv2.rectangle(img,(x1-2,y1-2),(x3+2,y3+2),(21,213,55),3)
#                     cv2.imwrite('final_perc_boxes.png',img)
                    print perc
                
    #print dict_vert_perc             
    return dict_vert_perc

def gvision_ocr_text(image_filenames):
    #api_key, *image_filenames = argv[1:]
    #python cloudvisreq.py api_key image1.jpg image2.png""")
    
    #image_filenames = ["cropped2.png"]
    response = request_ocr(api_key, image_filenames)
    if response.status_code != 200 or response.json().get('error'):
        print(response.text)
    else:
        try:
            for idx, resp in enumerate(response.json()['responses']):
                # save to JSON file
                imgname = image_filenames[idx]
                #jpath = join(basename(imgname) + '.json')
                
                datatxt = json.dumps(resp, indent=2)
                # print the plaintext to screen for convenience
                
                t = resp['textAnnotations'][0]
                #print("    Bounding Polygon:")
                bounding_vert= t['boundingPoly']
                #print(bounding_vert)
                #print("    Text:")
                text = t['description']
                #print(text)
        except:
            #print "* EMPTY BOX *"
            text = u''
            
    
    return text


# In[ ]:




# In[ ]:



