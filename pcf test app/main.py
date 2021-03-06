from flask import Flask
from flask import request
from flask import send_file
from flask.json import jsonify
import ntpath
from PIL import Image
import os
import pprint 
import logging
from OrgStructReader import main

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)


port = os.getenv('VCAP_APP_PORT', '5000')

@app.route('/')
def welcome():
	return 'Welcome to Orgchart Reader\nSupported API calls\n/instance: Instance Details\n/upload/<string:imageName>: Upload image\n/imageFiles(GET): List image files\n/imageFiles(DELETE): Delete image files\n/process_image/<string:imgName>/<string:clientName>: Run the image files'

# get details of the instance
@app.route('/instance')
def instance():
	return 'Instance Details \n\n' + pprint.pformat(str(os.environ))

# upload image file followed by the name, name should include .png / .jpg
@app.route('/upload/<string:imageName>', methods=['POST'])
def upload(imageName):
	imagefile = Image.open(request.files['file'])
	imagefile.save(imageName)
	return 'Success ! '

# get list of images files
@app.route('/imageFiles')
def get_imageFiles():
	cwd = os.getcwd()
	imageFiles = [x for x in os.listdir(cwd) if x.endswith(".png") or x.endswith(".jpg")]
	if len(imageFiles) == 0:
		return "No imageFiles"
	
	return jsonify({'imageFiles': imageFiles})

# delete all the image files in the dir
@app.route('/imageFiles', methods=['DELETE'])
def delete():
	cwd = os.getcwd()
	imageFiles = [x for x in os.listdir(cwd) if x.endswith(".png") or x.endswith(".jpg")]
	for file in imageFiles:
		os.remove(file)
	return "All Image files deleted"

	
# process the image, NOTE : img name should also contain .jpg / .png extension	
@app.route('/process_image/<string:imgName>/<string:clientName>', methods=['GET'])
def process_image(imgName,clientName):
	
	if main(imgName, clientName) == 0:
		return "Signature didn't match"

	try:
		file_name_witohut_ext = os.path.splitext(ntpath.basename(imgName))[0]
		file = 'output/'+ file_name_witohut_ext + '/out'
		return send_file(file, attachment_filename='out')
	except Exception as e:
		return str(e)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=int(port))



from flask import Flask

app = Flask(__name__)
