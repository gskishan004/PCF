from flask import Flask
from flask import request
from PIL import Image
import os
import pprint 
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)


port = os.getenv('VCAP_APP_PORT', '5000')

@app.route('/')
def welcome():
	return 'Welcome to Orgchart Reader \nSupported API calls \n1)/instance : Instance Details \n2)/run/<string:imgPath> : To process the image'

@app.route('/instance')
def instance():
	return 'Instance Details \n\n' + pprint.pformat(str(os.environ))

@app.route('/run/<string:imgPath>', methods=['GET'])
def run(imgPath):
    return 'imagePath - ' + imgPath

@app.route('/upload', methods=['POST'])
def upload():
	imagefile = Image.open(request.files['file'])
	imagefile.save("upload.png")
	return 'Success ! '
	
@app.route('/get_image')
def get_image():
    if request.args.get('type') == '1':
       filename = 'ok.gif'
    else:
       filename = 'error.gif'
    return send_file(filename, mimetype='image/gif')


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=int(port))



from flask import Flask

app = Flask(__name__)



