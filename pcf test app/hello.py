from flask import Flask
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


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=int(port))



from flask import Flask

app = Flask(__name__)



