from flask import Flask,render_template,request
from keras.models import load_model
import pickle as pkl
import numpy as np
import cv2
model = load_model('BrainTumorModel.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/output', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		photo = request.files['photo']
		photo.save('pic.jpg')
		photo.save('static/pic.jpg')
		photo = np.array([cv2.resize(cv2.imread('pic.jpg'), dsize=(240,240), interpolation=cv2.INTER_CUBIC) / 255.])
		pred = np.round(model.predict(photo)[0][0])
		if pred > 0.8:
			output = 'YES'
		else:
			output = 'NO'
	else:
		output = 'incorrect method'
	return render_template('output.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)