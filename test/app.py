from flask import Flask, request, jsonify
import mlflow.pyfunc
#import pandas as pd
import json

# Name of the apps module package
app = Flask(__name__)

# Load in the model at app startup
model = mlflow.pyfunc.load_model('./model')

# Load in our meta_data
f = open("./model/code/meta_data.txt", "r")
load_meta_data = json.loads(f.read())

# Meta data endpoint
@app.route('/', methods=['GET'])
def meta_data():

	return jsonify(load_meta_data)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
 req = request.get_json()
	
# Log the request
 print({'request': req})

# Format the request data in a DataFrame
# inf_df = pd.DataFrame(req['data'])

# Get model prediction - convert from np to list
# fakeimage = model.predict(inf_df).tolist()
 fakeimage = model.predict(0)

# Log the prediction
# print({'response': pred})
# img = outimage.numpy()
 print('\n\timg = ',type(fakeimage))
 print('\n\timg.shape = ',fakeimage.shape)

 imgnp = fakeimage.numpy()

 print('\n',type(imgnp))
 print('\n',imgnp.shape) #(64, 3, 64, 64)

 singleimg = imgnp[5,:,:,:]

# print('\nsingleimg = ',singleimg[:,0:64:5,0:64:10])

# img = Image.fromarray(img.astype("uint8"))
#	rawBytes = io.BytesIO()#
#	img.save(rawBytes, "JPEG")#
#	rawBytes.seek(0)
#	img_base64 = base64.b64encode(rawBytes.read())
#	return jsonify({'status':str(img_base64)})
 
# Return prediction as reponse
 return jsonify(['DONE : Ideally this should be an image'])

app.run(host='0.0.0.0', port=5000, debug=True)
