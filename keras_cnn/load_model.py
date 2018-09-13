import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

# Loading  model
json_file = open("saved_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("saved_model.h5")

# Comliling model
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

img_path = ''
while img_path != '.exit':
	# Img filepath select
	img_path = raw_input('Enter the file path, or ".exit" for exit: ')
	if img_path == '.exit':
		break
		
	img = image.load_img(img_path, target_size=(28, 28), grayscale=True)

	# Img to np array
	x = image.img_to_array(img)

	x /= 255
	x = np.expand_dims(x, axis=0)
	# Make pred
	prediction = loaded_model.predict(x)

	print("Digit {}({:.5%})".format(np.argmax(prediction), np.amax(prediction[0])))