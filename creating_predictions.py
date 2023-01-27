"""
Author: Fabio Pöschko
Matr.Nr.: K11905017
Exercise 5
"""

#creating the predictions on for the server
import pickle
import os
import torch

from PIL import Image
import numpy as np

def array_to_img(image):
    img = Image.fromarray(image,)
    img.show()

#open the testfile
with open(os.path.join('data','testset.pkl'), 'rb') as pickle_file:
    arrays = pickle.load(pickle_file)

# import the model and looking at it
model = torch.load(os.path.join("results", "my_model_A.pt"),map_location=torch.device('cpu'))
print(model)


#creating the predictions
predictions_img = []
predictions_server = []
for i in range(len(arrays['input_arrays'])):
    inp = arrays['input_arrays'][i].reshape(1,1,90,90)
    out = model(torch.tensor(inp,dtype = torch.float32)/255)
    predictions_img.append(out*255)
    mask = arrays['known_arrays'][i]
    predictions_server.append(np.array((out[0][0]*255)[mask==0].detach().numpy(),dtype=np.uint8))


#looking at an image, to eliminate errors
for r in range(10):
    array_to_img(predictions_img[r][0][0].detach().numpy())


#creating the submissions file
with open(os.path.join('data','cnn_final_prediction.pkl'), 'wb') as pickle_file:
    pickle.dump(predictions_server, pickle_file)
print("saved")

#open the submissionsfile, to eliminate errors
with open(os.path.join('data','cnn_final_prediction.pkl'), 'rb') as pickle_file:
    convolution = pickle.load(pickle_file)
print(convolution[:5])