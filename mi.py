import cv2
import torch
import matplotlib.pyplot as plt

#midas download

'''
midas = torch.hub.load('intel.isl/MiDaS','MiDaS_small')
midas.to('cpu')   # we have to connect to gpu or cpu but we taking cpu 
midas.eval()

transforms = torch.hub.load('intel.isl/MiDaS', 'transforms')
transform = transforms.small_transform

'''

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

use_large_model = True

if use_large_model:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
else:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

device = "cpu"
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if use_large_model:
    transform = midas_transforms.default_transform
else:
    transform = midas_transforms.small_transform

    

#hook into opencv
cap =  cv2.VideoCapture(0)
while cap.isOpened():

    ret, frame = cap.read()

    # transform input for midas

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),  
            size = img.shape[:2],     #size giving int error
            mode= 'bicubic',
            align_corners=False
      ).squeeze()
        
        output = prediction.cpu().numpy()
        
        print(output)
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.imshow(output)
    cv2.imshow('CV2Frame', frame)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

plt.show()
