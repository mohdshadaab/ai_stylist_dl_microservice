import requests
from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import base64
from io import StringIO, BytesIO
import io
import glob
import torch
import json
import os
import cv2 
import webcolors
from PIL import Image
from unet_segmentation.prediction.display import display_prediction
from u2net_image import unet_image
CUDA_VISIBLE_DEVICES=""
device = torch.device("cpu")
model = torch.load('unet_iter_1300000.pt',map_location='cpu')


app = FastAPI()


#to handle requests between cross-origin resources

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class Item(BaseModel):
	user_id: str
	base64: str
	color: list


# Take in base64 string and return cv image
def stringToImage(base64_string ):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour, spec='css3')
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return closest_name


#Sample code for using get_colour_name function
#requested_colour = (119, 172, 152)
#actual_name, closest_name = get_colour_name(requested_colour)
#print("Actual name: ", actual_name)
#print("Closest Name:", closest_name)
def hex_to_color(hex:str):
	color=[]
	for single_color in hex:
		color.append(get_colour_name(webcolors.hex_to_rgb(single_color)))
	return color


def unet(image:Image.Image, user_id:str):
	#os.system('mkdir u2net_results')
	
	img_name=image.filename.split('/')[::-1][0]
	#tmp_file='test_data/images/temp/'+ img_name
	#dst='/home/azureuser/background_removal_DL/test_data/images/input/'
	#shutil.copy2(tmp_file, dst)
#time.sleep(2)
#os.system("cp tmp_file dst")

	unet_image(user_id)
	file_name,file_extension=img_name.split(".")
#u2netresult
	u2netresult=cv2.imread('test_data/images/u2netp_results/'+file_name+'.png')
#orginalimage (CHANGE FILE EXTENSION HERE - BY DEFAULT: *.jpg)
	original=cv2.imread('test_data/images/input/'+user_id+'/'+img_name)
#subimage
	subimage=cv2.subtract(u2netresult,original)
#print(subimage)
	cv2.imwrite('test_data/images/output/'+user_id+'/'+file_name+'.png',subimage)

#subimage
	subimage=Image.open('test_data/images/output/'+user_id+'/'+file_name+'.png')
#originalimage
	original=Image.open('test_data/images/input/'+user_id+'/'+img_name)
	subimage=subimage.convert("RGB")
	original=original.convert("RGB")
	subdata=subimage.getdata()
	ogdata=original.getdata()
	newdata=[]
	for i in range(subdata.size[0]*subdata.size[1]):
		if subdata[i][0]==0 and subdata[i][1]==0 and subdata[i][2]==0:
			newdata.append((255,255,255,0))
		else:
			newdata.append(ogdata[i])
	subimage.putdata(newdata)
    #subimage.save('/home/azureuser/fyp_file/unet_output/'+file_name+'.png',"PNG")
	subimage.save('test_data/images/output/'+user_id+'/'+file_name+'.png')
	del subimage, ogdata,newdata,original,subdata,u2netresult
#%cd './test_data/images'|
	os.system('rm -rf test_data/images/u2netp_results')
	os.system('rm -rf test_data/images/input/'+user_id)
	#model = torch.load('/home/azureuser/fyp_file/unet_iter_1300000.pt',map_location='cpu')
	label_list=display_prediction(model, 'test_data/images/output/'+user_id+'/'+file_name+'.png')
	return label_list


@app.get("/")
def read_root():
    return {"Works": "Fine"}

@app.post("/dl_service/")
async def read_item(item: Item):
	user_id=item.user_id
	if user_id:
		url = 'http://aiwardrobe.southeastasia.cloudapp.azure.com:8000/account/get_user'

		payload=f'user_id={user_id}'
		headers = {
  			'Content-Type': 'application/x-www-form-urlencoded'
		}

		response = requests.request("POST", url, headers=headers, data=payload)
		data=response.json()
		print(item.color)
		if response.status_code == 200:
			os.system(f'mkdir test_data/images/input')
			os.system(f'mkdir test_data/images/input/{item.user_id}')
			os.system(f'mkdir test_data/images/output/{item.user_id}')
			img=stringToImage(item.base64)
			print(glob.glob(f"test_data/images/input/{item.user_id}/*"))
			img.save(f'test_data/images/input/{item.user_id}/{item.user_id}.jpeg')
			del img
			img=Image.open(f'test_data/images/input/{item.user_id}/{item.user_id}.jpeg')
			res= unet(img, item.user_id)
			url = "http://aiwardrobe.southeastasia.cloudapp.azure.com:8000/account/upload_image"
				
			payload={
						'user_id': user_id,
						'category_list': json.dumps(res),
						'color': json.dumps(hex_to_color(item.color))
					}
				
			headers = {
  				#'Content-Type': 'application/x-www-form-urlencoded'
			}
			files={'image': open('test_data/images/output/'+user_id+'/'+user_id+'.png','rb')}
			response = requests.request("POST", url, data=payload, files=files)
			print(payload)
			os.system(f'rm -rf test_data/images/output/{item.user_id}')
				#print(response.text)
		else:
			raise HTTPException(status_code=response.status_code, detail="Something went wrong")
	return payload

