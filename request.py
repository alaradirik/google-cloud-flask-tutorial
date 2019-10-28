import requests
import json
import cv2


url = "http://<external IP address>:5000/predict"
headers = {"content-type": "image/jpg"}

# encode image
image = cv2.imread('images/baggage_claim.jpg')
_, img_encoded = cv2.imencode(".jpg", image)

# send HTTP request to the server
response = requests.post(url, data=img_encoded.tostring(), headers=headers)
predictions = response.json()

# annotate the image
for pred in predictions:
	# print prediction
	print(pred)

	# extract the bounding box coordinates
	(x, y) = (pred["boxes"][0], pred["boxes"][1])
	(w, h) = (pred["boxes"][2], pred["boxes"][3])

	# draw a bounding box rectangle and label on the image
	cv2.rectangle(image, (x, y), (x + w, y + h), pred["color"], 2)
	text = "{}: {:.4f}".format(pred["label"], pred["confidence"])
	cv2.putText(
		image, 
		text, 
		(x, y - 5), 
		cv2.FONT_HERSHEY_SIMPLEX,
	    0.5, 
	    pred["color"], 
	    2
	)

# save annotated image
cv2.imwrite("annotated_image.jpg", image)