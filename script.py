import numpy as np
import sys
import dlib
import cv2
import openface
import importlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from openface_model import create_model


# Model von Sefik Ilkin Serengil https://drive.google.com/file/d/1LSe1YCV1x-BfNnfb7DFZTNpv_Q9jITxn/view
model_open_face = create_model()
model_open_face.load_weights('open_face.h5')


model_emotion = Sequential()

model_emotion.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model_emotion.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_emotion.add(MaxPooling2D(pool_size=(2, 2)))
model_emotion.add(Dropout(0.25))
model_emotion.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_emotion.add(MaxPooling2D(pool_size=(2, 2)))
model_emotion.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_emotion.add(MaxPooling2D(pool_size=(2, 2)))
model_emotion.add(Dropout(0.25))
model_emotion.add(Flatten())
model_emotion.add(Dense(1024, activation='relu'))
model_emotion.add(Dropout(0.5))
model_emotion.add(Dense(7, activation='softmax'))

# Load Emotion module
model_emotion.load_weights('src/model.h5')

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"


# Take the image file name from the command line
# file_name = sys.argv[1]

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)



# load video
input_movie = cv2.VideoCapture("unbekannt/sketch.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))


known_faces = []
known_faces_new = []
last_detected_faces = []
known_face_names = [
"Unbekannt 0",
]
emotion_array = []


# Initialize variables
frame_number = 0
count = 0
face_rec_var = 50
emotion_frame_count = 30
font = cv2.FONT_HERSHEY_DUPLEX

codec = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(input_movie.get(cv2.CAP_PROP_FPS))
frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_movie = cv2.VideoWriter("output.mp4", codec, fps, (frame_width,frame_height))

def findEuclideanDistance(source_representation, test_representation):
	euclidean_distance = source_representation - test_representation
	euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
	euclidean_distance = np.sqrt(euclidean_distance)
	#euclidean_distance = l2_normalize(euclidean_distance )
	return euclidean_distance

while True:
	# Find haar cascade to draw bounding box around face
	ret, frame = input_movie.read()
	frame_number += 1
	if not ret:
		break
#	if frame_number > 4800:
#		break
	if frame_number > 0:

		#facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

		# Run the HOG face detector on the image data.
		# The result will be the bounding boxes of the faces in our image.
		detected_faces = face_detector(frame, 1)
		

		#print("I found {} faces in the file {}".format(len(detected_faces), file_name))

		# Open a window on the desktop showing the image
		# win.set_image(frame)

	# Loop through each face we found in the image
		for i, face_rect in enumerate(detected_faces):

			# Detected faces are returned as an object with the coordinates 
			# of the top, left, right and bottom edges
			print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
			cv2.rectangle(frame, (face_rect.left()-50, face_rect.top()-50), (face_rect.right()+50, face_rect.bottom()+50), (255, 0, 0), 2)
		
			# Draw a box around each face we found
			#win.add_overlay(face_rect)
			#cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			# Get the the face's pose
			# pose_landmarks = face_pose_predictor(frame, face_rect)

			# Draw the face landmarks on the screen.
			# win.add_overlay(pose_landmarks)

			# Use openface to calculate and perform the face alignment
			alignedFace = face_aligner.align(534, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

			# Save the aligned image to a file
			cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)
			#roi_gray = alignedFace
			# roi_gray = gray[face_rect.top():face_rect.top() + face_rect.height(), face_rect.left():face_rect.left() + face_rect.right()]	
			roi_gray = face_aligner.align(534, gray, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
			#cropped_img = np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1)
			cropped_img = np.expand_dims(cv2.resize(alignedFace, (96, 96)), 0)
			cropped_img2 = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)


			prediction = model_open_face.predict(cropped_img)

			name_index = -1
			name_index_save = 0
			face_recognized = False
			old_result = 1.00
			same_faces = False
			known_faces = known_faces_new

			for known_face in known_faces:
				name_index += 1

				result = findEuclideanDistance(prediction, known_face)

				if result <= 0.15 and old_result > result:
					face_recognized = True
					print("Gesicht {} vergleichen erkannt: {}".format(name_index, result))
					#break
					if name_index < len(last_detected_faces):
						last_detected_face = last_detected_faces[name_index]

						if last_detected_face.left()-face_rec_var <= face_rect.left() and last_detected_face.right()+face_rec_var >= face_rect.right() and last_detected_face.bottom()+face_rec_var >= face_rect.bottom() and last_detected_face.top()-face_rec_var <= face_rect.top():
							print("Selber Platz und vergleichen erkannt: {}".format( result))
												
					old_result = result
					name_index_save = name_index
					#else:
				else:
					if name_index < len(last_detected_faces):
						last_detected_face = last_detected_faces[name_index]

						if last_detected_face.left()-face_rec_var <= face_rect.left() and last_detected_face.right()+face_rec_var >= face_rect.right() and last_detected_face.bottom()+face_rec_var >= face_rect.bottom() and last_detected_face.top()-face_rec_var <= face_rect.top():
							print("Selber Platz aber vergleichen nicht erkannt: {}".format( result))	
							if result <= 0.2:
								face_recognized = True		
								name_index_save = name_index
								old_result = result
			if face_recognized == False:
				print("Neues Gesicht")
				known_faces.append(prediction)
				name_index_save = len(known_faces) - 1
			if face_recognized == True:
				known_faces[name_index_save] = prediction

			if len(known_face_names) <= name_index_save:
				known_face_names.append("Unbekannt "+str(name_index_save))		
			cv2.putText(frame, known_face_names[name_index_save], (face_rect.left()-90, face_rect.bottom()+80), font, 0.8, (255, 255, 255), 1)
			print("Face Name: {}".format(known_face_names[name_index_save])) 

			prediction = model_emotion.predict(cropped_img2)
			maxindex = int(np.argmax(prediction))
			
			emotion_max_index = len(known_face_names)*emotion_frame_count
			emotion_index = name_index_save*emotion_frame_count-(frame_number%emotion_frame_count)+emotion_frame_count
			while len(emotion_array) <= emotion_max_index:
				emotion_array.append("Emotion Unbekannt")

			emotion_array[emotion_index] = emotion_dict[maxindex]
			
			emotion_array_fop = []
			i = emotion_frame_count
			while i > 0:
				emotion_array_fop.append(emotion_array[name_index_save*emotion_frame_count-(i)+emotion_frame_count])
				i -= 1
	
			liste = []
			for i in range(len(emotion_array_fop)):
  				liste.append(emotion_array_fop.count(emotion_array_fop[i]))
 
			maximum = max(liste)
 
			for i in range(len(liste)):
				if emotion_array_fop.count(emotion_array_fop[i]) == maximum:
					print("Das haeufigste Element lautet:", emotion_array_fop[i])
					cv2.putText(frame, emotion_array_fop[i], (face_rect.left()-50, face_rect.bottom()+120), font, 0.8, (255, 255, 255), 1)
					break

			#print("Emotion Index: {}".format(emotion_index))


		last_detected_faces = detected_faces
		# Schreiben des fertigen Bildes in das Ausgabe-Video
		output_movie.write(frame)
	
	print("Writing frame {} / {}".format(frame_number, length))
	cv2.imwrite("frames/frame%d.jpg" % count, frame)
	count +=1

# Fertig!
output_movie.release()
cv2.destroyAllWindows()      
