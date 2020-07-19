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

# Face Recognition model von Sefik Ilkin Serengil https://drive.google.com/file/d/1LSe1YCV1x-BfNnfb7DFZTNpv_Q9jITxn/view
model_open_face = create_model()
model_open_face.load_weights('src/open_face.h5')

# Emotion Module
# Load Emotion model from Atul Balaji
# https://github.com/atulapra/Emotion-detection
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
model_emotion.load_weights('src/model.h5')

# The required pre-trained face detection model from dlib:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"

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


# Initialize variables and arrays
known_faces = []
known_faces_new = []
last_detected_faces = []
known_face_names = [
"Unbekannt 0",
]
emotion_array = []
frame_number = 0
count = 0
face_rec_var = 50
emotion_frame_count = 30
font = cv2.FONT_HERSHEY_DUPLEX

# Initialize variables for video In- and Out-Put
codec = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(input_movie.get(cv2.CAP_PROP_FPS))
frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_movie = cv2.VideoWriter("output.mp4", codec, fps, (frame_width,frame_height))


# Calculate the EuclideanDistance of two arrays
def findEuclideanDistance(source_representation, test_representation):
	euclidean_distance = source_representation - test_representation
	euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
	euclidean_distance = np.sqrt(euclidean_distance)
	return euclidean_distance


while True:
	# Read one frame of the video
	ret, frame = input_movie.read()
	frame_number += 1
	if not ret:
		break
	if frame_number > 0:

		# Change the color of the frame to gray
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Run the HOG face detector on the frame data to find all faces in the frame.
		# Detected faces are returned as an object with the coordinates
		detected_faces = face_detector(frame, 1)
		


		# Loop through each face we found in the frame
		for i, face_rect in enumerate(detected_faces):

			# Initialize variables for the comparison of the predictions
			name_index = -1
			name_index_save = 0
			face_recognized = False
			old_result = 1.00
			known_faces = known_faces_new

			# Draw a box around each face we found
			print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
			cv2.rectangle(frame, (face_rect.left()-50, face_rect.top()-50), (face_rect.right()+50, face_rect.bottom()+50), (255, 0, 0), 2)
		

			# Use openface to calculate and perform the face alignment
			# One for the face-prediction with the rgb frame and one for the emotion-prediction with the gray frame
			alignedFace = face_aligner.align(534, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
			roi_gray = face_aligner.align(534, gray, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

			# Adjusting the face-image to the model to be compared
			cropped_img = np.expand_dims(cv2.resize(alignedFace, (96, 96)), 0)
			cropped_img2 = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

			# Find out the 128 Measurements of the face 
			prediction = model_open_face.predict(cropped_img)


			# Search in the array of known faces if the face is recognized in a former frame
			for known_face in known_faces:
				name_index += 1

				# Calcuates the EuclideanDistance of the face with a face already recognized
				# And checks if a face has been in the smae place in a previous frame
				result = findEuclideanDistance(prediction, known_face)

				if result <= 0.15 and old_result > result:
					face_recognized = True
					print("Gesicht {} vergleichen erkannt: {}".format(name_index, result))

					if name_index < len(last_detected_faces):
						last_detected_face = last_detected_faces[name_index]

						if last_detected_face.left()-face_rec_var <= face_rect.left() and last_detected_face.right()+face_rec_var >= face_rect.right() and last_detected_face.bottom()+face_rec_var >= face_rect.bottom() and last_detected_face.top()-face_rec_var <= face_rect.top():
							print("Selber Platz und vergleichen erkannt: {}".format( result))
												
					old_result = result
					name_index_save = name_index

				else:

					if name_index < len(last_detected_faces):
						last_detected_face = last_detected_faces[name_index]

						if last_detected_face.left()-face_rec_var <= face_rect.left() and last_detected_face.right()+face_rec_var >= face_rect.right() and last_detected_face.bottom()+face_rec_var >= face_rect.bottom() and last_detected_face.top()-face_rec_var <= face_rect.top():
							print("Selber Platz aber vergleichen nicht erkannt: {}".format( result))	

							if result <= 0.20:
								face_recognized = True		
								name_index_save = name_index
								old_result = result	

			# If no face matches an already recognized face, it adds its values to the array
			if face_recognized == False:
				print("Neues Gesicht")
				known_faces.append(prediction)
				name_index_save = len(known_faces) - 1
			if face_recognized == True:
				known_faces[name_index_save] = prediction

			# Write the name of the person into the frame
			if len(known_face_names) <= name_index_save:
				known_face_names.append("Unbekannt "+str(name_index_save))		
			cv2.putText(frame, known_face_names[name_index_save], (face_rect.left()-90, face_rect.bottom()+80), font, 0.8, (255, 255, 255), 1)
			print("Face Name: {}".format(known_face_names[name_index_save])) 


			# Find out the 128 Measurements of the face 
			prediction = model_emotion.predict(cropped_img2)

			# Look which emotion is past this value
			maxindex = int(np.argmax(prediction))

			# Expands the array of all pre-recognized emotions
			# emotion_frame_count idicates how many emotions in this array belong to one person			
			emotion_max_index = len(known_face_names)*emotion_frame_count
			while len(emotion_array) <= emotion_max_index:
				emotion_array.append("Emotion Unbekannt")

			# Adds the emotion of a person recognized in this frame in the array
			emotion_index = name_index_save*emotion_frame_count-(frame_number%emotion_frame_count)+emotion_frame_count
			emotion_array[emotion_index] = emotion_dict[maxindex]

			# extract all emotions of a person from the array and stores them into the array where only emotions of the currently viewed person		
			emotion_array_fop = []
			i = emotion_frame_count
			while i > 0:
				emotion_array_fop.append(emotion_array[name_index_save*emotion_frame_count-(i)+emotion_frame_count])
				i -= 1
	
			# Now the emotion that occurs most often in the array is searched and will be written into the frame
			liste = []
			for i in range(len(emotion_array_fop)):
  				liste.append(emotion_array_fop.count(emotion_array_fop[i]))
 
			maximum = max(liste)
 
			for i in range(len(liste)):
				if emotion_array_fop.count(emotion_array_fop[i]) == maximum:
					print("Das haeufigste Element lautet:", emotion_array_fop[i])
					cv2.putText(frame, emotion_array_fop[i], (face_rect.left()-50, face_rect.bottom()+120), font, 0.8, (255, 255, 255), 1)
					break

		# all recognized faces are now saved into an array, so you can compare the faces in the next frame
		last_detected_faces = detected_faces
		# write the analysed frame in the output video
		output_movie.write(frame)

	# Write the analysed frame in the frame folder	
	print("Writing frame {} / {}".format(frame_number, length))
	cv2.imwrite("frames/frame%d.jpg" % count, frame)
	count +=1

# Finish!
output_movie.release()
cv2.destroyAllWindows()      
