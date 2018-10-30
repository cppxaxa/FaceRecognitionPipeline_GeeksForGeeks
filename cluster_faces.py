from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import os
import pickle
import cv2
import shutil
import time

class FaceClusterUtility:
	def __init__(self, EncodingFilePath):
		self.EncodingFilePath = EncodingFilePath
	
	def Cluster(self):
		InputEncodingFile = self.EncodingFilePath
		if not (os.path.isfile(InputEncodingFile) and os.access(InputEncodingFile, os.R_OK)):
			print('The input encoding file, ' + str(InputEncodingFile) + ' does not exists or unreadable')
			exit()

		NumberOfParallelJobs = -1

		# load the serialized face encodings + bounding box locations from
		# disk, then extract the set of encodings to so we can cluster on
		# them
		print("[INFO] Loading encodings")
		data = pickle.loads(open(InputEncodingFile, "rb").read())
		data = np.array(data)

		encodings = [d["encoding"] for d in data]

		# cluster the embeddings
		print("[INFO] Clustering")
		clt = DBSCAN(eps=0.5, metric="euclidean", n_jobs=NumberOfParallelJobs)
		clt.fit(encodings)

		# determine the total number of unique faces found in the dataset
		labelIDs = np.unique(clt.labels_)
		numUniqueFaces = len(np.where(labelIDs > -1)[0])
		print("[INFO] # unique faces: {}".format(numUniqueFaces))

		return clt.labels_

class FaceImageGenerator:
	def __init__(self, EncodingFilePath):
		self.EncodingFilePath = EncodingFilePath

	def GenerateImages(self, labels, OutputFolderName = "ClusteredFaces", MontageOutputFolder = "Montage"):
		output_directory = os.getcwd()

		OutputFolder = os.path.join(output_directory, OutputFolderName)
		if not os.path.exists(OutputFolder):
			os.makedirs(OutputFolder)
		else:
			shutil.rmtree(OutputFolder)
			time.sleep(0.5)
			os.makedirs(OutputFolder)

		MontageFolderPath = os.path.join(OutputFolder, MontageOutputFolder)
		os.makedirs(MontageFolderPath)

		data = pickle.loads(open(self.EncodingFilePath, "rb").read())
		data = np.array(data)

		labelIDs = np.unique(labels)
		# loop over the unique face integers
		for labelID in labelIDs:
			# find all indexes into the `data` array that belong to the
			# current label ID, then randomly sample a maximum of 25 indexes
			# from the set
			
			print("[INFO] faces for face ID: {}".format(labelID))

			FaceFolder = os.path.join(OutputFolder, "Face_" + str(labelID))
			os.makedirs(FaceFolder)

			idxs = np.where(labels == labelID)[0]
			idxs = np.random.choice(idxs, size=min(25, len(idxs)),
				replace=False)

			# initialize the list of faces to include in the montage
			# faces = []
			portraits = []

			# loop over the sampled indexes
			counter = 1
			for i in idxs:
				# load the input image and extract the face ROI
				image = cv2.imread(data[i]["imagePath"])
				(o_top, o_right, o_bottom, o_left) = data[i]["loc"]

				height, width, channel = image.shape

				widthMargin = 100
				heightMargin = 150

				top = o_top - heightMargin
				if top < 0:
					top = 0
				
				bottom = o_bottom + heightMargin
				if bottom > height:
					bottom = height
				
				left = o_left - widthMargin
				if left < 0:
					left = 0
				
				right = o_right + widthMargin
				if right > width:
					right = width

				portrait = image[top:bottom, left:right]

				if len(portraits) < 25:
					portraits.append(portrait)

				portrait = rescale_by_width(portrait, 400)

				FaceFilename = "face_" + str(counter) + ".jpg"

				FaceImagePath = os.path.join(FaceFolder, FaceFilename)
				cv2.imwrite(FaceImagePath, portrait)





				widthMargin = 20
				heightMargin = 20

				top = o_top - heightMargin
				if top < 0:
					top = 0
				
				bottom = o_bottom + heightMargin
				if bottom > height:
					bottom = height
				
				left = o_left - widthMargin
				if left < 0:
					left = 0
				
				right = o_right + widthMargin
				if right > width:
					right = width

				AnnotationFilename = "face_" + str(counter) + ".txt"
				AnnotationFilePath = os.path.join(FaceFolder, AnnotationFilename)
				
				f = open(AnnotationFilePath, 'w')
				f.write(str(labelID) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + "\n")
				f.close()


				counter += 1

			montage = build_montages(portraits, (96, 120), (5, 5))[0]
			
			MontageFilenamePath = os.path.join(MontageFolderPath, "Face_" + str(labelID) + ".jpg")
			cv2.imwrite(MontageFilenamePath, montage)







if __name__ == "__main__":
	EncodingPickleFilePath = "encodings.pickle"

	faceClusterUtility = FaceClusterUtility(EncodingPickleFilePath)
	faceImageGenerator = FaceImageGenerator(EncodingPickleFilePath)

	labelIDs = faceClusterUtility.Cluster()
	faceImageGenerator.GenerateImages(labelIDs, "ClusteredFaces", "Montage")
