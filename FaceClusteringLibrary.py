from sklearn.cluster import DBSCAN
from imutils import build_montages, paths
import numpy as np
import os
import pickle
import cv2
import shutil
import time
import dlib
from pyPiper import Node, Pipeline
from tqdm import tqdm
import face_recognition

''' Common utilities '''
'''
Credits: AndyP at StackOverflow
The ResizeUtils provides resizing function to keep the aspect ratio intact
'''
class ResizeUtils:
    # Given a target height, adjust the image by calculating the width and resize
    def rescale_by_height(self, image, target_height, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_height` (preserving aspect ratio)."""
        w = int(round(target_height * image.shape[1] / image.shape[0]))
        return cv2.resize(image, (w, target_height), interpolation=method)

    # Given a target width, adjust the image by calculating the height and resize
    def rescale_by_width(self, image, target_width, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_width` (preserving aspect ratio)."""
        h = int(round(target_width * image.shape[0] / image.shape[1]))
        return cv2.resize(image, (target_width, h), interpolation=method)



''' Frames extractor from video footage '''
'''
The FramesGenerator extracts image frames from the given video file
The image frames are resized for dlib processing
'''
class FramesGenerator:
    def __init__(self, VideoFootageSource):
        self.VideoFootageSource = VideoFootageSource

    # Resize the given input to fit in a specified 
    # size for face embeddings extraction
    def AutoResize(self, frame):
        resizeUtils = ResizeUtils()

        height, width, _ = frame.shape

        if height > 500:
            frame = resizeUtils.rescale_by_height(frame, 500)
            self.AutoResize(frame)
        
        if width > 700:
            frame = resizeUtils.rescale_by_width(frame, 700)
            self.AutoResize(frame)
        
        return frame

    # Extract 1 frame from each second from video footage 
    # and save the frames to a specific folder
    def GenerateFrames(self, OutputDirectoryName):
        cap = cv2.VideoCapture(self.VideoFootageSource)
        _, frame = cap.read()

        fps = cap.get(cv2.CAP_PROP_FPS)
        TotalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        print("[INFO] Total Frames ", TotalFrames, " @ ", fps, " fps")
        print("[INFO] Calculating number of frames per second")

        CurrentDirectory = os.path.curdir
        OutputDirectoryPath = os.path.join(CurrentDirectory, OutputDirectoryName)

        if os.path.exists(OutputDirectoryPath):
            shutil.rmtree(OutputDirectoryPath)
            time.sleep(0.5)
        os.mkdir(OutputDirectoryPath)

        CurrentFrame = 1
        fpsCounter = 0
        FrameWrittenCount = 1
        while CurrentFrame < TotalFrames:
            _, frame = cap.read()
            if (frame is None):
                continue
            
            if fpsCounter > fps:
                fpsCounter = 0

                frame = self.AutoResize(frame)

                filename = "frame_" + str(FrameWrittenCount) + ".jpg"
                cv2.imwrite(os.path.join(OutputDirectoryPath, filename), frame)

                FrameWrittenCount += 1
            
            fpsCounter += 1
            CurrentFrame += 1

        print('[INFO] Frames extracted')



''' Face clustering multithreaded pipeline '''
'''
Following are nodes for pipeline constructions. It will create and asynchronously
execute threads for reading images, extracting facial features and storing 
them independently in different threads
'''
# Keep emitting the filenames into the pipeline for processing
class FramesProvider(Node):
    def setup(self, sourcePath):
        self.sourcePath = sourcePath
        self.filesList = []
        for item in os.listdir(self.sourcePath):
            _, fileExt = os.path.splitext(item)
            if fileExt == '.jpg':
                self.filesList.append(os.path.join(item))
        self.TotalFilesCount = self.size = len(self.filesList)
        self.ProcessedFilesCount = self.pos = 0

    # Emit each filename in the pipeline for parallel processing
    def run(self, data):
        if self.ProcessedFilesCount < self.TotalFilesCount:
            self.emit({'id': self.ProcessedFilesCount, 
                'imagePath': os.path.join(self.sourcePath, 
                                self.filesList[self.ProcessedFilesCount])})
            self.ProcessedFilesCount += 1
            
            self.pos = self.ProcessedFilesCount
        else:
            self.close()

# Encode the face embedding, reference path and location 
# and emit to pipeline
class FaceEncoder(Node):
    def setup(self, detection_method = 'cnn'):
        self.detection_method = detection_method
        # detection_method can be cnn or hog

    def run(self, data):
        id = data['id']
        imagePath = data['imagePath']
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=self.detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc} 
                for (box, enc) in zip(boxes, encodings)]

        self.emit({'id': id, 'encodings': d})

# Recieve the face embeddings for clustering and 
# id for naming the distinct filename
class DatastoreManager(Node):
    def setup(self, encodingsOutputPath):
        self.encodingsOutputPath = encodingsOutputPath
    def run(self, data):
        encodings = data['encodings']
        id = data['id']
        with open(os.path.join(self.encodingsOutputPath, 
                            'encodings_' + str(id) + '.pickle'), 'wb') as f:
            f.write(pickle.dumps(encodings))

# Inherit class tqdm for visualization of progress
class TqdmUpdate(tqdm):
    # This function will be passed as progress callback function
    # Setting the predefined variables for auto-updates in visualization
    def update(self, done, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.n = done
        super().refresh()



''' Pickle files merging '''
'''
PicklesListCollator takes multiple pickle files as input and merge them together
It is made specifically to support our use-case of merging distinct pickle
files into one
'''
class PicklesListCollator:
    def __init__(self, picklesInputDirectory):
        self.picklesInputDirectory = picklesInputDirectory
    
    # Here we will list down all the pickles files generated from 
    # multiple threads, read the list of results
    # append them to a common list and create another pickle
    # with combined list as content
    def GeneratePickle(self, outputFilepath):
        datastore = []

        ListOfPickleFiles = []
        for item in os.listdir(self.picklesInputDirectory):
            _, fileExt = os.path.splitext(item)
            if fileExt == '.pickle':
                ListOfPickleFiles.append(os.path.join(self.picklesInputDirectory, item))

        for picklePath in ListOfPickleFiles:
            with open(picklePath, "rb") as f:
                data = pickle.loads(f.read())
                datastore.extend(data)

        with open(outputFilepath, 'wb') as f:
            f.write(pickle.dumps(datastore))



''' Face clustering functionality '''
class FaceClusterUtility:
	def __init__(self, EncodingFilePath):
		self.EncodingFilePath = EncodingFilePath
	
    # Credits: Arian's pyimagesearch for the clustering code
    # Here we are using the sklearn.DBSCAN functioanlity
    # cluster all the facial embeddings to get clusters 
    # representing distinct people
	def Cluster(self):
		InputEncodingFile = self.EncodingFilePath
		if not (os.path.isfile(InputEncodingFile) and os.access(InputEncodingFile, os.R_OK)):
			print('The input encoding file, ' + 
                    str(InputEncodingFile) + ' does not exists or unreadable')
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

    # Credits: Adrian's pyimagesearch for reference codes
    # Here we are creating montages for first 25 faces for each 
    # distinct face. 
    # We will also generate images for all the distinct faces by using the 
    # labels from clusters and image url from the encodings pickle 
    # file.
    
    # The face bounding box is increased a little more for 
    # training purposes and we also created the exact 
    # annotation for each face image (similar to darknet YOLO)
    # to easily adapt the annotation for future use in supervised training
	def GenerateImages(self, labels, OutputFolderName = "ClusteredFaces", 
                                            MontageOutputFolder = "Montage"):
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

			# initialize the list of faces to include in the montage
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

				resizeUtils = ResizeUtils()
				portrait = resizeUtils.rescale_by_width(portrait, 400)

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
				f.write(str(labelID) + ' ' + 
                        str(left) + ' ' + str(top) + ' ' + 
                        str(right) + ' ' + str(bottom) + "\n")
				f.close()


				counter += 1

			montage = build_montages(portraits, (96, 120), (5, 5))[0]
			
			MontageFilenamePath = os.path.join(MontageFolderPath, "Face_" + str(labelID) + ".jpg")
			cv2.imwrite(MontageFilenamePath, montage)

