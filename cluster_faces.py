from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import os
import pickle
import cv2
import shutil
import time
import FaceClusteringLibrary


class FaceClusterUtility:
    def __init__(self, encoding_file_path):
        self.EncodingFilePath = encoding_file_path

    def cluster(self):
        input_encoding_file = self.EncodingFilePath
        if not (os.path.isfile(input_encoding_file) and os.access(input_encoding_file, os.R_OK)):
            print('The input encoding file, ' + str(input_encoding_file) + ' does not exists or unreadable')
            exit()

        number_of_parallel_jobs = -1

        # load the serialized face encodings + bounding box locations from
        # disk, then extract the set of encodings to so we can cluster on
        # them
        print("[INFO] Loading encodings")
        data = pickle.loads(open(input_encoding_file, "rb").read())
        data = np.array(data)

        encodings = [d["encoding"] for d in data]

        # cluster the embeddings
        print("[INFO] Clustering")
        clt = DBSCAN(eps=0.5, metric="euclidean", n_jobs=number_of_parallel_jobs)
        clt.fit(encodings)

        # determine the total number of unique faces found in the dataset
        label_ids = np.unique(clt.labels_)
        num_unique_faces = len(np.where(label_ids > -1)[0])
        print("[INFO] # unique faces: {}".format(num_unique_faces))

        return clt.labels_


class FaceImageGenerator:
    def __init__(self, encoding_file_path):
        self.EncodingFilePath = encoding_file_path
        self.resizeUtils = FaceClusteringLibrary.ResizeUtils()

    def generate_images(self, labels, output_folder_name="ClusteredFaces", montage_output_folder="Montage"):
        output_directory = os.getcwd()

        output_folder = os.path.join(output_directory, output_folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            shutil.rmtree(output_folder)
            time.sleep(0.5)
            os.makedirs(output_folder)

        montage_folder_path = os.path.join(output_folder, montage_output_folder)
        os.makedirs(montage_folder_path)

        data = pickle.loads(open(self.EncodingFilePath, "rb").read())
        data = np.array(data)

        label_ids = np.unique(labels)
        # loop over the unique face integers
        for labelID in label_ids:
            # find all indexes into the `data` array that belong to the
            # current label ID, then randomly sample a maximum of 25 indexes
            # from the set

            print("[INFO] faces for face ID: {}".format(labelID))

            face_folder = os.path.join(output_folder, "Face_" + str(labelID))
            os.makedirs(face_folder)

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

                width_margin = 100
                height_margin = 150

                top = o_top - height_margin
                if top < 0:
                    top = 0

                bottom = o_bottom + height_margin
                if bottom > height:
                    bottom = height

                left = o_left - width_margin
                if left < 0:
                    left = 0

                right = o_right + width_margin
                if right > width:
                    right = width

                portrait = image[top:bottom, left:right]

                if len(portraits) < 25:
                    portraits.append(portrait)

                portrait = self.resizeUtils.rescale_by_width(portrait, 400)

                face_filename = "face_" + str(counter) + ".jpg"

                face_image_path = os.path.join(face_folder, face_filename)
                cv2.imwrite(face_image_path, portrait)

                width_margin = 20
                height_margin = 20

                top = o_top - height_margin
                if top < 0:
                    top = 0

                bottom = o_bottom + height_margin
                if bottom > height:
                    bottom = height

                left = o_left - width_margin
                if left < 0:
                    left = 0

                right = o_right + width_margin
                if right > width:
                    right = width

                annotation_filename = "face_" + str(counter) + ".txt"
                annotation_file_path = os.path.join(face_folder, annotation_filename)

                f = open(annotation_file_path, 'w')
                f.write(str(labelID) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + "\n")
                f.close()

                counter += 1

            montage = build_montages(portraits, (96, 120), (5, 5))[0]

            montage_filename_path = os.path.join(montage_folder_path, "Face_" + str(labelID) + ".jpg")
            cv2.imwrite(montage_filename_path, montage)


if __name__ == "__main__":
    EncodingPickleFilePath = "encodings.pickle"

    faceClusterUtility = FaceClusterUtility(EncodingPickleFilePath)
    faceImageGenerator = FaceImageGenerator(EncodingPickleFilePath)

    labelIDs = faceClusterUtility.cluster()
    faceImageGenerator.generate_images(labelIDs, "ClusteredFaces", "Montage")
