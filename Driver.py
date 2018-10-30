from FaceClusteringLibrary import *

if __name__ == "__main__":
    ''' Generate the frames from given video footage '''
    framesGenerator = FramesGenerator("Footage.mp4")
    framesGenerator.GenerateFrames("Frames")


    ''' Design and run the face clustering pipeline '''
    CurrentPath = os.getcwd()
    FramesDirectory = "Frames"
    FramesDirectoryPath = os.path.join(CurrentPath, FramesDirectory)
    EncodingsFolder = "Encodings"
    EncodingsFolderPath = os.path.join(CurrentPath, EncodingsFolder)

    if os.path.exists(EncodingsFolderPath):
        shutil.rmtree(EncodingsFolderPath, ignore_errors=True)
        time.sleep(0.5)
    os.makedirs(EncodingsFolderPath)

    pipeline = Pipeline(
                    FramesProvider("Files source", sourcePath=FramesDirectoryPath) | 
                    FaceEncoder("Encode faces") | 
                    DatastoreManager("Store encoding", 
                    encodingsOutputPath=EncodingsFolderPath), 
                    n_threads = 3, quiet = True)
    pbar = TqdmUpdate()
    pipeline.run(update_callback=pbar.update)

    print()
    print('[INFO] Encodings extracted')


    ''' Merge all the encodings pickle files into one '''
    CurrentPath = os.getcwd()
    EncodingsInputDirectory = "Encodings"
    EncodingsInputDirectoryPath = os.path.join(CurrentPath, EncodingsInputDirectory)

    OutputEncodingPickleFilename = "encodings.pickle"

    if os.path.exists(OutputEncodingPickleFilename):
        os.remove(OutputEncodingPickleFilename)

    picklesListCollator = PicklesListCollator(EncodingsInputDirectoryPath)
    picklesListCollator.GeneratePickle(OutputEncodingPickleFilename)

    # To manage any delay in file writing
    time.sleep(0.5)


    ''' Start clustering process and generate output images with annotations '''
    EncodingPickleFilePath = "encodings.pickle"
    
    faceClusterUtility = FaceClusterUtility(EncodingPickleFilePath)
    faceImageGenerator = FaceImageGenerator(EncodingPickleFilePath)
    
    labelIDs = faceClusterUtility.Cluster()
    faceImageGenerator.GenerateImages(labelIDs, "ClusteredFaces", "Montage")

