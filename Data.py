import numpy as np
import tqdm
import json
import glob
import pyvista as pv
import os
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import math
from os.path import join
import itertools

def removeOutliers(inChunkFile, newChunkFile, outlierIds):
    data = json.load(open(inChunkFile))
    pts = np.array(data['Pts'])
    pts[:, outlierIds, :] = [0,0,-1]

    data['Pts'] = pts.tolist()

    json.dump(data, open(newChunkFile, 'w'), indent=2)
    return newChunkFile

def toPolyData(verts, faces):
    nFaces = np.array([len(vs) for vs in faces])
    faces = np.hstack([nFaces.reshape(nFaces.shape[0],1), faces])

    mesh = pv.PolyData(verts, faces)
    return mesh


def readBatchedSkelParams(inBatchFile, numRotations=16):
    files = json.load(open(inBatchFile))['BatchFiles']

    quaternions = []
    translations = []
    for pf in files:
        r,t = readSkelParams(pf)
        quaternions.append(r)
        translations.append(t)

    return quaternions, translations, files

def readSkelParams(inFile, numRotations=16):
    fpPose = open(inFile)

    paramsRaw = fpPose.read().split('\n')
    r = np.fromstring('\n'.join(paramsRaw[:numRotations]), sep=' ').reshape(numRotations, -1)
    t = np.fromstring(paramsRaw[numRotations], sep=' ')
    return r, t

def getFrameName(fileName):
    return Path(fileName).stem

def makeBatchFile(files, batchFile):
    json.dump({"BatchFiles": files}, open(batchFile, 'w'), indent=2)

def makeBatchFileFromFolder(folder, ext, batchFile, sort=True, range=None, step=1):
    files = glob.glob(folder + r'\*.' + ext)
    if sort:
        files.sort()
    if range is not None:
        files = files[range[0]:range[1]:step]
    makeBatchFile(files, batchFile)


def toFrameData(infile, outJasonFile):
    capture = pv.read(infile).points
    frameData = {
        'Pts': capture.tolist()
    }
    json.dump(frameData, open(outJasonFile, 'w'), indent=2)


def pointCloudFilesToJsonBatch(inFrameDataFolder, jsonFrameDataFolder, extName='obj', processJsonInterval = []):
    os.makedirs(jsonFrameDataFolder, exist_ok=True)

    cloudFiles = glob.glob(inFrameDataFolder + r'\*.' + extName)
    
    outFile = 'BatchFile.json'

    if processJsonInterval is not None:
        if len(processJsonInterval) == 2:
            cloudFiles = cloudFiles[processJsonInterval[0]: processJsonInterval[1]]
            outFile = 'BatchFile' + str(processJsonInterval[0]) + "_" + str(processJsonInterval[1]) + '.json'

    jsonFiles = []
    for f in tqdm.tqdm(cloudFiles):
        pf = Path(f)
        jFileName = jsonFrameDataFolder + r'\\' + pf.stem + '.json'
        toFrameData(f, jFileName)
        jsonFiles.append(jFileName)

    s = glob.glob(jsonFrameDataFolder + r'\*.json')


    outFile = join(jsonFrameDataFolder, outFile)
    json.dump({"BatchFiles" : jsonFiles}, open(outFile, 'w'), indent=2)

    return outFile


def pointCloudFilesToChunk(inFolder, chunkFile, interval = None, indent=2, inputExt = 'obj'):
    inFiles = glob.glob(join(inFolder, '*.'+inputExt))
    if interval is not None:
        inFiles = inFiles[interval[0]:interval[1]]

    allPts = []
    for pf in tqdm.tqdm(inFiles):
        pc = pv.PolyData(pf)
        allPts.append(pc.points.tolist())

    if interval is not None:
        filename, file_extension = os.path.splitext(chunkFile)
        chunkFile = filename + '_' + str(interval[0]) + '_' + str(interval[1]) + file_extension

    json.dump({"BatchFiles": inFiles, "Pts": allPts}, open(chunkFile, 'w'), indent=indent)

def scanFilesToChunkFromBatchFile(batchFile, chunkFile, interval =[], indent=2):
    files = json.load(open(batchFile))["BatchFiles"]

    allPts = []
    if len(interval) == 2:
        files = files[interval[0]: interval[1]]

    for f in tqdm.tqdm(files):
        allPts.append(json.load(open(f))["Pts"])

    json.dump({"BatchFiles":files, "Pts":allPts}, open(chunkFile, 'w'), indent=indent)

def killOutliers(targetFiles, fittingFoler, threshold, outFolderKillOutliers, chunkedOutput=False):
    outFilesKillOutlier = []
    for tf in tqdm.tqdm(targetFiles):
        tfp = Path(tf)
        objF = fittingFoler + '\\' + tfp.stem + '.obj'
        fit = pv.PolyData(objF)
        fitpts = fit.points

        tjdata = json.load(open(tf))
        targetPoints = np.array(tjdata["Pts"])

        newTargetPoints = np.copy(targetPoints)

        for i in range(fitpts.shape[0]):
            if targetPoints[i, 2] > 0:
                dis = np.linalg.norm(targetPoints[i, :] - fitpts[i, :])
                if dis > threshold:
                    # print("Outlier in: ", objF)
                    # print("Vid: ", i, "Dis: ", dis)
                    newTargetPoints[i, :] = np.array([0, 0, -1])

        outFile = outFolderKillOutliers + '\\' + tfp.stem + '.json'
        outFilesKillOutlier.append(outFile)
        json.dump({"Pts": newTargetPoints.tolist()}, open(outFile, 'w'), indent=2)

    outBatchJsonFile = outFolderKillOutliers + '\\' + 'BatchFile.json'
    json.dump({"BatchFiles":outFilesKillOutlier}, open(outBatchJsonFile, 'w'), indent=2)
    return outBatchJsonFile

def killOutliersChunked(chunkedTargetFile, chunkedErrsFile, threshold, outFile):
    chunkedTargetData = json.load(open(chunkedTargetFile))

    ptsAll = chunkedTargetData["Pts"]
    errsAll = json.load(open(chunkedErrsFile))

    newPts = []
    for pts, errs in zip(ptsAll, errsAll):
        pts = np.array(pts)
        errs = np.array(errs)
        pts[np.where(errs>threshold)[0], :] = np.array([0,0,-1])
        newPts.append(pts.tolist())

    chunkedTargetData["Pts"] = newPts
    json.dump(chunkedTargetData, open(outFile, 'w'), indent=2)
    return outFile, errs

def readFittingErrs(inErrFolder):
    files = glob.glob(inErrFolder + r'\*.txt')
    allErrs = []
    for f in tqdm.tqdm(files):
        errsF = np.loadtxt(f)
        errsF = errsF[np.where(errsF != 0.)[0]]
        errsF = errsF[:, np.newaxis]
        allErrs.append(errsF)
    errs = np.vstack(allErrs)
    return errs

def readScannedData(files):
    nFrames = len(files)
    allData = np.zeros((nFrames, 1487, 3))

    scannedMasks = np.zeros((nFrames, 1487))
    for i, f in enumerate(files):
        jBatchData = json.load(open(f))
        pts = np.array(jBatchData["Pts"])
        allData[i, :, :] = pts
        scannedMasks[i, :] = pts[:, 2] > 0
    return allData, scannedMasks

def readChunkScannedData(chunkFile, getObsMask=True, getFileNames=False):
    chunkedTargetData = json.load(open(chunkFile))

    ptsAll = np.array(chunkedTargetData["Pts"])

    scannedMasks = np.zeros((ptsAll.shape[0], 1487))
    if getObsMask:
        for iFrame in range(ptsAll.shape[0]):
            scannedMasks[iFrame, :] = ptsAll[iFrame, :, 2] > 0
    if getFileNames:
        return ptsAll, scannedMasks, chunkedTargetData["BatchFiles"]
    else:
        return ptsAll, scannedMasks

def unpackChunkData(chunkFile, outFolder, interval=[], outBatchFileName = "BatchFile", addRange=True, indent=2, outputType="json"):
    ptsAll, _, files = readChunkScannedData(chunkFile, getObsMask=False, getFileNames=True)

    itRange = range(len(files)) if len(interval) !=2 else range(interval[0], interval[1])
    os.makedirs(outFolder, exist_ok=True)
    outJsonFiles = []
    for i in tqdm.tqdm(itRange, desc="Unpacking Chunk File"):
        pts = ptsAll[i,:,:].tolist()
        f = files[i]
        fp = Path(f)

        if outputType == "json":
            outFile = os.path.join(outFolder, fp.stem + ".json")
            json.dump({"Pts": pts}, open(outFile, 'w'), indent=indent)
        elif outputType == "ply" or outputType == "obj":
            data = pv.PolyData()
            data.points = ptsAll[i,:,:]
            outFile = os.path.join(outFolder, fp.stem + "." + outputType)
            data.save(outFile, binary=False)

        outJsonFiles.append(outFile)

    if addRange and len(interval) == 2:
        rangeStr = str(interval[0]) + "_" + str(interval[1])
    else:
        rangeStr = ""
    batchFileName = os.path.join(outFolder, outBatchFileName + rangeStr + ".json")
    json.dump({"BatchFiles":outJsonFiles}, open(batchFileName, 'w'), indent=indent)


def SampleDataSatistics(sampleData, numPts = 1487):
    sampleMaskes = np.zeros((sampleData.shape[0], numPts))
    sampleMaskes[:, :] = sampleData[:, :, 2] > 0
    appearenceTimes = np.sum(sampleMaskes, axis=0)
    return appearenceTimes


def filterOutliers(inBatchFile, cacheFolder, filteringThreshold,
                   modelInput=r"C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\PrepareData1487\005_SkelData1487MMBetterSparsity.json",
                   plotHistogram = False, chunkInput=False, chunkOutput=False,
                   exePath = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\CornersFitReducedQuaternionFToF1487v2.exe'):

    for threshold in filteringThreshold:

        iterationOutFolder = cacheFolder + '\\' + 'Run' + str(threshold)
        fittingFolder = iterationOutFolder + '\\' + 'Fitting'
        os.makedirs(fittingFolder, exist_ok=True)

        cmd = [exePath, inBatchFile, fittingFolder, '-s', modelInput, '-r', str(threshold), '--outputSkipStep', '100']
        if chunkInput:
            cmd.append('-c')
        if chunkOutput:
            cmd.append('--outputChunkFile')


        subprocess.call(cmd)
        outFolderKillOutliers = iterationOutFolder + '\\Filtered'

        outChunkFileKillOutliers = outFolderKillOutliers + r'\ChunkFile.json'
        errFile = fittingFolder + r'\Errs\Errors.json'

        os.makedirs(outFolderKillOutliers, exist_ok=True)
        if chunkOutput:
            if not chunkInput:
                print("Currently non-chunked input is not supported!")
                assert False
            else:
                inBatchFile, errs = killOutliersChunked(inBatchFile, errFile, threshold, outChunkFileKillOutliers)
        else:
            jBatchData = json.load(open(inBatchFile))
            files = jBatchData["BatchFiles"]
            inBatchFile = killOutliers(files, fittingFolder, threshold, outFolderKillOutliers)

        if plotHistogram:
            if not chunkOutput:
                errs = readFittingErrs(fittingFolder + r'\Errs')

            n, bins, patches = plt.hist(x=errs, bins='auto', color='#0504aa',
                                        alpha=0.7, rwidth=0.85, log=True)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('My Very Own Histogram')
            plt.text(23, 45, r'$\mu=15, b=3$')
            maxfreq = n.max()
            # Set a clean upper y-axis limit.
            plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
            plt.show()

    return inBatchFile

def QuaternionToAngleAxis(quaternion):
  angle_axis = np.zeros((3))

  q1 = quaternion[1];
  q2 = quaternion[2];
  q3 = quaternion[3];
  sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

  # For quaternions representing non-zero rotation, the conversion
  # is numerically stable.
  if (sin_squared_theta > 0.0):
    sin_theta = sin_squared_theta
    cos_theta = quaternion[0]

    if (cos_theta < 0.0):
      two_theta = 2.0 * math.atan2(-sin_theta, -cos_theta)
    else:
      two_theta = 2.0 * math.atan2(sin_theta, cos_theta)

    k = two_theta / sin_theta
    angle_axis[0] = q1 * k
    angle_axis[1] = q2 * k
    angle_axis[2] = q3 * k
  else:
    k = 2.0
    angle_axis[0] = q1 * k
    angle_axis[1] = q2 * k
    angle_axis[2] = q3 * k
  return angle_axis

def loadPoseChunkFile(poseChunkFile):
    poseData = json.load(open(poseChunkFile))
    quaternions = [p['JointAngles'] for p in poseData]
    translations = [p['Translation'] for p in poseData]

    return quaternions, translations

def captureStatistics(inFolder, outFolder=None, extName='obj'):

    pointcloudFiles = glob.glob(join(inFolder, '*.'+extName))

    observations = []
    obsMaskEachFrame = []
    numObservedPts = []
    for pcF in tqdm.tqdm(pointcloudFiles):
        pc = pv.PolyData(pcF)
        observations.append(pc.points)

        obsMask = pc.points[:, 2] != -1
        obsMaskEachFrame.append(obsMask)
        numObservedPts.append(np.where(obsMask)[0].shape[0])

    observations = np.array(observations)
    obsMaskEachFrame = np.array(obsMaskEachFrame)
    numObservedPts = np.array(numObservedPts)

    if outFolder is not None:
        os.makedirs(outFolder, exist_ok=True)

        outObsFile = join(outFolder, 'All.npy')
        outObsMaskEachFrameFile = join(outFolder, 'ObsMaskEachFrame.npy')
        outNumObservedPtsFile = join(outFolder, 'NumObservedPts.npy')

        np.save(outObsFile,  observations)
        np.save(outObsMaskEachFrameFile,  obsMaskEachFrame)
        np.save(outNumObservedPtsFile,  numObservedPts)

    return observations, obsMaskEachFrame, numObservedPts, pointcloudFiles

def searchQuadIds(qCode, codeSet):
    qvIds = [-1, -1, -1, -1]

    for iV, cCode in enumerate(codeSet):
        ids = [i for i, x in enumerate(cCode['Code']) if x == qCode]
        if len(ids) != 0:
            assert  len(ids) == 1
            qvIds[cCode['Id'][ids[0]]] = iV

    return qvIds

def generateQuadStructure(pts, codeSet,characterDictionary=['1', '2', '3', '4', '5', '6', '7', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'K', 'L',
                           'M', 'P', 'Q', 'R', 'T', 'U', 'V', 'Y']):
    quadsCodes=[]
    quads = []
    ptsUsedMask = np.zeros(pts.shape[0])

    for c1, c2 in itertools.product(characterDictionary, characterDictionary):
        quadsCodes.append(c1 + c2)
    for qc in quadsCodes:
        # print(qc)
        qvIds = searchQuadIds(qc, codeSet)

        if -1 not in qvIds:
            if np.all(pts[qvIds, 2] != -1):
                quads.append(qvIds)
                # print(qvIds)
                ptsUsedMask[qvIds] = 1

    return quads, ptsUsedMask

def readCIdFile(cIdFile):
    CIDList = open(cIdFile).read()
    cornerCodes = CIDList.split('\n')
    codeSet = []
    for i in range(len(cornerCodes)):
        if len(cornerCodes[i]) >= 3:
            code2 = cornerCodes[i].split(' ')
            codeSet.append({'Code': [c[:2] for c in code2], 'Id': [int(c[2:]) for c in code2]})

    return codeSet