import numpy as np
import tqdm
import json
import glob
import pyvista as pv
import os
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path

def makeBatchFile(files, batchFile):
    json.dump({"BatchFiles": files}, open(batchFile, 'w'), indent=2)

def makeBatchFileFromFolder(folder, ext, batchFile, sort=True):
    files = glob.glob(folder + r'\*.' + ext)
    if sort:
        files.sort()
    makeBatchFile(files, batchFile)


def toFrameData(infile, outJasonFile):
    capture = pv.read(infile).points
    frameData = {
        'Pts': capture.tolist()
    }
    json.dump(frameData, open(outJasonFile, 'w'), indent=2)

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


def SampleDataSatistics(sampleData):
    sampleMaskes = np.zeros((sampleData.shape[0], 1487))
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

        cmd = [exePath, inBatchFile, fittingFolder, '-s', modelInput, '-r', str(threshold), '--outputSkipStep', '100',
               '--outputFittedModels', '0']
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
