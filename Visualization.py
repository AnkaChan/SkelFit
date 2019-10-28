from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import json
import glob
import os
import pyvista as pv
import vtk
import tqdm

def fittingToVtk(inFitFolder, observeHistograms = None, removeUnobserved = False, visualizeFittingError = False,
                 meshWithFaces=r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\cornersRegisteredToSMpl.ply'):
    outVTKFolder = inFitFolder + r'\vtk'
    os.makedirs(outVTKFolder, exist_ok=True)
    meshWithFaces = pv.read(meshWithFaces)

    if observeHistograms is not None:
        if removeUnobserved:
            unobserved = np.where(observeHistograms == 0)[0]
            faceIdToPreserve = []
            faces = meshWithFaces.faces.reshape(-1, 4)
            for i in range(faces.shape[0]):
                if not (int(faces[i][1]) in unobserved or int(faces[i][2]) in unobserved or int(faces[i][3]) in unobserved):
                    faceIdToPreserve.append(i)
            faces = faces[faceIdToPreserve, :]
            nFaces = faces.shape[0]
            # meshWithFaces.n_faces = nFaces
            meshWithFaces.faces = faces.flatten()

    objFiles = glob.glob(inFitFolder + r'\*.obj')
    errFolder = inFitFolder + r'\Errs'
    for objF in tqdm.tqdm(objFiles):
        fp = Path(objF)

        mesh = pv.read(objF)
        mesh.faces = meshWithFaces.faces

        if visualizeFittingError:
            errs = np.loadtxt(errFolder + '\\' + fp.stem + '.txt')
            mesh.point_arrays['Errs'] = errs

        mesh.save(outVTKFolder + r'\\' + fp.stem + '.vtk')

def obj2vtk(objF, vtkF, meshWithFaces=r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\cornersRegisteredToSMpl.ply'):
    meshWithFaces = pv.read(meshWithFaces)
    mesh = pv.read(objF)
    mesh.faces = meshWithFaces.faces
    mesh.save(vtkF)

def highlightTarget(vtkF, vtkFHighlighted, highlightIds):
    target = pv.read(vtkF)
    highlightMask = np.zeros(target.points.shape[0])
    highlightMask[highlightIds] = 1
    target['Highlight'] = highlightMask
    target.save(vtkFHighlighted)


def writeCorrs(scanFile, fitFile, outCorrFile, outTargetFile):
    scanData = pv.PolyData()
    jData = json.load(open(scanFile))
    scanData.points = np.array(jData["Pts"])

    fittingData = pv.PolyData(fitFile)
    goodPts = np.array(scanData.points[:,2]>0)

    numPts = scanData.points.shape[0]

    ptsVtk = vtk.vtkPoints()
    ptsAll = np.vstack([scanData.points, fittingData.points])
    # pts.InsertNextPoint(p1)
    for i in range(ptsAll.shape[0]):
        ptsVtk.InsertNextPoint(ptsAll[i, :].tolist())

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsVtk)

    lines = vtk.vtkCellArray()

    for i in range(goodPts.shape[0]):
        if goodPts[i]:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)  # the second 0 is the index of the Origin in the vtkPoints
            line.GetPointIds().SetId(1, i + numPts)  # the second 1 is the index of P0 in the vtkPoints
            # line.
            lines.InsertNextCell(line)

    polyData.SetLines(lines)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polyData)
    writer.SetFileName(outCorrFile)
    writer.Update()

    if outTargetFile != '':
        scanData.save(outTargetFile)

def visualizeCorrs(targetFiles, fittingDataFolder, outputFolder, sanityCheck=True):
    objFiles = glob.glob(fittingDataFolder + r'\*.obj')
    objFiles.sort()
    os.makedirs(outputFolder, exist_ok=True)
    if sanityCheck:
        assert (len(objFiles) == len(targetFiles))

    for targetF, objF in tqdm.tqdm(zip(targetFiles, objFiles)):
        objFP = Path(objF)
        targetFP = Path(targetF)
        assert (objFP.stem == targetFP.stem)
        writeCorrs(targetF, objF, outputFolder + r'\Corrs' + objFP.stem + '.vtk', outputFolder + r'\Target' + objFP.stem + '.vtk')

def VisualizeBones(inSkelJsonFile, outSkelVTK):
    jData = json.load(open(inSkelJsonFile))
    jointPosition = jData["JointPos"]

    numJoints = len(jointPosition[0])

    ptsVtk = vtk.vtkPoints()
    # pts.InsertNextPoint(p1)
    for i in range(numJoints):
        ptsVtk.InsertNextPoint([jointPosition[0][i], jointPosition[1][i], jointPosition[2][i]])

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsVtk)

    lines = vtk.vtkCellArray()

    for i in range(1, numJoints):
        iParent = jData['Parents'].get(str(i))
        if iParent != None:
            line = vtk.vtkLine()

            line.GetPointIds().SetId(0, i)  # the second 0 is the index of the Origin in the vtkPoints
            line.GetPointIds().SetId(1, iParent)  # the second 1 is the index of P0 in the vtkPoints
            lines.InsertNextCell(line)

    polyData.SetLines(lines)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polyData)
    writer.SetFileName(outSkelVTK)
    writer.Update()

def VisualizeVertRestPose(inSkelJsonFile, outSkelVTK, visualizeWeights = True, observeHistograms = None, removeUnobserved = False,
        visualizeFittingError = False, fittingErrorFolder = '', visualizeBoneActivation = False, chunked = False,
        meshWithFaces = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\cornersRegisteredToSMpl.ply'):
    meshWithFaces = pv.read(meshWithFaces)

    jData = json.load(open(inSkelJsonFile))
    jointPosition = np.array(jData["VTemplate"])

    mesh = pv.PolyData()
    mesh.points = np.transpose(jointPosition[0:3, :])

    if visualizeWeights:
        weights = np.array(jData["Weights"])
        numJoints = weights.shape[0]

        for i in range(numJoints):
            mesh.point_arrays['Weight_%02i' % i] = weights[i, :]

    if observeHistograms is not None:
        mesh.point_arrays["TimesObserved"] = observeHistograms
        if removeUnobserved:
            unobserved = np.where(observeHistograms == 0)[0]
            faceIdToPreserve = []
            faces = meshWithFaces.faces.reshape(-1, 4)
            for i in range(faces.shape[0]):
                if not (int(faces[i][1]) in unobserved or int(faces[i][2]) in unobserved or int(faces[i][3]) in unobserved):
                    faceIdToPreserve.append(i)
            faces = faces[faceIdToPreserve, :]
            nFaces = faces.shape[0]
            # meshWithFaces.n_faces = nFaces
            meshWithFaces.faces = faces.flatten()

    if visualizeFittingError and fittingErrorFolder != '':
        if chunked:
            errs = np.transpose(np.array(json.load(open(fittingErrorFolder))))
        else:
            errFiles = glob.glob(fittingErrorFolder + r'\*.txt')
            errs = np.zeros((mesh.points.shape[0], len(errFiles)))

            for i, errF in enumerate(errFiles):
                errs[:,i] = np.loadtxt(errF)

        maxErrs = errs.max(axis = 1)
        avgErrs = np.sum(errs, axis=1) / (np.sum(errs!=0, axis=1)+0.001)
        mesh.point_arrays['MaxErr'] = maxErrs
        mesh.point_arrays['AvgErr'] = avgErrs
    if visualizeBoneActivation:
        numBones = weights.shape[0]
        activeBoneMarks = np.zeros(weights.shape)
        activeBoneTable = jData["ActiveBoneTable"]
        for i, table in enumerate(activeBoneTable):
            activeBoneMarks[table, i] = 1
        for iBone in range(numBones):
            mesh.point_arrays['Activation_%02i' % iBone] = activeBoneMarks[iBone, :]


    mesh.faces = meshWithFaces.faces
    # mesh.n_faces = meshWithFaces.n_faces
    mesh.save(outSkelVTK)