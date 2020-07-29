from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import json
import glob
import os
import pyvista as pv
import vtk
import tqdm
import json
from .SkeletonModel import  *

def jsonToVtk(batchFile, outVTKFolder, addFaces = False, meshWithFaces=r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\cornersRegisteredToSMpl.ply'):
    jsonFiles = json.load(open(batchFile))["BatchFiles"]
    os.makedirs(outVTKFolder, exist_ok=True)

    if addFaces:
        meshWithFaces = pv.PolyData(meshWithFaces)

    for jF in tqdm.tqdm(jsonFiles):
        fp = Path(jF)
        pts = json.load(open(jF))["Pts"]
        mesh = pv.PolyData()

        mesh.points = np.array(pts)

        if addFaces:
            faceIdToPreserve = []
            faces = meshWithFaces.faces.reshape(-1, 4)
            for i in range(faces.shape[0]):
                if pts[faces[i][1]][2] != -1 and pts[faces[i][2]][2] != -1 and pts[faces[i][3]][2] != -1:
                    faceIdToPreserve.append(i)
            faces = faces[faceIdToPreserve, :]
            nFaces = faces.shape[0]
            mesh.faces = faces.flatten()

        mesh.save(outVTKFolder + r'\\' + fp.stem + '.vtk')


def fittingToVtk(inFitFolder, observeHistograms = None, removeUnobserved = False, visualizeFittingError = False, outVTKFolder = None, extName= 'obj', addABeforeName = False, outExtName='vtk',
                 meshWithFaces=r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\cornersRegisteredToSMpl.ply', addGround =False, groundLevel=0):

    if outVTKFolder is None:
        outVTKFolder= inFitFolder + r'\vtk'
    os.makedirs(outVTKFolder, exist_ok=True)
    if meshWithFaces is not None:
        meshWithFaces = pv.read(meshWithFaces)
    else:
        meshWithFaces = None

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


    objFiles = glob.glob(inFitFolder + r'\*.' + extName)
    errFolder = inFitFolder + r'\Errs'
    for objF in tqdm.tqdm(objFiles, desc='Fitting to vtk'):
        fp = Path(objF)

        mesh = pv.read(objF)
        mesh.faces = meshWithFaces.faces

        if visualizeFittingError:
            errs = np.loadtxt(errFolder + '\\' + fp.stem + '.txt')
            mesh.point_arrays['Errs'] = errs
        if observeHistograms is not None:
            mesh.point_arrays['NumPts'] = observeHistograms

        if addGround:
            nPts = mesh.points.shape[0]
            groundPts =  np.array([[3000, 3000, 0], [-3000, 3000, 0], [-3000, -3000, 0], [3000, -3000, 0]])
            points = np.vstack([mesh.points, groundPts])
            faces = np.hstack([mesh.faces, [4, nPts, nPts + 1, nPts + 2, nPts + 3]])

            mesh = pv.PolyData(points, faces)

        if addABeforeName:
            mesh.save(outVTKFolder + r'\\A' + fp.stem + '.' + outExtName)
        else:
            mesh.save(outVTKFolder + r'\\' + fp.stem + '.' + outExtName)

def obj2vtkFolder(inObjFolder, inFileExt='obj', outVtkFolder=None, processInterval=[], addFaces = False, addABeforeName=True, faceMesh=None):


    # addFaces = True
    if outVtkFolder is None:
        outVtkFolder = inObjFolder + r'\vtk'
        
    objFiles = glob.glob(inObjFolder + r'\*.' + inFileExt)
    
    if faceMesh is not None:
        meshWithFaces = pv.read(faceMeshFile)
    else:
        meshWithFaces = pv.PolyData()
        
    os.makedirs(outVtkFolder, exist_ok=True)
    if len(processInterval) == 2:
        objFiles = objFiles[processInterval[0]: processInterval[1]]
    for f in tqdm.tqdm(objFiles, desc=inFileExt + " to vtk"):
        fp = Path(f)

        mesh = pv.read(f)
        if addFaces:
            mesh.faces = meshWithFaces.faces
        # else:
        #     mesh.faces = np.empty((0,), dtype=np.int32)
        if addABeforeName:
            outName = outVtkFolder + r'\\A' + fp.stem + '.vtk'
        else:
            outName = outVtkFolder + r'\\' + fp.stem + '.vtk'
        mesh.save(outName)

def obj2vtk(objF, vtkF, meshWithFaces=r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\cornersRegisteredToSMpl.ply'):
    mesh = pv.read(objF)
    if meshWithFaces is not None:
        meshWithFaces = pv.read(meshWithFaces)
        mesh.faces = meshWithFaces.faces
    mesh.save(vtkF)

def highlightTarget(vtkF, vtkFHighlighted, highlightIds):
    target = pv.read(vtkF)
    highlightMask = np.zeros(target.points.shape[0])
    highlightMask[highlightIds] = 1
    target['Highlight'] = highlightMask
    target.save(vtkFHighlighted)

def drawCorrs(pts1, pts2, outCorrFile):
    ptsVtk = vtk.vtkPoints()
    ptsAll = np.vstack([pts1, pts2])
    numPts = pts1.shape[0]

    assert pts1.shape[0] == pts2.shape[0]

    # pts.InsertNextPoint(p1)
    for i in range(ptsAll.shape[0]):
        ptsVtk.InsertNextPoint(ptsAll[i, :].tolist())

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsVtk)

    lines = vtk.vtkCellArray()

    for i in range(pts1.shape[0]):
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

def writeCorrs(scanFile, fitFile, outCorrFile, outTargetFile):

    meshFilename, meshFileExtension = os.path.splitext(scanFile)

    if meshFileExtension == '.json':
        scanData = pv.PolyData()
        jData = json.load(open(scanFile))
        scanData.points = np.array(jData["Pts"])
    else:
        scanData = pv.PolyData(scanFile)

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

def visualizeCorrs(targetFiles, fittingDataFolder, outputFolder, sanityCheck=True, fitFileExt = 'obj'):
    objFiles = glob.glob(fittingDataFolder + r'\*.' + fitFileExt)
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
        visualizeFittingError = False, fittingErrorFolder = '', visualizeBoneActivation = False, chunked = False, addLogScaleWeights = False,
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
            if addLogScaleWeights:
                mesh.point_arrays['Weights_Log_%02i' % i] = np.log(np.abs(10e-16 + weights[i, :]))

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


def drawBones(bones, outFile):
    ptsVtk = vtk.vtkPoints()
    ptsAll = np.vstack(bones)
    # pts.InsertNextPoint(p1)
    for i in range(ptsAll.shape[0]):
        ptsVtk.InsertNextPoint(ptsAll[i, :].tolist())

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsVtk)

    lines = vtk.vtkCellArray()

    for i in range(int(ptsAll.shape[0]/2)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i * 2)  # the second 0 is the index of the Origin in the vtkPoints
        line.GetPointIds().SetId(1, i * 2 +1)  # the second 1 is the index of P0 in the vtkPoints
        # line.
        lines.InsertNextCell(line)

    polyData.SetLines(lines)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polyData)
    writer.SetFileName(outFile)
    writer.Update()

def visualizeSkeleton(paramChunkFile, skelDataFile, outFolder):
    os.makedirs(outFolder, exist_ok =True)
    vRestpose, J, weights, poseBlendShape, kintreeTable, parent, faces = readSkeletonData(skelDataFile)
    params = json.load(open(paramChunkFile))

    bones = [(bone[1], bone[0]) for bone in parent.items()]
    bonesInitialData = [np.array([J[bone[0], :], J[bone[1], :]]) for bone in bones]

    drawBones(bonesInitialData, join(outFolder, 'BonesInitialPoses.vtk'))

    for ip, param in enumerate(params):
        jointAngles = param['JointAngles']
        translations = param['Translation']
        RMats = quaternionsToRotations(jointAngles)
        translations = np.array(translations)

        v, G = deformVertsWithJointTranslation(vRestpose, RMats, translations, J, weights, kintreeTable, parent, returnTransf=True)

        print(G)

        mesh = Data.toPolyData(v, faces)
        outDeformedFile = join(outFolder, 'A' + str(ip).zfill(5) + '.ply')
        mesh.save(outDeformedFile)

        boneDataDeformed = []
        for bone, boneData in zip(bones, bonesInitialData):
            T = G[bone[0], :, :]
            newBoneData = np.hstack([T @ np.vstack([boneData[0,:].reshape(3,1) , [1]]), T @ np.vstack([boneData[1,:].reshape(3,1), [1]])])
            newBoneData = newBoneData[:3, :].transpose()
            newBoneData[0, :] += translations[0,:]
            newBoneData[1, :] += translations[0,:]

            boneDataDeformed.append(newBoneData)

        drawBones(boneDataDeformed, join(outFolder, 'Bones' +str(ip).zfill(5)+ '.vtk'))

if __name__ == '__main__':
    deformedCompleteMeshFolder = r'F:\WorkingCopy2\2020_01_13_FinalAnimations\Katey_NewJointTLap\LongSequence\SLap_SBiLap_True_TLap_50_JTW_100000_JBiLap_0_Step200_Overlap100\Deformed'
    finalVisFolder = r'F:\WorkingCopy2\2020_01_13_FinalAnimations\Katey_NewJointTLap\LongSequence\SLap_SBiLap_True_TLap_50_JTW_100000_JBiLap_0_Step200_Overlap100\Vis\Final'
    inOriginalRestPoseQuadMesh = r'Z:\2020_01_16_KM_Edited_Meshes\KateyCalibrated_edited2.obj'

    fittingToVtk(deformedCompleteMeshFolder, outVTKFolder=finalVisFolder, visualizeFittingError=False, addABeforeName=True, addGround=True,
                 meshWithFaces=inOriginalRestPoseQuadMesh)