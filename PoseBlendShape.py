# import SMPL_reimp.smpl_np as smpl_np
from matplotlib import pyplot as plt
import numpy as np
from SkelFit import Data
from SkelFit import Debug
from SkelFit import Visualization
import pyvista as pv
from os.path import join
import json
from scipy.spatial.transform import Rotation as R
from pathlib import Path


def quaternionsToRotations(qs):
    Rs = [R.from_quat([q[1], q[2], q[3], q[0]]) for q in qs]
    Rs = [r.as_dcm() for r in Rs]

    return np.array(Rs)

def applyPoseBlendShapes(vRestpose, poseBlendShapes, R):
    I_cube = np.broadcast_to(
        np.expand_dims(np.eye(3), axis=0),
        (R.shape[0]-1, 3, 3)
    )
    lrotmin = np.squeeze((R[1:] - I_cube).reshape(9 * (R.shape[0]-1), 1))
    # how pose affect body shape in zero pose
    v_posed = vRestpose + poseBlendShapes.dot(lrotmin)

    return v_posed


def deformBatch(paramJsonFile, deformedMeshFolder, skelDataFile,  outExt = 'ply'):
    quanternions, trans, files = Data.readBatchedSkelParams(paramJsonFile)

    skelData = json.load(open(skelDataFile))
    vRestpose = (np.array(skelData['VTemplate']).transpose())[:,:3]
    J = (np.array(skelData['JointPos']).transpose())[:,:3]
    weights = np.array(skelData['Weights']).transpose()
    poseBlendShape = np.array(skelData['PoseBlendShapes'])

    parent = skelData["Parents"]
    parent = {int(key[0]): key[1] for key in parent.items()}

    kintreeTable = np.array(skelData['KintreeTable']).transpose()

    for q, t, f in zip(quanternions, trans, files):
        R = quaternionsToRotations(q)
        deformedVerts = deformWithPoseBlendShapes(vRestpose, poseBlendShape, R, t, J, weights, kintreeTable, parent)
        outFile = join(deformedMeshFolder, Path(f).stem + '.' + outExt)

        mesh = Data.toPolyData(deformedVerts, skelData['Faces'])
        mesh.save(outFile)

def deformWithPoseBlendShapes(vRestpose, poseBlendShapes, R, trans, J, weights, kintree_table, parent):
    # numActiveJoints = weights.shape[1] - len(jointsIdToReduce)
    # allJoints = list(range(0, 24))
    # preservedJoints = allJoints
    # for id in jointsIdToReduce:
    #     preservedJoints.remove(id)
    #
    # R24 = np.array([np.eye(3) for i in range(24)])
    # for iJ, iJOrg in enumerate(preservedJoints):
    #     R24[iJOrg, :, :] = R[iJ]
    #
    # I_cube = np.broadcast_to(
    #     np.expand_dims(np.eye(3), axis=0),
    #     (23, 3, 3)
    # )
    # lrotmin = np.squeeze((R24[1:] - I_cube).reshape(207, 1))
    # # how pose affect body shape in zero pose
    # v_posed = vRestpose + poseBlendShapes.dot(lrotmin)

    v_posed = applyPoseBlendShapes(vRestpose, poseBlendShapes, R)

    return deformVerts(v_posed, R, trans, J, weights, kintree_table, parent)


def pack( x):
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

def with_zeros( x):
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

def deformVerts(vRestpose, R, trans, J, weights, kintree_table, parent):
    numJoints = J.shape[0]
    # world transformation of each joint
    G = np.empty((numJoints, 4, 4))
    G[0] = with_zeros(np.hstack((R[0], J[0, :].reshape([3, 1]))))
    for i in range(1, numJoints):
        G[i] = G[parent[i]].dot(
            with_zeros(
                np.hstack(
                    [R[i], ((J[i, :] - J[parent[i], :]).reshape([3, 1]))]
                )
            )
        )
    # remove the transformation due to the rest pose
    G = G - pack(
        np.matmul(
            G,
            np.hstack([J, np.zeros([numJoints, 1])]).reshape([numJoints, 4, 1])
        )
    )
    # transformation of each vertex
    T = np.tensordot(weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((vRestpose, np.ones([vRestpose.shape[0], 1])))
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    verts = v + trans.reshape([1, 3])

    return verts

def deformVertsWithJointTranslation(vRestpose, R, trans, J, weights, kintree_table, parent, returnTransf=False):
    numJoints = J.shape[0]
    # world transformation of each joint
    G = np.empty((numJoints, 4, 4))
    G[0] = with_zeros(np.hstack((R[0], J[0, :].reshape([3, 1]))))
    for i in range(1, numJoints):
        G[i] = G[parent[i]].dot(
            with_zeros(
                np.hstack(
                    [R[i], ((J[i, :] - J[parent[i], :] + trans[i, :]).reshape([3, 1]))]
                )
            )
        )
    # remove the transformation due to the rest pose
    G = G - pack(
        np.matmul(
            G,
            np.hstack([J, np.zeros([numJoints, 1])]).reshape([numJoints, 4, 1])
        )
    )
    # transformation of each vertex
    T = np.tensordot(weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((vRestpose, np.ones([vRestpose.shape[0], 1])))
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    # verts = v + trans.reshape([1, 3])
    if returnTransf:
        return v, G
    else:
        return v

def transformJoints(R, trans, J, kintree_table, parent):
    numJoints = J.shape[0]

    # world transformation of each joint
    G = np.empty((kintree_table.shape[1], 4, 4))
    G[0] = with_zeros(np.hstack((R[0], J[0, :].reshape([3, 1]))))
    for i in range(1, kintree_table.shape[1]):
        G[i] = G[parent[i]].dot(
            with_zeros(
                np.hstack(
                    [R[i], ((J[i, :] - J[parent[i], :]).reshape([3, 1]))]
                )
            )
        )
    # remove the transformation due to the rest pose
    G = G - pack(
        np.matmul(
            G,
            np.hstack([J, np.zeros([numJoints, 1])]).reshape([numJoints, 4, 1])
        )
    )
    # Transform the joints to the new position:
    newJs = []
    for i in range(G.shape[0]):
        T = G[i, :, :]
        j = np.transpose(J[i, :])
        j = j.reshape((3, 1))
        j = np.vstack([j, 1])
        j = np.matmul(T, j)
        newJs.append(np.transpose(j[0:3, 0]) + trans)

    return np.array(newJs)


if __name__ == '__main__':
    outFolder = r'F:\WorkingCopy2\2020_01_08_SMPL_PoseBlendShapes'

    inOriginalRestPoseMesh = r'Z:\2019_12_27_FinalLadaMesh\FinalMesh2_OnlyQuad\Mesh2_OnlyQua_No_Hole2_Tri.obj'

    model_path = r'C:/Data/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'

    trans = np.load(r"C:\Code\MyRepo\ChbCapture\06_Deformation\SMPLOptimization\ActorModelFit\OptimizedTranslation_ICPTriangle.npy")
    beta = np.load(r"C:\Code\MyRepo\ChbCapture\06_Deformation\SMPLOptimization\ActorModelFit\OptimizedBetas_ICPTriangle.npy")
    pose = np.load(r"C:\Code\MyRepo\ChbCapture\06_Deformation\SMPLOptimization\ActorModelFit\OptimizedPoses_ICPTriangle.npy")

    inCompleteSkelData = r'00_SkelDataManuallyComplete.json'

    smpl_npModel = smpl_np.SMPLModel(model_path)
    smpl_npModel.set_params(pose=pose, beta=beta, trans=trans)
    smplFaces = smpl_npModel.faces

    # for i in range(207):
    #     outMesh = join(outFolder, str(i).zfill(3) + '.obj')
    #
    #     result = smpl_npModel.posedirs[:,:,i] + smpl_npModel.v_template
    #
    #     fp = open(outMesh, 'w')
    #     for v in result:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #
    #     for f in smplFaces + 1:
    #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    #
    # Visualization.obj2vtkFolder(outFolder)

    skelData = json.load(open(inCompleteSkelData))

    jointsIdToReduce = [10, 11, 12, 15, 20, 21, 22, 23]

    vTemplate = np.array(skelData["VTemplate"]).transpose()[:,:3]
    weights = np.array(skelData["Weights"]).transpose()
    J = np.array(skelData["JointPos"]).transpose()[:,:3]

    # kintree_table = np.array(skelData['KintreeTable'])


    parent = skelData['Parents']
    parent = {int(key[0]):key[1] for key in parent.items()}

    numActiveJoints = weights.shape[1] - len(jointsIdToReduce)
    allJoints = list(range(0, 24))
    preservedJoints = allJoints
    for id in jointsIdToReduce:
        preservedJoints.remove(id)

    oldJIdToNewJId = {oldJId: i for i, oldJId in enumerate(preservedJoints)}
    new_kintree_table = [[0,-1]]

    for pair in np.transpose(smpl_npModel.kintree_table):
        if oldJIdToNewJId.get(pair[0], -1) != -1 and oldJIdToNewJId.get(pair[1], -1) != -1:
            new_kintree_table.append([oldJIdToNewJId.get(pair[0], -1), oldJIdToNewJId.get(pair[1], -1)])

    new_kintree_table =np.array(new_kintree_table)

    RsS2F = smpl_npModel.R[preservedJoints]

    RsF2S = [np.linalg.inv(R) for R in RsS2F]
    deformedVerts = deformVerts(vTemplate, RsF2S, -trans, J, weights, new_kintree_table.transpose(), parent)

    fullBodyMesh = pv.PolyData(inOriginalRestPoseMesh)
    fullBodyMesh.points = deformedVerts

    fullBodyMesh.save('BodyMeshToSMPL.vtk')

    # with open('BodyMeshToSMPL.obj', 'w') as fp:
    #     for v in deformedVerts:

    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #
    #     for f in smplFaces + 1:
    #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))