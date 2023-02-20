from scipy.spatial.transform import Rotation as R
import numpy as np
from os.path import join
from pathlib import Path
import json
from . import Data

def readSkeletonData(skelDataFile):
    skelData = json.load(open(skelDataFile))
    vRestpose = (np.array(skelData['VTemplate']).transpose())[:, :3]
    J = (np.array(skelData['JointPos']).transpose())[:, :3]
    weights = np.array(skelData['Weights']).transpose()

    if 'PoseBlendShapes' in skelData:
        poseBlendShape = np.array(skelData['PoseBlendShapes'])
    else:
        poseBlendShape = None

    if 'Faces' in skelData:
        faces = np.array(skelData['Faces'])
    else:
        faces = None

    parent = skelData["Parents"]
    parent = {int(key[0]): key[1] for key in parent.items()}

    kintreeTable = np.array(skelData['KintreeTable']).transpose()

    return vRestpose, J, weights, poseBlendShape, kintreeTable, parent, faces

def quaternionsToRotations(qs):
    Rs = [R.from_quat([q[1], q[2], q[3], q[0]]) for q in qs]
    Rs = [r.as_matrix() for r in Rs]

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

def deformVerts(vRestpose, RMats, trans, J, weights, kintree_table, parent):
    numJoints = J.shape[0]
    # world transformation of each joint
    G = np.empty((numJoints, 4, 4))
    G[0] = with_zeros(np.hstack((RMats[0], J[0, :].reshape([3, 1]))))
    for i in range(1, numJoints):
        G[i] = G[parent[i]].dot(
            with_zeros(
                np.hstack(
                    [RMats[i], ((J[i, :] - J[parent[i], :]).reshape([3, 1]))]
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

def deformVertsWithJointTranslation(vRestpose, RMats, trans, J, weights, kintree_table, parent, returnTransf=False):
    numJoints = J.shape[0]
    # world transformation of each joint
    G = np.empty((numJoints, 4, 4))
    G[0] = with_zeros(np.hstack((RMats[0], J[0, :].reshape([3, 1]))))
    for i in range(1, numJoints):
        G[i] = G[parent[i]].dot(
            with_zeros(
                np.hstack(
                    [RMats[i], ((J[i, :] - J[parent[i], :] + trans[i, :]).reshape([3, 1]))]
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
    v = v + trans[0,:].reshape([1, 3])

    if returnTransf:
        return v, G
    else:
        return v

def transformJoints(R, trans, J, parent):
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