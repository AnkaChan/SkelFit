import json
import glob
import subprocess
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

def readFittingErrs(inErrFolder):
    files = glob.glob(inErrFolder + r'\*.txt')
    errs = np.array([[0]])

    # iterRange =  tqdm(files) if _verbose else files
    for f in tqdm(files):
        errsF = np.loadtxt(f)
        errsF = errsF[np.where(errsF != 0.)[0]]
        errsF = errsF[:, np.newaxis]

        errs = np.vstack([errs, errsF])
    return errs

def readFittingErrsPerVerts(inErrFolder):
    files = glob.glob(inErrFolder + r'\*.txt')
    if len(files) == 0:
        return None
    errs0 = np.loadtxt(files[0])
    numVerts = errs0.shape[0]

    errs = np.zeros((numVerts, len(files)))

    # iterRange =  tqdm(files) if _verbose else files
    for i in tqdm(range(len(files))):
        f = files[i]
        errsF = np.loadtxt(f)

        errs[:, i] = errsF
    return errs

def errHistogram(errs, title = 'My Very Own Histogram'):
    n, bins, patches = plt.hist(x=errs, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85, log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


def prepareData(inJsonTargetPath, sequence = []):
    jFiles = glob.glob(inJsonTargetPath + '/*.json')

    if len(sequence) != 0:
        jFiles = jFiles[sequence[0]:sequence[1]]
        outBatchFile = inJsonTargetPath + '/BatchFile' + str(sequence[0]) + '_' + str(sequence[1]) + '.json'
    else:
        outBatchFile = inJsonTargetPath + '/BatchFile' + '.json'

    json.dump({"BatchFiles":jFiles}, open(outBatchFile, 'w'))

    return outBatchFile, jFiles

