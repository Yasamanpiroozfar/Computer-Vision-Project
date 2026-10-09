import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap


def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-str', '--suffix_train',
                        default='.png',
                        help='only chose those images with a specific suffix')
    parser.add_argument('-ste', '--suffix_test',
                        default='.jpg',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--to_binary', action='store_true',
                       help='use OTSU binarization')

    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        # TODO 
        desc = computeDescs(files[i], True)
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def toBinary(mask):
    # test if not already binary
    if mask[mask==255].sum() != np.sum(mask):
        # maybe binary between 0,1?
        if mask[mask==1].sum() == mask.sum():
            mask *= 255
        else: # make it binary
           ret, mask = cv2.threshold(mask, 125, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    return mask

def computeDescs(fname, norm_hellinger=False, to_binary=False):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if to_binary:
        img = toBinary(img)
    sift = cv2.SIFT_create()
    keypoints=sift.detect(img, None)
    for kp in keypoints:
        kp.angle = 0
    
    keypoints, descriptors = sift.compute(img, keypoints)    
    print(len(keypoints), descriptors.shape if descriptors is not None else None)    

    if norm_hellinger:
        eps = 1e-12
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps) # axis 1: sum over columns, keepdims to keep the shape for broadcasting
        descriptors = np.sqrt(descriptors)
    
    return descriptors


def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=10000,
        verbose=1
    )
    kmeans.fit(descriptors)


    return kmeans.cluster_centers_.astype(np.float32)
def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors.astype(np.float32),
                               clusters.astype(np.float32),
                               k=1)

    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )

    for i, m in enumerate(matches):
        cluster_idx = m[0].trainIdx
        assignment[i, cluster_idx] = 1.0

    return assignment

def vlad(files, mus, powernorm):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        desc = computeDescs(f, True, True)
        a = assignments(desc, mus)
        
        T,D = desc.shape
        f_enc = np.zeros( (D*K), dtype=np.float32)
        for k in range(mus.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the 
            # difference to the cluster center than computing the differences
            # first and then select
#hinzugefügt von mir.
            assigned = desc[a[:, k] == 1]

            if len(assigned) > 0:
                residuals = assigned - mus[k]
                f_enc[k * D:(k + 1) * D] = residuals.sum(axis=0)
            
        # c) power normalization
        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        norm = np.linalg.norm(f_enc)
        if norm > 0:
            f_enc = f_enc / norm

        encodings.append(f_enc)

    return np.vstack(encodings)

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """


    # set up labels
    # TODO

    def loop(i):
        # compute SVM 
        # and make feature transformation
        x_pos = encs_test[i:i+1]

        X = np.vstack([x_pos, encs_train])

        y = np.zeros(len(X), dtype=np.int32)
        y[0] = 1
        y[1:] = -1

        svm = LinearSVC(C=C, class_weight='balanced', max_iter=10000)
        svm.fit(X, y)

        x = svm.coef_.astype(np.float32)

        x = normalize(x, norm='l2')

        return x

    # let's do that in parallel: 
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    new_encs = list(map(loop, tqdm(range(len(encs_test)))))
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    dists = 1.0 - np.dot(encs, encs.T)
    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42) # fix random seed
   
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix_train,
                                         args.labels_train)
    assert (len(files_train) == len(labels_train))
    print('#train: {}'.format(len(files_train)))
    print(files_train[0])
    descriptors = loadRandomDescriptors(files_train, 500000)
    print('> computed/loaded {} descriptors:'.format(len(descriptors)))
    # TODO loadRandomDescriptors
   
    if not os.path.exists('mus.pkl.gz'):

        # cluster centers
        print('> compute dictionary')
        mus = dictionary(descriptors, 100)
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

  
    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix_test,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        enc_test = vlad(files_test, mus, args.powernorm)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)
   
    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        enc_train = vlad(files_train, mus, args.powernorm)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    enc_test = esvm(enc_test, enc_train, args.C)

    # eval
    evaluate(enc_test, labels_test)
    print('> evaluate')
