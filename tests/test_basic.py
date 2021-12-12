import cmake_example as m
import numpy as np
import time
import scipy.sparse.linalg


def test_bbs():

    frame1 = np.loadtxt('data/kinect_paper/frame0.txt');
    frame50 = np.loadtxt('data/kinect_paper/frame50.txt');

    t = time.time();

    concat = np.concatenate([frame1, frame50], axis=1)

    mins = concat.min(axis=1)
    maxs = concat.max(axis=1)

    a = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.float64)

    bbs = m.BBS(mins[0], maxs[0], 40, mins[1], maxs[1], 40, 2)

    coloc = bbs.coloc(frame50[0,:], frame50[1,:])

    lambdas = np.ones([37, 37]) * 1e-5;

    bending = bbs.bending(lambdas)

    ctrlpts = bbs.getCtrlpts(coloc, bending, frame1[:2]);

    ev = bbs.eval(ctrlpts, frame50[0,:], frame50[1,:], 0,0)

    print(time.time() - t)


def test_warp():
    frame1 = np.loadtxt('data/kinect_paper/frame0.txt');
    frame50 = np.loadtxt('data/kinect_paper/frame50.txt');


    t = time.time();
    warp = m.Warp(frame1, [frame50], True)
    print(time.time() - t)

    print(warp.J21.a)



if __name__ == '__main__':
    test_bbs()
    test_warp()
