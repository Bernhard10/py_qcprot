from py_qcprot import rmsd as rmsd_qc
import unittest
import math
import numpy as np
from timeit import timeit

RAND_TEST_ITERATIONS = 20

def center_coords(coords):
    centroid = np.sum(coords, axis=0)
    centroid /= float(len(coords))
    #Center on centroids
    return coords - centroid

def rmsd_kabsch(coords1, coords2):
    #Center on centroids
    coords1 = center_coords(coords1)
    coords2 = center_coords(coords2)
    # Get the rotation matrix matrix
    correlation_matrix = np.dot(np.transpose(coords1), coords2)
    v, s, w_tr = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(v) * np.linalg.det(w_tr)) < 0.0
    if is_reflection:
        v[:, -1] = -v[:, -1]
    rotation_matrix = np.dot(v, w_tr)
    coords1_aligned = np.dot(coords1, rotation_matrix)
    diff_vecs = (coords2 - coords1_aligned)
    vec_lengths = np.sum(diff_vecs * diff_vecs, axis=1)
    return math.sqrt(sum(vec_lengths) / len(vec_lengths))

def get_random_coord_arrays(num_coords = 500):
    b1 = ((np.random.rand(num_coords, 3)-0.5)*300)
    b2 = (b1+(np.random.rand(num_coords, 3)-0.5)*20)
    return b1, b2

class TestRMSD(unittest.TestCase):
    def setUp(self):
        self.a1 = np.array([[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]], order="C")
        self.a2 = np.array([[2., 2., 2.], [0., 0., 0.], [-2., -2., -2.]], order="C") #RMSD of sqrt(2) to a1
        self.a3 = np.array([[2., 2., 1.], [1., 1., 0.], [0., 0., -1.]], order="F") #a1 only shifted (no rotation)
        self.a4 = np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]) #a1 rotated

    def test_rmsd_kabsch(self):
        self.assertAlmostEqual(rmsd_kabsch(self.a1, self.a1), 0)
        self.assertAlmostEqual(rmsd_kabsch(self.a1, self.a2), math.sqrt(2))
        self.assertAlmostEqual(rmsd_kabsch(self.a1, self.a3), 0)
        self.assertAlmostEqual(rmsd_kabsch(self.a1, self.a4), 0)

    def test_rmsd_qc(self):
        self.assertAlmostEqual(rmsd_qc(self.a1, self.a1), 0) #Special case, treated seperately in python wrapper
        self.assertAlmostEqual(rmsd_qc(self.a1, self.a2), math.sqrt(2))

    @unittest.expectedFailure #"Currently qcprot cannot handle the case RMSD==0"
    def test_rmsd_qc_2(self):
        self.assertAlmostEqual(rmsd_qc(self.a1, self.a3), 0)
        self.assertAlmostEqual(rmsd_qc(self.a1, self.a4), 0)
    def test_rmsd_bigger_dataset(self):
        for i in range(RAND_TEST_ITERATIONS):
            b1 , b2 = get_random_coord_arrays()
            self.assertAlmostEqual(rmsd_kabsch(b1, b2), rmsd_qc(b1, b2))            
    def test_rmsd_skipping_entries(self):
        for i in range(RAND_TEST_ITERATIONS):
            b1 , b2 = get_random_coord_arrays()
            self.assertAlmostEqual(rmsd_kabsch(b1[::2], b2[::2]), rmsd_qc(b1[::2], b2[::2]))
    def test_qcrmsd_centered(self):
        for i in range(RAND_TEST_ITERATIONS):
            b1 , b2 = get_random_coord_arrays()
            b1 = center_coords(b1)
            b2 = center_coords(b2)
            self.assertAlmostEqual(rmsd_kabsch(b1, b2), rmsd_qc(b1, b2, True))
    def test_handles_wrong_shapes_well(self):
        b1, b2 = get_random_coord_arrays(20)
        c1, c2 = get_random_coord_arrays(21)
        with self.assertRaises(ValueError):
            rmsd_qc(b1, c1)

class ProfileRMSD(unittest.TestCase):
    """
    This is not run by standard test-discovery with nose2.
    It can be run manually
    """
    def profile_rmsd(self):
        #Note: The numbers in the arrays are not the same between the two approaches, 
        #but it should make no difference after 10000 iterations (each time differeent coordinates are used)
        qc = timeit('rmsd_qc(b1, b2)', number=10000, setup = 'from test import get_random_coord_arrays; from py_qcprot import rmsd as rmsd_qc; b1, b2 = get_random_coord_arrays()')  
        kabsch = timeit('rmsd_kabsch(b1, b2)', number=10000, setup = 'from test import get_random_coord_arrays, rmsd_kabsch; b1, b2 = get_random_coord_arrays()')
        print("Kabsch {}, QCprot {}".format(kabsch, qc)) 
        self.assertLess(qc, kabsch) #Almost 10 times faster on my system
