#!/usr/bin/env python
# coding: utf-8
"""
This module provides the following:
    -- ``rankStat'' class. This is the main class for rank arrangements. All other functionalities are accessed through this class.
    -- computation of (convex) R-estimators using several algorithms. Namely:
    ---- ``WoA'' -- an exact algorithm relying on linear programming: https://arxiv.org/abs/1912.12750
    ---- ``IRLS'' -- a heuristic algorith with good asymptotic properties, can fail on rather smaller instances: https://epubs.siam.org/doi/10.1137/0611032 or https://doi.org/10.1007/978-3-642-57338-5
    ---- ``two_phase'' -- combination of the two previous algorithms: https://dx.doi.org/10.1007/978-3-030-62509-2_14
    -- computation of (general) R-estimators (https://doi.org/10.1080/02331934.2020.1812604) -- currently only indirectly as a byproduct of visualization
    -- visualisations of 2D rank arrangements with various filling (shading, contour lines) and coloring strategies. Various elements could be drawn (optima, progress of algorithms, normals on cells...).
    """

from copy import copy, deepcopy
from itertools import combinations
from subprocess import Popen, DEVNULL
from os import chdir, getcwd
from random import random
import numpy as np
import gurobipy as g
from numpy.linalg import lstsq
from scipy.special import ndtri
from rsIncEnu2 import rsIncEnu
# from enumeration import EnumerationProcess, IncEnu, mRS, FlIncEnu, IncEnu0
# from rsRS import rankArrRSEnu
from time import perf_counter

inequality_senses = {
    -1: g.GRB.LESS_EQUAL,
    0: g.GRB.EQUAL,
    1: g.GRB.GREATER_EQUAL
}
g.setParam('OutputFlag', 0)
g.setParam('Predual', 0)
""" Gurobi initializations and helper. """

colorvariations = {
    'rainbow': [[0, (1, 0, 0)], [1 / 6, (1, 1, 0)], [2 / 6, (0, 1, 0)],
                [3 / 6, (0, 1, 1)], [4 / 6, (0, 0, 1)], [5 / 6, (1, 0, 1)],
                [1, (1, 0, 0)]],
    'grayscale': [[0, (1, 1, 1)], [1, (0, 0, 0)]],
}
colorvariations['qrainbow'] = [[col[0]**2, col[1]]
                               for col in colorvariations['rainbow']]
"""
Predefined coloring strategies for visualisations. Structure of the inner 2-tuple: [normalized_F(beta)_value, 3-tuple_with_rgb_description of a color].
"""

ZERO = 10**-12
""" Number less than ZERO are considered to be 0. """


def hypind(i, j):
    """ Returns a tuple identifier for a hyperplane of ith and jth observation,
    regardless of ordering.
    """
    if i > j:
        return (j, i)
    else:
        return (i, j)


def strHypInd(i, j):
    """ Returns string identifier for a hyperplane of ith and jth observation,
    regardless of ordering.
    """
    if i > j:
        i, j = j, i
    return str(i) + '-' + str(j)


def random_matrix(n, d, r=(-50, 50)):
    """ Returns a numpy n times d array with range r, r defaults to (-50,50)
    """
    return np.random.randint(*r, (n, d))


def random_data(n, d, r=(-50, 50)):
    """ Returns a pair of numpy arrays, the first is n times d array, the
    second is n times 1 array. The entries are from range r, r defaults to
    (-50, 50).
    """
    return random_matrix(n, d, r), random_matrix(n, 1, r)


def generate_6_data(n, yrandom=False, generate_outliers=False):
    """ Returns a pair of numpy arrays with the same structure as in
    random_data. Here, a highly degenerate instance is created with
    an oaptimum known in advance: (1,2)^T.
    """
    X = []
    y = []
    for i in range(n):
        x1 = random() * 2 - 1
        x2 = random() * 2 - 1
        if yrandom:
            yr = random() + 0.5
        else:
            yr = 1
        X.append([x1, x2])
        X.append([x1, x2])
        X.append([x1, x2])
        X.append([-x1, -x2])
        X.append([-x1, -x2])
        X.append([-x1, -x2])
        y.append([x1 + 2 * x2])
        y.append([x1 + 2 * x2 - yr])
        y.append([x1 + 2 * x2 + yr])
        y.append([-x1 - 2 * x2])
        y.append([-x1 - 2 * x2 - yr])
        y.append([-x1 - 2 * x2 + yr])
    if generate_outliers:
        m = 100
        X.append([x1, x2])
        X.append([x1, x2])
        y.append([x1 + 2 * x2 + 1.5 * m])
        y.append([x1 + 2 * x2 - 0.5 * m])

    X = np.array(X)
    y = np.array(y)
    return X, y


class Hyperplane(object):
    """ Hyperplane object. Just for convenience.
    """
    def __init__(self, normal, rhs, indices, num):
        self.normal = normal
        self.rhs = rhs
        self.indices = indices
        self.num = num
        self.nlength = np.sqrt(self.normal @ self.normal)
        if self.nlength == 0:
            self.nnormal = np.array(self.normal)
            self.nrhs = rhs
        else:
            if rhs < 0:
                self.nnormal = -self.normal / self.nlength
                self.nrhs = -self.rhs / self.nlength
            else:
                self.nnormal = self.normal / self.nlength
                self.nrhs = self.rhs / self.nlength


class Vertex(object):
    def __init__(self, x, hinds):
        self.x = x
        self.obj = 0
        self.hyperplanes = hinds
        self.name = "-".join((strHypInd(*hind) for hind in hinds))
        self.cells = []
        self.is_global_min = False
        self.is_local_min = False


def generate_a(n, type_a='random'):
    """
    Generates n-tuple of weights according to type_a parameter.
    Accepts the following values of type_a:
    - random: random integer entries, uniformly distributed over (-50, 50).
    - sign: sign score function
    - wilcoxon: wilcoxon score function
    - vanderwaerden: van der waerden score function
    """
    if type_a == 'random':
        return random_matrix(n, 1)
    elif type_a == 'sign':
        return np.array([np.sign(i / (n + 1) - 0.5)
                         for i in range(1, n + 1)]).reshape(1, -1).T
    elif type_a == 'wilcoxon':
        return np.array([i / (n + 1) - 0.5
                         for i in range(1, n + 1)]).reshape(1, -1).T
    elif type_a == 'vanderwaerden':
        return np.array([ndtri(i / (n + 1))
                         for i in range(1, n + 1)]).reshape(1, -1).T
    else:
        raise NotImplementedError(
            'type of a "{}" not implemented'.format(type_a))


# coloring failed on

# In [12]: print(rs.X)
# [[ 19 -27]
# [ 45   8]
# [  1  22]
# [ 38 -48]
# [ 41  18]]

# In [13]: print(rs.y)
# [[ 42]
# [ 31]
# [-49]
# [-11]
# [-40]]


class rankStat(object):
    """
    Main object to work with an rank regression instance.
    Requires two params (n and d), determining the dimensions of the instance.
    """
    ZERO = ZERO

    def __init__(self,
                 n,
                 d,
                 X=None,
                 y=None,
                 conv=True,
                 a=None,
                 type_a='random',
                 constant_present=False,
                 zero=ZERO,
                 namesuffix=None,
                 enumeration_class=rsIncEnu,
                 duplicity_check=True,
                 zero_check=True):
        if X is None or y is None:
            X, y = random_data(n, d)

        if not a:
            a = generate_a(n, type_a)
        self.X = X
        self.y = y
        self.n = n
        self.d = d
        self.a = a
        if conv:
            self.a.sort(axis=0)
        self.ZERO = zero
        self.constant = constant_present
        self.namesuffix = namesuffix or ''
        self.name = "n{}-d{}".format(self.n, self.d) + self.namesuffix
        self.enumeration_class = enumeration_class

        self.hyperplanes = []
        self.hindices = {}
        self.cells = []

        self.hyperplanes_created = False
        self.vertices_created = False
        self.cells_created = False
        self.bounding_box_created = False
        self.generators_created = False
        self.minima_computed = False

        self.create_hyperplanes(zero_check=zero_check,
                                duplicity_check=duplicity_check)

        self.prepare_lp_improving_directions()

        self.WoA_iterations = None
        self.IRLS_iterations = None
        self.WoA_time_per_iteration = None
        self.IRLS_time_per_iteration = None
        self.global_minima = []
        self.local_minima = []

        self.borderline_point = self.compute_borderline_point()
        self.random_point = self.borderline_point * np.array(
            [[2 * (np.random.rand() - 0.5)] for i in range(self.d)])

        self.lstsq_beta = None

    def prepare_lp_improving_directions(self):
        self.lp_improving_directions = g.Model()
        self.lpid_l = self.lp_improving_directions.addMVar(self.d,
                                                           lb=-g.GRB.INFINITY,
                                                           obj=0.0)
        self.lpid_r = self.lp_improving_directions.addMVar(self.n,
                                                           lb=-g.GRB.INFINITY,
                                                           obj=0.0)
        self.lpid_s = self.lp_improving_directions.addMVar(self.n,
                                                           lb=-g.GRB.INFINITY,
                                                           obj=0.0)
        self.lp_improving_directions.update()

    def swap(self, i, j):
        self.a[i, 0], self.a[j, 0] = self.a[j, 0], self.a[i, 0]
        return self.a

    def create_hyperplanes(self, duplicity_check=True, zero_check=True):
        """ create hyperplanes fom observations """
        self.zero_hyperplanes = set()
        self.redundant_hyperplanes = set()
        self.N = {}
        num = 0
        if self.constant:
            start = 1
        else:
            start = 0
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                self.hyperplanes.append(
                    Hyperplane(self.X[i, start:] - self.X[j, start:],
                               self.y[i, start:] - self.y[j, start:],
                               hypind(i, j), num))
                self.hindices[hypind(i, j)] = self.hyperplanes[-1]
                num += 1
        if zero_check:
            self.check_hyperplanes_zero_normal()
        if duplicity_check:
            self.check_hyperplanes_duplicity()
        self.remove_unnecessary_hyperplanes()

        self.hyperplanes_created = True

    def remove_unnecessary_hyperplanes(self):
        for indices in self.zero_hyperplanes:
            self.hyperplanes.remove(self.hindices[indices])
        for indices in self.redundant_hyperplanes:
            self.hyperplanes.remove(self.hindices[indices])
        for hyperplane in self.hyperplanes:
            self.N[hyperplane.indices] = hyperplane.indices

    def check_observation_duplicity(self):
        pass

    def check_hyperplanes_zero_normal(self):
        for hyperplane in self.hyperplanes:
            if hyperplane.nlength <= self.ZERO:
                self.zero_hyperplanes.add(hyperplane.indices)

    def check_hyperplanes_duplicity(self):
        for (i, hyp1) in enumerate(self.hyperplanes[:-1]):
            if hyp1.indices in self.redundant_hyperplanes or hyp1.indices in self.zero_hyperplanes:
                continue
            for hyp2 in self.hyperplanes[i + 1:]:
                if hyp2.indices in self.zero_hyperplanes:
                    continue
                if all(abs(hyp1.nnormal - hyp2.nnormal) <= self.ZERO
                       ) and abs(hyp1.nrhs - hyp2.nrhs) <= self.ZERO:
                    self.redundant_hyperplanes.add(hyp2.indices)
                    self.N[hyp2.indices] = hyp1.indices

    def create_generators(self):
        self.gens = [[0 for i in range(self.d)] + [1]]
        self.gens += [
            h.normal.tolist() + (-1 * h.rhs).tolist() for h in self.hyperplanes
        ]

        self.generators_created = True

    def prepare_enumeration(self, output=False):
        self.enumeration = self.enumeration_class(generators=deepcopy(
            self.gens),
                                                  symmetry=True,
                                                  problem=self)
        self.enumeration.problem = self
        self.enumeration.OUTPUT = output

    def enumeration_run(self):
        self.enumeration.run()

    def residuals(self, beta):
        return self.y - self.X @ beta

    def permutation(self, residuals=None, beta=None):
        if residuals is None:
            if beta is None:
                raise RuntimeError('Not enough data to compute permutation.')
            else:
                residuals = self.residuals(beta)
        return np.lexsort(residuals.T)

    def inv_permutation(self, perm):
        iperm = np.zeros(self.n, int)
        for i, p in enumerate(perm):
            iperm[p] = i
        return iperm

    def group_residuals(self, beta=None, permutation=None):
        if beta is None:
            raise RuntimeError('Not enough data to compute groupping.')
        residuals = self.residuals(beta)
        permutation = self.permutation(residuals)
        groups = []
        groups.append([permutation[0]])
        for i in permutation[1:]:
            if abs(residuals[i] - residuals[groups[-1][-1]]) < self.ZERO:
                groups[-1].append(i)
            else:
                groups.append([i])
        return groups

    def get_pairs(self, beta):
        groups = self.group_residuals(beta=beta)
        if len(groups) < self.n:
            raise RuntimeError('Beta is not an interior point.')
        pairs = []
        for i in range(self.n - 1):
            pairs.append(self.hindices[hypind(groups[i][0],
                                              groups[i + 1][0])].num)
        return pairs

    def F(self, beta, residuals=None, perm=None):
        if residuals is None:
            r = self.residuals(beta)
        else:
            r = residuals
        if perm is None:
            p = self.permutation(r)
        else:
            p = perm
        return (self.a.T @ r[p])[0, 0]

    def find_locally_minimal_intersection_on_ray(self, start, direction):
        """ Currently evaluating F from scratch at every intersection.
        """
        best_obj = self.F(start)
        best_sol = None
        D = []
        for hyperplane in self.hyperplanes:
            denominator = hyperplane.normal @ direction
            if abs(denominator) <= self.ZERO:
                # print('too small denominator for {}'.format(direction))
                continue
            d = ((hyperplane.rhs - hyperplane.normal @ start) /
                 (denominator))[0]
            if d >= self.ZERO:
                hinds = hyperplane.indices
                D.append((d, hinds))
        if len(D) == 0:
            return None
        D.sort()
        # print(start, direction, D)
        for d in D:
            beta = start + d[0] * direction
            obj = self.F(beta)
            if obj + self.ZERO <= best_obj:
                best_sol = beta
                best_obj = obj
            elif obj - self.ZERO >= best_obj:
                break
        return best_sol

    def compute_borderline_point(self):
        alpha = 0
        direction = np.array([[np.random.rand() * 10] for i in range(self.d)])
        start = np.array([[0]] * self.d)
        # print(direction, start)
        for hyperplane in self.hyperplanes:
            divisor = (hyperplane.normal.T @ direction)[0]
            # print(divisor)
            if abs(divisor) > self.ZERO:
                current_alpha = (
                    (hyperplane.rhs - hyperplane.normal.T @ start) /
                    divisor)[0]
                if current_alpha >= self.ZERO:
                    if current_alpha > alpha:
                        alpha = current_alpha
        x = (alpha + 1) * direction
        return x

    def rayshooting(self, start, direction):
        """ Returns a tuple (hind, intersection). Computes the first hyperplane
        met by the ray from a given start point in a given direction. Hind
        stands for indices of the hyperplane, intersection are the
        coordinates of the intersection.
        """
        alpha = 0
        for hyperplane in self.hyperplanes:
            current_alpha = ((hyperplane.rhs - hyperplane.normal.T @ start) /
                             hyperplane.normal.T @ direction)[0, 0]
            if current_alpha >= self.ZERO:
                if current_alpha < alpha:
                    alpha = current_alpha
                    hind = hyperplane.indices
        if alpha > 0:
            x = start + alpha * direction
            return x, hind
        else:
            return None, None

    def create_vertices(self, check_duplicity=True):
        if not self.hyperplanes_created:
            self.create_hyperplanes()

        self.vertices = []
        self.bbmin = np.Inf * np.ones((self.d, 1))
        self.bbmax = -np.Inf * np.ones((self.d, 1))
        for hinds in combinations(self.hindices, self.d):
            if self.d == 2:
                if hinds[0][1] == hinds[1][0] or hinds[0][1] == hinds[1][1]:
                    continue
            x = intersection([self.hindices[hind] for hind in hinds],
                             self.ZERO)
            if x is not None:
                add = True
                if check_duplicity:
                    for vertex in self.vertices:
                        if all(abs(x - vertex.x) <= self.ZERO):
                            add = False
                            break
                if add:
                    self.vertices.append(Vertex(x, hinds))
                    self.vertices[-1].obj = self.F(self.vertices[-1].x)
                    for i in range(self.d):
                        if x[i, 0] < self.bbmin[i, 0]:
                            self.bbmin[i, 0] = x[i, 0]
                        if x[i, 0] > self.bbmax[i, 0]:
                            self.bbmax[i, 0] = x[i, 0]

        self.vertices_created = True

    def find_improving_direction(self, beta):
        m = self.lp_improving_directions
        m.remove(m.getConstrs())
        m.update()

        groups = self.group_residuals(beta=beta)
        m.addConstr(sum(self.lpid_r) + sum(self.lpid_s) == -1)

        starting_index = 0
        for group in groups:
            gsize = len(group)
            for i in range(starting_index, gsize + starting_index):
                for j in group:
                    m.addConstr(self.a[i] * self.X[j, :] @ self.lpid_l +
                                self.lpid_r[i] + self.lpid_s[j] >= 0)
            starting_index += gsize

        m.update()
        m.optimize()

        if m.status == 12:
            raise RuntimeError('Numerical difficulties in gurobi')
        elif m.status != 2:
            return None
        else:
            return self.lpid_l.X.reshape((self.d, 1))

    def IRLS(
        self,
        starting_point=None,
        use_lstsq_default=False,
        too_small_residual_factor=1e1,
        iteration_limit=1000,
        wo_improvement_limit=None,
    ):

        if wo_improvement_limit is None:
            wo_improvement_limit = self.n
        elif wo_improvement_limit == 0:
            wo_improvement_limit = iteration_limit
        if starting_point is None:
            if use_lstsq_default:
                if self.lstsq_beta is None:
                    self.least_squares()
            if self.lstsq_beta is None:
                beta = np.array([[0] for i in range(self.d)])
            else:
                beta = self.lstsq_beta
        else:
            beta = starting_point

        iterations = []

        ending_condition = False

        too_small_residual_factor = self.ZERO * too_small_residual_factor

        n_iterations = 0

        # iterations.append((beta, n_iterations, 'beta'))

        iteration_start_time = perf_counter()
        iteration_count = 0

        residuals = self.residuals(beta=beta)
        perm = self.permutation(residuals)
        inv_perm = self.inv_permutation(perm)

        best_F = self.F(beta, residuals=residuals, perm=perm)
        best_beta = beta
        iterations_wo_improvement = 0
        iteration_stop_time = iteration_start_time

        while not ending_condition:

            d = np.zeros(self.n)
            n_zero_res = 0
            n_too_small_res = 0
            for (i, (a, r)) in enumerate(zip(self.a[inv_perm], residuals)):
                if abs(r) < self.ZERO:
                    n_zero_res += 1
                    continue
                if abs(r) < too_small_residual_factor:
                    n_too_small_res += 1
                    r = too_small_residual_factor

                d[i] = a / r

            D = np.diag(d)
            A = self.X.T @ D @ self.X
            b = self.X.T @ D @ self.y

            U, sigma, V = np.linalg.svd(A)
            for i, s in enumerate(sigma):
                if abs(s) <= self.ZERO:
                    sigma[i] = 0
                else:
                    sigma[i] = 1 / s

            beta_new = V @ np.diag(sigma) @ V.T @ b
            # print(beta_new)

            diff = beta - beta_new
            if (diff.T @ diff)[0, 0] <= self.ZERO:
                ending_condition = True

            n_iterations += 1
            iterations.append((beta, beta_new, n_iterations, 'beta',
                               n_too_small_res, n_zero_res))

            beta = beta_new

            residuals = self.residuals(beta=beta)
            perm = self.permutation(residuals)
            inv_perm = self.inv_permutation(perm)

            Fvalue = self.F(beta, residuals=residuals, perm=perm)

            if Fvalue < best_F:
                best_F = Fvalue
                best_beta = beta
                iterations_wo_improvement = 0
            else:
                iterations_wo_improvement += 1

            if iterations_wo_improvement > wo_improvement_limit:
                ending_condition = True

            print(diff, diff.T @ diff)
            print(Fvalue, iterations_wo_improvement)
            print(len(iterations))

            if n_iterations >= iteration_limit:
                ending_condition = True

            iteration_stop_time = perf_counter()
            iteration_count += 1

        self.IRLS_iterations = iterations
        self.IRLS_time_per_iteration = (iteration_stop_time -
                                        iteration_start_time) / iteration_count

        self.IRLS_min = best_beta
        self.IRLS_Fvalue = best_F

    def WoA(self, starting_point=None, use_lstsq_default=False):
        if starting_point is None:
            if use_lstsq_default:
                if self.lstsq_beta is None:
                    self.least_squares()
            if self.lstsq_beta is None:
                beta = np.array([[0] for i in range(self.d)])
            else:
                beta = self.lstsq_beta
        else:
            beta = starting_point

        iterations = []
        iteration_count = 0
        iteration_start_time = perf_counter()
        iteration_stop_time = iteration_start_time

        while True:
            iterations.append([beta])
            iterations[-1].append('over cell')
            optfound, x, obj = self.maximize_over_cell(beta=beta)
            if not optfound:
                # print('Unbounded in cell {}'.format(
                # self.permutation(beta=beta)))
                break

            beta = x
            iterations[-1].append(beta)
            iterations[-1].append(obj)

            l = self.find_improving_direction(beta)
            if l is None:
                # print('Optimum found {}'.format(beta))
                break

            iterations.append([beta])
            iterations.append([beta])
            iterations[-1].append('improving direction')
            iterations[-2].append('ray shooting')
            iterations[-1].append(beta + l)

            # print(beta, l)

            beta0 = self.find_locally_minimal_intersection_on_ray(beta, l)
            if beta0 is None:
                del iterations[-2]
                # print('Unbounded ray {}'.format(l))
                break
            else:
                iterations[-2].append(beta0)
                beta = beta0
            iteration_count += 1
            iteration_stop_time = perf_counter()
            # print(iteration_count, beta, obj)

        if iteration_count > 0:
            self.WoA_time_per_iteration = (
                iteration_stop_time - iteration_start_time) / iteration_count
        else:
            self.WoA_time_per_iteration = 0

        self.WoA_iterations = iterations
        self.WoA_min = beta
        self.WoA_Fvalue = self.F(beta)

    def two_phase(self, starting_point=None, strategy=0):

        start_IRLS = perf_counter()

        if strategy == -1:
            self.IRLS_min = None
            pass

        elif strategy == 0:
            self.IRLS(starting_point=self.random_point,
                      iteration_limit=self.n,
                      wo_improvement_limit=self.d)

        elif strategy == 1:
            self.IRLS(starting_point=self.random_point,
                      iteration_limit=self.n * self.d,
                      wo_improvement_limit=self.n)

        elif strategy == 2:
            self.IRLS(starting_point=self.random_point,
                      iteration_limit=self.n * self.d,
                      wo_improvement_limit=np.log(self.n) * self.d)

        elif strategy == 3:
            self.IRLS(starting_point=self.borderline_point,
                      iteration_limit=self.n,
                      wo_improvement_limit=self.d)

        elif strategy == 4:
            self.IRLS(starting_point=self.borderline_point,
                      iteration_limit=self.n * self.d,
                      wo_improvement_limit=self.n)

        elif strategy == 5:
            self.IRLS(starting_point=self.borderline_point,
                      iteration_limit=self.n * self.d,
                      wo_improvement_limit=np.log(self.n) * self.d)

        stop_IRLS = perf_counter()
        if self.IRLS_min is not None:
            self.WoA(starting_point=self.IRLS_min)
        else:
            self.WoA(starting_point=self.random_point)
            self.IRLS_min = self.WoA_min
            self.IRLS_Fvalue = self.WoA_Fvalue
            self.IRLS_iterations = []
        stop_WoA = perf_counter()

        return start_IRLS, stop_IRLS, stop_WoA

    def remove_outlier_vertices(self):
        if self.d != 2:
            raise NotImplementedError('Implemented only for d = 2')

        x = np.array([vertex.x[0, 0] for vertex in self.vertices])
        y = np.array([vertex.x[1, 0] for vertex in self.vertices])

        repeat = True
        while repeat:
            sd = np.std(x)
            m = np.mean(x)
            l1 = len(x)
            x = x[abs(x - m) <= 3 * sd]
            if len(x) == l1:
                repeat = False

        repeat = True
        while repeat:
            sd = np.std(y)
            m = np.mean(y)
            l2 = len(y)
            y = y[abs(y - m) <= 3 * sd]
            if len(y) == l2:
                repeat = False

        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        self.vertices = [
            vertex for vertex in self.vertices
            if vertex.x[0, 0] >= xmin and vertex.x[0, 0] <= xmax
            and vertex.x[1, 0] >= ymin and vertex.x[1, 0] <= ymax
        ]

        xmin, xmax = min(vertex.x[0, 0] for vertex in self.vertices), max(
            vertex.x[0, 0] for vertex in self.vertices)
        ymin, ymax = min(vertex.x[1, 0] for vertex in self.vertices), max(
            vertex.x[1, 0] for vertex in self.vertices)
        self.span = max(xmax - xmin, ymax - ymin)
        blowing = self.span / 333.75146872
        self.bbmax = np.array([[xmax], [ymax]]) + blowing
        self.bbmin = np.array([[xmin], [ymin]]) - blowing

    def compute_minima(self):
        for vertex in self.vertices:
            if abs(vertex.obj - self.objmin) <= self.ZERO:
                vertex.is_global_min = True
                vertex.is_local_min = True
                self.global_minima.append(vertex)
                self.local_minima.append(vertex)
            else:
                for cell in vertex.cells:
                    if vertex.obj - cell.minobj >= self.ZERO:
                        break
                else:
                    vertex.is_local_min = True
                    self.local_minima.append(vertex)
        self.minima_computed = True

    def create_bounding_box(self,
                            remove_outliers=True,
                            custom_bb=None,
                            custom_ovr=None):
        if not self.vertices_created:
            self.create_vertices()

        if len(self.vertices) == 0:
            raise RuntimeError('No vertices')

        if self.d != 2:
            return False

        if custom_bb is None:
            if remove_outliers:
                self.remove_outlier_vertices()

            # Blowing is not very useful with removed outliers
            if not remove_outliers:
                self.span = max(self.bbmax - self.bbmin)[0]
                blowing = self.span / 10
                self.bbmax += blowing
                self.bbmin -= blowing
                self.span *= 1.2

        else:
            xmin, ymin = custom_bb[0][0, 0], custom_bb[0][1, 0]
            xmax, ymax = custom_bb[1][0, 0], custom_bb[1][1, 0]
            self.vertices = [
                vertex for vertex in self.vertices
                if vertex.x[0, 0] >= xmin and vertex.x[0, 0] <= xmax
                and vertex.x[1, 0] >= ymin and vertex.x[1, 0] <= ymax
            ]
            self.bbmax = custom_bb[1]
            self.bbmin = custom_bb[0]
            self.span = max(xmax - xmin, ymax - ymin)

        self.vertices.append(
            Vertex(np.array([[self.bbmax[0, 0]], [self.bbmax[1, 0]]]),
                   [('max', '0'), ('max', '1')]))
        self.vertices[-1].obj = self.F(self.vertices[-1].x)
        self.vertices.append(
            Vertex(np.array([[self.bbmax[0, 0]], [self.bbmin[1, 0]]]),
                   [('max', '0'), ('min', '1')]))
        self.vertices[-1].obj = self.F(self.vertices[-1].x)
        self.vertices.append(
            Vertex(np.array([[self.bbmin[0, 0]], [self.bbmax[1, 0]]]),
                   [('min', '0'), ('max', '1')]))
        self.vertices[-1].obj = self.F(self.vertices[-1].x)
        self.vertices.append(
            Vertex(np.array([[self.bbmin[0, 0]], [self.bbmin[1, 0]]]),
                   [('min', '0'), ('min', '1')]))
        self.vertices[-1].obj = self.F(self.vertices[-1].x)

        for hind in self.hindices:
            hyperplane = self.hindices[hind]
            hyperplane.endpoints = []

            x = intersection([
                hyperplane,
                Hyperplane(np.array([1, 0]), np.array([self.bbmax[0, 0]]),
                           'pom-max-0', 0)
            ], self.ZERO)
            if x is not None:
                if all(x - self.ZERO <= self.bbmax) and all(
                        x + self.ZERO >= self.bbmin):
                    self.vertices.append(Vertex(x, [hind, ('max', '0')]))
                    self.vertices[-1].obj = self.F(self.vertices[-1].x)
                    hyperplane.endpoints.append(x)

            x = intersection([
                hyperplane,
                Hyperplane(np.array([1, 0]), np.array([self.bbmin[0, 0]]),
                           'pom-min-0', 0)
            ], self.ZERO)
            if x is not None:
                if all(x - self.ZERO <= self.bbmax) and all(
                        x + self.ZERO >= self.bbmin):
                    self.vertices.append(Vertex(x, [hind, ('min', '0')]))
                    self.vertices[-1].obj = self.F(self.vertices[-1].x)
                    hyperplane.endpoints.append(x)

            x = intersection([
                hyperplane,
                Hyperplane(np.array([0, 1]), np.array([self.bbmax[1, 0]]),
                           'pom-max-1', 0)
            ], self.ZERO)
            if x is not None:
                if all(x - self.ZERO <= self.bbmax) and all(
                        x + self.ZERO >= self.bbmin):
                    self.vertices.append(Vertex(x, [hind, ('max', '1')]))
                    self.vertices[-1].obj = self.F(self.vertices[-1].x)
                    hyperplane.endpoints.append(x)

            x = intersection([
                hyperplane,
                Hyperplane(np.array([0, 1]), np.array([self.bbmin[1, 0]]),
                           'pom-min-1', 0)
            ], self.ZERO)
            if x is not None:
                if all(x - self.ZERO <= self.bbmax) and all(
                        x + self.ZERO >= self.bbmin):
                    self.vertices.append(Vertex(x, [hind, ('min', '1')]))
                    self.vertices[-1].obj = self.F(self.vertices[-1].x)
                    hyperplane.endpoints.append(x)

        for v in self.vertices:
            if abs(v.obj) < self.ZERO:
                v.obj = 0

        if custom_ovr:
            self.objmin, self.objmax = custom_ovr[0], custom_ovr[1]
        else:
            self.objmin, self.objmax = min(v.obj for v in self.vertices), max(
                v.obj for v in self.vertices)

    def reset_objects(self):
        self.vertices_created = False
        self.generators_created = False
        self.bounding_box_created = False
        self.cells_created = False
        self.hyperplanes_created = False
        self.minima_computed = False
        self.hyperplanes = []
        self.hindices = {}
        self.vertices = []
        self.gens = []
        self.cells = []
        self.global_minima = []
        self.local_minima = []
        self.create_hyperplanes()

    def least_squares(self):
        beta, rsum, rank, sv = lstsq(self.X, self.y, rcond=None)
        self.lstsq_beta = beta

    def visualize(self,
                  createpdf=True,
                  wait=True,
                  cells=True,
                  reset=True,
                  colorvariation='rainbow',
                  run_WoA=False,
                  WoA_start=None,
                  bb=None,
                  target_size=40,
                  density_contours=2.5,
                  cell_filling_mode='shading',
                  visualizationsuffix='',
                  cell_descriptions=False,
                  obj_value_range=None):
        if reset:
            self.reset_objects()

        if not self.vertices_created:
            print("Creating vertices")
            self.create_vertices()
            print("Vertices created")

        if not self.bounding_box_created:
            print("Creating BB")
            self.create_bounding_box(custom_bb=bb, custom_ovr=obj_value_range)
            print("BB created")

        if cells:
            print("Creating cells")
            self.create_cells(output=True)
            print("Cells created")

        if not self.minima_computed:
            print("Computing minima")
            self.compute_minima()
            print("Minima computed")

        if colorvariation in colorvariations:
            colarray = dict(colors=colorvariations[colorvariation])
        else:
            colarray = {}

        if run_WoA:
            self.WoA(starting_point=WoA_start)

        cwd = getcwd()  # store current directory
        chdir('./visualize')  # change directory

        filename = "vis" + self.name + visualizationsuffix + '.tex'
        o = open(filename, 'w')
        o.write(open('./header.tex', 'r').read())

        scalefactor = target_size / self.span

        lines = []

        if cell_filling_mode == 'shading' or cell_filling_mode == 'shading and contours':
            compute_shadings = True
        else:
            compute_shadings = False

        if cell_filling_mode == 'shading and contours' or cell_filling_mode == 'colored contours':
            compute_contours = True
            num_contours = int(target_size * density_contours)
            if cell_filling_mode == 'colored contours':
                colored_contours = True
            else:
                colored_contours = False
        else:
            compute_contours = False

        # preparing contours
        if compute_contours:

            # distance between two contours measured by objective value
            covs = (self.objmax - self.objmin) / (num_contours + 1)

            # prepare colors
            lines.append(r'\colorlet{concolor-0}{gray}')
            if colored_contours:
                for i in range(1, num_contours + 1):
                    color = find_color(i / (num_contours + 1), **colarray)
                    lines.append(r'\definecolor{concolor-' + str(i) +
                                 '}{rgb}{' + '{:.6f},{:.6f},{:.6f}'.format(
                                     color[0], color[1], color[2]) + '}')
        lines.append(
            r'\begin{{tikzpicture}}[scale={:.6f}]'.format(scalefactor))

        # draw bounding box
        lines.append(
            r'\path[bb,use as bounding box] ({:.6f},{:.6f}) rectangle ({:.6f},{:.6f});'
            .format(self.bbmin[0, 0], self.bbmin[1, 0], self.bbmax[0, 0],
                    self.bbmax[1, 0]))

        # draw cells

        if len(self.cells) > 0:
            for i, c in enumerate(self.cells):
                # print(self.objmin,self.objmax,c.minobj,c.maxobj)

                shading = ''
                cell_contours = ''

                # compute shading
                if compute_shadings:

                    colors, line = define_color_transition(
                        self.objmin, self.objmax, c.minobj, c.maxobj,
                        'sh' + str(c.name), **colarray)
                    lines.append(line)
                    shading = r'\path [shading=sh{}] ({},{}) rectangle ({},{});'.format(
                        c.name, c.minbb[0, 0], c.minbb[1, 0], c.maxbb[0, 0],
                        c.maxbb[1, 0])

                # compute contours
                if compute_contours:
                    # contours
                    cont_cell_min_num = int(
                        (c.minobj - self.objmin) // covs) + 1
                    cont_cell_max_num = num_contours - int(
                        (self.objmax - c.maxobj) // covs)

                    # print(covs, c.minobj, self.objmin, c.maxobj, self.objmax,
                    # cont_cell_max_num, cont_cell_min_num)

                    if cont_cell_max_num < cont_cell_min_num:
                        cell_contours = ""
                    else:
                        if not colored_contours:
                            contours = [
                                add_contour(
                                    c.minbb, c.maxbb,
                                    ((self.objmin + i * covs) - c.minobj) /
                                    (c.maxobj - c.minobj)) for i in
                                range(cont_cell_min_num, cont_cell_max_num + 1)
                            ]
                        else:
                            contours = [
                                add_contour(
                                    c.minbb, c.maxbb,
                                    ((self.objmin + i * covs) - c.minobj) /
                                    (c.maxobj - c.minobj), i) for i in
                                range(cont_cell_min_num, cont_cell_max_num + 1)
                            ]

                        cell_contours = "\n".join(contours)

                # print(colors)
                clipping = r"\path[clip] (" + ")--(".join(
                    "{},{}".format(col[0], col[1])
                    for col in c.rot_vertices.T) + ") -- cycle;"
                line = r"\node[rotate=-{}] at ({},{}) {{\tikz[scale={:.6f}] {{{}{}{}}} }};".format(
                    c.degangle, c.refpoint[0, 0], c.refpoint[1, 0],
                    scalefactor, clipping, shading, cell_contours)
                lines.append(line)
                if cell_descriptions:
                    lines.append(
                        r"\node[font=\tiny] at ({:.6f},{:.6f}) {{{} {}}};".
                        format(c.m[0, 0], c.m[1, 0], i, int(c.lpminobj)))

                # draw line to minimum
                # lines.append(
                # r"\draw[line width=0.2pt,dotted,gray] ({:.6f},{:.6f}) -- ({:.6f},{:.6f});"
                # .format(c.m[0, 0], c.m[1, 0], c.lpminx[0, 0], c.lpminx[1,
                # 0]))

                # draw normal
                # lines.append(
                # r"\draw[line width=0.2pt,dashed,gray] ({:.6f},{:.6f}) -- +({:.6f},{:.6f});"
                # .format(c.lpminx[0, 0], c.lpminx[1, 0], c.Fnormal[0],
                # c.Fnormal[1]))

        # if len(self.cells) > 0:
        # for c in self.cells:
        # line = r"\path[draw=green] (" + ")--(".join("{},{}".format(vertex.x[0,0],vertex.x[1,0]) for vertex in c.vertices) + ") -- cycle;"
        # lines.append(line)

        # draw hyperplanes
        for h in self.hyperplanes:
            if len(h.endpoints) == 2:
                lines.append(
                    r'\path[hyperplane] ({:.6f},{:.6f}) -- ({:.6f},{:.6f}) node[hyplabel] {{{}}};'
                    .format(h.endpoints[0][0, 0], h.endpoints[0][1, 0],
                            h.endpoints[1][0, 0], h.endpoints[1][1, 0],
                            strHypInd(*h.indices)))
            if len(h.endpoints) == 1:
                raise ValueError(
                    'One endpoint only! This should never happen!')

        # draw vertices

        for v in self.vertices:
            lines.append(r'\coordinate ({}) at ({:.6f},{:.6f});'.format(
                v.name, v.x[0, 0], v.x[1, 0]))
            # lines.append(r'\node[vertex] at ({}) {{{:.0f}}};'.format(
            # v.name, v.obj))

        # draw WoA iterations
        if self.WoA_iterations:
            for iteration in self.WoA_iterations:
                lines.append(
                    r'\draw [{}] ({:.6f},{:.6f}) -- ({:.6f},{:.6f});'.format(
                        iteration[1], iteration[0][0, 0], iteration[0][1, 0],
                        iteration[2][0, 0], iteration[2][1, 0]))

        # draw IRLS iterations
        if self.IRLS_iterations:
            for iteration in self.IRLS_iterations[:20]:
                lines.append(
                    r'\draw [IRLS iteration] ({:.6f},{:.6f}) -- ({:.6f},{:.6f});'
                    .format(iteration[0][0, 0], iteration[0][1, 0],
                            iteration[1][0, 0], iteration[1][1, 0]))

        # draw minima
        if self.local_minima:
            for minimum in self.local_minima:
                lines.append(
                    r'\node [local minimum] at ({:.6f},{:.6f}) {{}};'.format(
                        minimum.x[0, 0], minimum.x[1, 0]))
        if self.global_minima:
            for minimum in self.global_minima:
                lines.append(
                    r'\node [minimum] at ({:.6f},{:.6f}) {{}};'.format(
                        minimum.x[0, 0], minimum.x[1, 0]))

        o.write('\n'.join(lines))

        o.write(open('./footer.tex', 'r').read())
        o.close()

        if createpdf:
            process = Popen(['lualatex', '-interaction=nonstopmode', filename],
                            stdout=DEVNULL)
            if wait:
                process.wait()

        chdir(cwd)  # go back to previous directoroy

    def create_one_cell(self, signs):
        cell = Cell(signs)
        cell.name = "-".join([str(sign[0]) for sign in signs])

        for vertex in self.vertices:
            if len(signs) != len(self.hyperplanes):
                raise ValueError(
                    'Number of signs does not match number of hyperplanes')
            add = True
            for (sign, hind) in signs:
                hyperplane = self.hindices[hind]
                if sign * (hyperplane.normal
                           @ vertex.x)[0] + self.ZERO < sign * hyperplane.rhs:
                    add = False
                    break
            if add:
                cell.vertices.append(vertex)
                vertex.cells.append(cell)

        if len(cell.vertices) > 0:
            cell.m = sum(vertex.x
                         for vertex in cell.vertices) / len(cell.vertices)
            m = cell.m

            cell.vertices.sort(key=lambda vertex: np.arctan2(
                (vertex.x - m)[0, 0], (vertex.x - m)[1, 0]))

            self.cells.append(cell)
            # print(len(self.cells))

            cell.p = self.permutation(beta=m)
            # print(m,p,self.residuals(m))
            cell.Fnormal = np.sum(-self.a * self.X[cell.p], axis=0)
            cell.angle = np.arctan2(cell.Fnormal[0], cell.Fnormal[1])

            cell.maxobj = max(vertex.obj for vertex in cell.vertices)
            cell.minobj = min(vertex.obj for vertex in cell.vertices)

            cell.lpopt, cell.lpminx, cell.lpminobj = self.maximize_over_cell(
                cell.p)

            cell.rot_vertices = rot_matrix(cell.angle) @ np.column_stack(
                [vertex.x for vertex in cell.vertices])
            cell.minbb, cell.maxbb = np.min(cell.rot_vertices, axis=1).reshape(
                1, -1).T, np.max(cell.rot_vertices, axis=1).reshape(1, -1).T
            cell.refpoint = rot_matrix(cell.angle).T @ (
                (cell.minbb + cell.maxbb) / 2)
            cell.degangle = cell.angle / np.pi * 180

    def create_cells(self, output=True):
        if not self.vertices_created:
            self.create_vertices()

        if not self.generators_created:
            self.create_generators()

        self.prepare_enumeration(output=output)

        self.cells = []
        self.enumeration_run()

        self.cells_created = True

    def maximize_over_cell(self, permutation=None, beta=None):
        """ Returns maximum over a cell given by a permutation or by a point.
        If permutation is not given, it is computed from beta. No check for
        beta being interior is performed.
        """
        if permutation is None:
            if beta is None:
                raise RuntimeError('A cell must be given')
            permutation = self.permutation(beta=beta)

        constant = (self.a.T @ self.y[permutation])[0, 0]
        Fnormal = np.sum(-self.a * self.X[permutation], axis=0)
        # print(Fnormal)

        m = g.Model()
        v = []
        for i in range(self.d):
            v.append(m.addVar(lb=-g.GRB.INFINITY, obj=Fnormal[i]))
        m.update()

        for i in range(self.n - 1):
            i1, i2 = permutation[i], permutation[i + 1]
            hyperplane = self.hindices[hypind(i1, i2)]
            if i1 < i2:
                sense = 1
            else:
                sense = -1
            m.addConstr(g.LinExpr(hyperplane.normal, v),
                        inequality_senses[sense], hyperplane.rhs)

        m.optimize()
        if m.status == 2:
            optfound = True
            x = np.array([vi.x for vi in v]).reshape(self.d, 1)
            obj = m.getObjective().getValue() + constant
            # print(obj, self.F(x))
        else:
            optfound = False
            x = None
            obj = None
        del m
        return optfound, x, obj


def add_contour(minbb, maxbb, ratio, contour_num=0):
    return r"\path[contour,draw=concolor-{3}] ({0},{2}) -- ({1},{2});".format(
        minbb[0, 0], maxbb[0, 0],
        minbb[1, 0] + ratio * (maxbb[1, 0] - minbb[1, 0]), contour_num)


def find_color(y, colors):
    # print(y, colors)
    last = colors[0]
    for col in colors[1:]:
        if y >= last[0] and y <= col[0]:
            coeff = (y - last[0]) / (col[0] - last[0])
            return tuple(last[1][i] + coeff * (col[1][i] - last[1][i])
                         for i in range(3))
        last = col
    raise ValueError('y falls to no interval')


def define_color_transition(gmin,
                            gmax,
                            lmin,
                            lmax,
                            name='rainbow',
                            colors=None):
    if not colors:
        colors = [[0, (1, 0, 0)], [1 / 6, (1, 1, 0)], [2 / 6, (0, 1, 0)],
                  [3 / 6, (0, 1, 1)], [4 / 6, (0, 0, 1)], [5 / 6, (1, 0, 1)],
                  [1, (1, 0, 0)]]

    gspan = gmax - gmin
    rmin, rmax = (lmin - gmin) / gspan, (lmax - gmin) / gspan
    rspan = rmax - rmin
    if rspan == 0:
        col = find_color(rmin, colors)
        rcolors = [[0, col], [1, col]]
    else:
        cmin = find_color(rmin, colors)
        cmax = find_color(rmax, colors)
        rcolors = [[(col[0] - rmin) / rspan, col[1]] for col in colors]
        rcolors = [col for col in rcolors if col[0] > 0 and col[0] < 1]
        rcolors = [[0, cmin]] + rcolors + [[1, cmax]]
    for c in rcolors:
        c[0] = c[0] / 2 + 0.25
    rcolors = [[0, rcolors[0][1]]] + rcolors + [[1, rcolors[-1][1]]]

    pgfshading = r"\pgfdeclareverticalshading{{{}}}{{100bp}}{{".format(
        name) + ";".join("rgb({:.6f}bp)=({:.6f},{:.6f},{:.6f})".format(
            100 * c[0], c[1][0], c[1][1], c[1][2]) for c in rcolors) + "}"
    return rcolors, pgfshading


def rot_matrix(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]])


class Cell(object):
    def __init__(self, signs):
        self.signs = signs
        self.vertices = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def intersection(hyperplanes, ZERO=ZERO):
    if len(hyperplanes) != 2:
        raise NotImplementedError('Intersections implemented only for d = 2.')
    h0 = hyperplanes[0]
    h1 = hyperplanes[1]

    d = h0.normal[0] * h1.normal[1] - h0.normal[1] * h1.normal[0]
    if abs(d) < ZERO:
        return None

    d1 = h0.rhs[0] * h1.normal[1] - h0.normal[1] * h1.rhs[0]
    d2 = h0.normal[0] * h1.rhs[0] - h0.rhs[0] * h1.normal[0]
    return np.array([[d1 / d], [d2 / d]])


if __name__ == "__main__":
    pass


def tryit():
    global s
    global r
    data52 = random_data(8, 2)
    r = rankStat(8, 2, X=data52[0], y=data52[1])
    s = rankStat(8, 2, X=data52[0], y=data52[1], enumeration_class=rsIncEnu)
    r.visualize()
    print('waiting')
    time.sleep(3)
    s.visualize()
    s.enumeration.print_stats()
    r.enumeration.print_stats()
