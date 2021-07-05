import gurobipy as gp
import numpy as np
import time
import random
from copy import deepcopy

inequality_senses = {
    -1: gp.GRB.LESS_EQUAL,
    0: gp.GRB.EQUAL,
    1: gp.GRB.GREATER_EQUAL
}
gp.setParam('OutputFlag', 0)
gp.setParam('Predual', 0)


def hypind(a1, a2):
    if a1 < a2:
        return (a1, a2)
    else:
        return (a2, a1)


class rsIncEnu(object):
    """
    Class responsible for enumeration of cells of an arrangement of hyperplanes that arises from rank estimators.
    It implements algorithm RSIncEnu from paper https://doi.org/10.1080/02331934.2020.1812604.

    An instance is created from a rankStat instance, passed as variable ``problem'' to the constructor.
    The following methods are intended to be used:
    -- run -- it actually runs the enumeration,
    -- print_stats -- human-readable format,
    -- return_stats -- one line in csv format.
    """

    ZERO = 10**-12
    name = "IncEnu for rank arrangements"

    def __init__(self, problem=None, output=True, **kwargs):
        if not problem:
            raise ValueError('no problem given, there is nothing to do')
        self.problem = problem
        self.ZERO = problem.ZERO
        self.N = self.problem.N
        self.R = self.problem.redundant_hyperplanes
        self.Z = self.problem.zero_hyperplanes
        self.d = self.problem.d + 1
        self.m = len(self.problem.hyperplanes)
        self.OUTPUT = output
        self.cell_number = 0
        self.fixable = True
        self.create_generators()

        self.flp = 0
        self.ilp = 0

        self.prepare_solver()

        self.compute_initial_beta()

    def create_generators(self):
        self.symmetric_gen = [0] * (self.d - 1) + [1]
        self.generators = {}
        for hyperplane in self.problem.hyperplanes:
            self.generators[hyperplane.indices] = hyperplane.normal.tolist(
            ) + (-1 * hyperplane.rhs).tolist()

    def compute_inverse_permutation(self, perm):
        iperm = [0] * self.problem.n
        for i, p in enumerate(perm):
            iperm[p] = i
        return iperm

    def update_constraints(self, perm):
        unord_pairs = []
        ord_pairs = []
        iperm = self.compute_inverse_permutation(perm)

        for i in range(self.problem.n - 1):
            ord_pairs.append(hypind(perm[i], perm[i + 1]))
            unord_pairs.append((perm[i], perm[i + 1]))
        for constraint in list(self.constraints.keys()):
            if not hypind(*constraint) in ord_pairs:
                self.model.remove(self.constraints[constraint])
                del self.constraints[hypind(*constraint)]
        self.model.update()
        used_pairs = set()
        for pair in unord_pairs:
            if hypind(*pair) in self.R:
                pair = self.N[hypind(*pair)]
                if iperm[pair[1]] < iperm[pair[0]]:
                    pair = (pair[1], pair[0])
            if hypind(*pair) in self.Z:
                continue
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            constraint = self.constraints.get(hypind(*pair))
            if constraint:
                if constraint.rhs == 1 and pair[0] > pair[1]:
                    constraint.rhs = -1
                    constraint.sense = inequality_senses[-1]
                elif constraint.rhs == -1 and pair[0] < pair[1]:
                    constraint.rhs = 1
                    constraint.sense = inequality_senses[1]
            else:
                if pair[0] > pair[1]:
                    sense = -1
                else:
                    sense = 1
                self.constraints[hypind(*pair)] = self.model.addConstr(
                    self.le[hypind(*pair)], inequality_senses[sense], sense)
        self.model.update()

    def fix_constraint(self, fixed):
        if not self.fixable:
            raise ValueError("A constraint already fixed!")

        self.fixable = False

        if not fixed in self.constraints:
            raise ValueError("{} not in list of constraints!".format(fixed))

        self.fixed_constraint = self.constraints[fixed]

        self.fix_sense, self.fix_rhs = self.fixed_constraint.sense, self.fixed_constraint.rhs

        self.fixed_constraint.sense = inequality_senses[0]
        self.fixed_constraint.rhs = 0

    def release_constraint(self):
        if self.fixable:
            raise ValueError("Nothing to be released")

        self.fixable = True
        self.fixed_constraint.sense = self.fix_sense
        self.fixed_constraint.rhs = self.fix_rhs

    def solve(self, perm, fixed=None):
        self.update_constraints(perm)
        if fixed:
            self.fix_constraint(fixed)
        self.model.optimize()

        if self.model.status < 2 or self.model.status > 3:
            self.model.write('model.lp')
            print("warning: model unbounded")

        if self.model.status == 2 or self.model.status == 5:
            self.flp += 1
            self.last_beta = self.get_solution()
            self.last_feasible = True
            # self.model.write('pom'+str(time.time())+'.lp')

        elif self.model.status == 3:
            self.ilp += 1
            self.last_feasible = False

        else:
            raise RuntimeError("Bad status {}".format(self.model.status))

        if fixed:
            self.release_constraint()

    def compute_initial_beta(self):
        beta = [1 + random.random() for i in range(self.d - 1)] + [1]
        perm = self.recompute_permutation(beta)

        self.solve(perm)

        if self.last_feasible:
            self.initial_beta = self.last_beta

    def new_beta(self, old, intersection):
        direction = [ii - oi for (oi, ii) in zip(old, intersection)]

        alphas = []

        for gen in self.generators.values():
            spo = sum(oi * gi for (oi, gi) in zip(old, gen))
            spd = sum(di * gi for (di, gi) in zip(direction, gen))
            if abs(spd) < self.ZERO:
                continue
            if -spo / spd > self.ZERO:
                alphas.append(-spo / spd)

        spo = sum(oi * gi for (oi, gi) in zip(old, self.symmetric_gen))
        spd = sum(di * gi for (di, gi) in zip(direction, self.symmetric_gen))
        if abs(spd) < self.ZERO:
            pass
        elif -spo / spd > self.ZERO:
            alphas.append(-spo / spd)

        alphas.sort()
        if len(alphas) == 1:
            beta = [
                2 * alphas[0] * di + oi for (oi, di) in zip(old, direction)
            ]
        elif len(alphas) > 1:
            alpha = (alphas[0] + alphas[1]) / 2
            beta = [alpha * di + oi for (oi, di) in zip(old, direction)]
        return beta

    def prepare_solver(self):
        self.model = gp.Model()

        for i in range(self.d):
            self.model.addVar(lb=-gp.GRB.INFINITY, obj=0.0, name=str(i))

        self.model.update()
        v = self.model.getVars()

        self.model.addConstr(gp.LinExpr(self.symmetric_gen, v),
                             inequality_senses[1], 1)

        self.le = {}
        self.constraints = {}

        for indices, generator in self.generators.items():
            self.le[indices] = gp.LinExpr(generator, v)

    def get_solution(self):
        return [var.x for var in self.model.getVars()]

    def run(self):
        start = time.time()
        self.inner_run()
        self.duration = time.time() - start

    def output(self, p):
        if self.OUTPUT:
            s = []
            for i in range(self.problem.n - 1):
                for j in range(i + 1, self.problem.n):
                    if p[i] < p[j]:
                        sig = 1
                        a1, a2 = p[i], p[j]
                    else:
                        sig = -1
                        a1, a2 = p[j], p[i]
                    s.append(((a1, a2), sig))
            s.sort()
            s = [(sig[1], sig[0]) for sig in s if sig[0] in self.generators]
            self.problem.create_one_cell(s)

    def inner_run(self):
        self.rsIncEnu(self.initial_beta, set())

    def recompute_permutation(self, beta):
        beta0 = np.array(beta).reshape((self.d, 1))
        beta0 = beta0 / beta0[-1, 0]
        beta0 = beta0[:-1, :]
        return self.problem.permutation(beta=beta0)

    def rsIncEnu(self, beta, oL):
        perm = self.recompute_permutation(beta)
        self.output(perm)
        L = deepcopy(oL)
        self.cell_number += 1
        # if self.cell_number % 5000 == 0:
        # print(self.cell_number)

        for i in range(self.problem.n - 1):
            hi = hypind(perm[i], perm[i + 1])
            if hi in L:
                continue
            if hi in self.R:
                continue
            if hi in self.Z:
                continue
            self.solve(perm, fixed=hi)
            if self.last_feasible:
                beta0 = self.new_beta(beta, self.last_beta)
                L.add(hi)
                self.rsIncEnu(beta0, deepcopy(L))

    def print_stats(self):
        print("""
        Algorithm {5}
        m = {0}
        d = {1}
        flp = {2}
        ilp = {3}
        total = {7}
        cn = {4}
        time = {6}
        """.format(self.m, self.d, self.flp, self.ilp, self.cell_number,
                   self.name, self.duration, (self.flp + self.ilp)))

    def return_stats(self):
        return "{0};{1};{2};{3};{4};{5};{6};{7}\n".format(
            self.problem.n, self.m, self.d, self.flp, self.ilp,
            self.cell_number, self.duration, self.name)
