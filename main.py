import numpy as np
import math
import argparse
from numba import njit
import graf
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--regime', default='base')
args = parser.parse_args()

with open('in.txt', 'r') as inp:
    h1 = float(inp.readline())
    h2 = float(inp.readline())
    delta = float(inp.readline())
    L = float(inp.readline())
    L1 = float(inp.readline())
    W = float(inp.readline())
    lambda_ = float(inp.readline())
    ni = int(inp.readline())
    nj = int(inp.readline())
    k_max = int(inp.readline())
    eps = float(inp.readline())
    pres = int(inp.readline())


@njit
def h(x, L1, h1, h2, delta):
    if x <= L1:
        h_x = h1 - (((h1 - delta - h2) * x) / L1)
    else:
        h_x = h2
    return h_x


@njit
def integral(p, ni, nj, ds):
    s = 0
    for i in range(ni - 1):
        for j in range(nj - 1):
            s += (((p[i][j] + p[i + 1][j] + p[i + 1][j + 1] + p[i][j + 1]) * ds) / 4)
    return s


@njit
def theory(x, L1, h1, h2, delta, lambda_):
    s = lambda_ * (h1 - 1) / (2 * (h1 ** 3 + 1))
    if x < L1:
        return s * x / L1
    else:
        return s * (x - 1) / (L1 - 1)


@njit
def theory2(x, L1, h1, h2, delta, lambda_):
    return lambda_ * (h1 - h(x, L1, h1, h2, delta)) * (h(x, L1, h1, h2, delta) - h2) / (
            h(x, L1, h1, h2, delta)**2 * (h1**2 - h2**2))


@njit
def calculations_only(ni, nj, dx, dy, p, node_L1, h_i, h1, pres):
    shod_out = []
    force = 0

    b = (h2**3) / dx
    t = ((h2 + delta)**3) / dx

    for k in range(k_max):
        e = 0
        for i in range(1, ni - 1):
            for j in range(1, nj):
                if j == nj - 1:
                    if pres == 1:
                        new_pij = p[i, j - 1]
                    else:
                        new_pij = (4.0 / 3.0) * p[i, j - 1] - (1.0 / 3.0) * p[i, j - 2]
                    e = max(e, abs(p[i][j] - new_pij))
                    p[i, j] = new_pij
                else:
                    if i == node_L1:
                        if pres == 1:
                            new_pij = (b * p[i + 1, j] + t * p[i - 1, j] + delta) / (t + b)
                        else:
                            temp1 = 4.0 * p[i + 1, j] - p[i + 2, j]
                            temp2 = -4.0 * p[i - 1, j] + p[i - 2, j]
                            new_pij = ((b / 2) * temp1 - (t / 2) * temp2 + delta) / ((3.0 / 2.0) * (t + b))
                        e = max(e, abs(p[i][j] - new_pij))
                        p[i, j] = new_pij
                    if i < node_L1:
                        h_i_1 = h((i + 0.5) * dx, L1, h1, h2, delta)
                        h_i_2 = h((i - 0.5) * dx, L1, h1, h2, delta)
                        h_i_3 = h((i - 1) * dx, L1, h1, h2, delta)
                        h_i_4 = h((i + 1) * dx, L1, h1, h2, delta)
                        h_j = h(i * dx, L1, h1, h2, delta)
                        temp1 = h_i_1**3 / dx**2
                        temp2 = h_i_2**3 / dx**2
                        temp3 = h_j**3 / dy**2
                        temp4 = (h_i_4 - h_i_3) / (2 * dx)
                        temp5 = temp1 + temp2 + 2 * temp3
                        new_pij = (temp1 * p[i + 1, j] +
                                   temp2 * p[i - 1, j] +
                                   temp3 * (p[i, j + 1] + p[i, j - 1]) -
                                   temp4) / temp5
                        e = max(e, abs(p[i][j] - new_pij))
                        p[i, j] = new_pij
                    if i > node_L1:
                        s1 = (p[i + 1][j] + p[i - 1][j]) / dx ** 2
                        s2 = (p[i][j + 1] + p[i][j - 1]) / dy ** 2
                        s3 = (1 / dx ** 2 + 1 / dy ** 2) * 2
                        new_pij = (s1 + s2) / s3
                        e = max(e, abs(p[i][j] - new_pij))
                        p[i, j] = new_pij

        force = integral(p, ni, nj, dx * dy)
        shod_out.append([k, e, force])
        if e < eps:
            break
    return shod_out, p, force


def main(ni, nj, L1, delta, h1, pres):
    dx = L / (ni - 1)
    dy = W / (nj - 1) / 2
    node_L1 = int(L1 / dx)
    h_i = np.empty(ni, dtype='float64')

    for i in range(ni):
        if i <= node_L1:
            h_i[i] = h1 - (i * dx / L1) * (h1 - h2 - delta)
        else:
            h_i[i] = h2

    p = np.zeros([ni, nj], dtype='float64')

    shod, p_result, force_result = calculations_only(ni, nj, dx, dy, p, node_L1, h_i, h1, pres)
    print(f'Iter: {int(shod[-1][0])}')

    with open('data' + os.sep + 'pole.txt', 'w') as file:
        for j in range(nj):
            for i in range(ni):
                file.write(f'{(i - 1) * dx} {(j - 1) * dy} {p_result[i][j]}\n')

    with open('data' + os.sep + 'shod.txt', 'w') as file:
        for k in shod:
            file.write(f'{k[0]} {k[1]} {k[2]}\n')

    with open('data' + os.sep + 'theory.txt', 'w') as file:
        for i in range(ni):
            file.write(f'{i * dx} {p_result[i][nj - 1]} {theory2(i * dx, L1, h1, h2, delta, lambda_)}\n')

    with open('data' + os.sep + 'h.txt', 'w') as file:
        for i in range(ni):
            file.write(f'{i * dx} {h(i * dx, L1, h1, h2, delta)}\n')

    with open(file='data' + os.sep + 'result.plt', mode='w') as output:
        output.write('VARIABLES="X", "Y", "P", "H"\n')
        output.write(f'Zone i={ni}\n')
        output.write(f'j={nj}\n')
        for j in range(nj):
            for i in range(ni):
                output.write(f'{(i * dx):.4f}    {(j * dy):.4f}    {p_result[i, j]:.4f}   {h(i * dx, L1, h1, h2, delta)}\n')

    return force_result


if args.regime == 'base':
    t0 = time.time()
    print(f'F = {main(ni, nj, L1, delta, h1, pres)}')
    print(f't = {time.time() - t0}')
    graf.shod()
    graf.theory()
    graf.h_graf()

elif args.regime == 'table':
    def dx(ni):
        return L / (ni - 1)

    def dy(nj):
        return W / (nj - 1) / 2

    for pres in [1, 2]:
        print(f'Pres = {pres}')
        ns = []
        for fix in ['j', 'i']:
            results = []
            flex = [51, 101, 201]
            if fix == 'i':
                ni = flex[0]
                for nj in flex:
                    f = main(ni, nj, L1, delta, h1, pres)
                    print(f'ni = {ni}, nj = {nj}, dx = {dx(ni):.3e}, dy = {dy(nj):.3e},  F = {f:.3e}')
                    results.append(f)
            else:
                nj = flex[0]
                for ni in flex:
                    f = main(ni, nj, L1, delta, h1, pres)
                    print(f'ni = {ni}, nj = {nj}, dx = {dx(ni):.3e}, dy = {dy(nj):.3e},  F = {f:.3e}')
                    results.append(f)

            d1 = results[0] - results[1]
            d2 = results[1] - results[2]
            ddelta1 = abs(d1 / results[1]) if pres == 1 else abs(d1 / 3 / results[1])
            ddelta2 = abs(d2 / results[2]) if pres == 1 else abs(d2 / 3 / results[2])
            n1 = ddelta1 * 1000 if pres == 1 else (ddelta1 * 1000) ** (1 / 2)
            n2 = ddelta2 * 1000 if pres == 1 else (ddelta2 * 1000) ** (1 / 2)
            ns.append(n2 * flex[2] if fix == 'j' else n2 * flex[2])
            print(f'nn {n1:.3} {n2:.3}')
            print(f'Delta F = {d1:.3e}, {d2:.3e}')
            print(f'%delta = {(ddelta1 * 100):.3}%, {(ddelta2 * 100):.3}%')
            print(f'm = {(math.log2(abs(d1 / d2))):.3}')
        print(f'recommended {int(ns[0])} x {int(ns[1])}')

elif args.regime == 'l1':
    l1s = []
    fs = []
    dots = 19
    for index in range(dots):
        L1 = (index + 1) * (L / (dots + 1))
        f = main(ni, nj, L1, delta, h1, pres)
        l1s.append(L1)
        fs.append(f)
        print(f'L1 = {L1}, F = {f}')
    with open('data/l1.txt', 'w') as file:
        for index in range(len(l1s)):
            file.write(f'{l1s[index]} {fs[index]}\n')
    graf.l1()

elif args.regime == 'delta':
    deltas = []
    fs = []
    dots = 19
    delta_max = 5
    for index in range(dots):
        delta = (index + 1) * (delta_max / (dots + 1))
        h1 = delta + h2
        f = main(ni, nj, L1, delta, h1, pres)
        deltas.append(delta)
        fs.append(f)
        print(f'Delta = {delta}, F = {f}')
    with open('data/delta.txt', 'w') as file:
        for index in range(len(deltas)):
            file.write(f'{deltas[index]} {fs[index]}\n')
    graf.delta()