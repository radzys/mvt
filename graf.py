from matplotlib import pyplot as plt
import os


def shod():
    iterations = []
    force = []
    e = []
    with open('data' + os.sep + 'shod.txt', 'r') as inp:
        for line in inp:
            line = line.split()
            iterations.append(float(line[0]))
            e.append(float(line[1]))
            force.append(float(line[2]))

    index01 = 0
    maxforce = max(force)
    for i in range(len(force)):
        if force[i] >= maxforce * 0.999:
            index01 = i
            break


    fig = plt.figure()
    ax = fig.add_subplot()
    ax2 = ax.twinx()
    ax.grid()

    ax.plot(iterations, force, label='Force', c='red')

    ax2.semilogy(iterations, e, label='Res', c='blue')
    ax.legend(bbox_to_anchor=(1.0, 0.6))
    ax2.legend(bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Force')
    ax.vlines(x=index01, ymin=0, ymax=force[index01])
    # ax.hlines(y=force[index01], xmin=0, xmax=index01)
    plt.savefig('figs/force')
    print(f'force fig saved, 0.999 force: iterations={index01}')


def theory():
    x = []
    y = []
    p = []
    p_theor = []
    with open('data' + os.sep + 'theory.txt', 'r') as pressure:
        for line in pressure:
            line = line.split()
            x.append(float(line[0]))
            p.append(float(line[1]))
            p_theor.append(float(line[2]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    ax.plot(x, p, label='Calc', c='red')
    ax.plot(x, p_theor, label='Theory', c='blue', linestyle=':')
    ax.legend(loc='upper left')
    ax.set_xlabel('x')
    ax.set_ylabel('P')
    plt.savefig('figs/pressure')
    print('pressure fig saved')


def h_graf():
    xs = []
    hs = []
    with open('data' + os.sep + 'h.txt', 'r') as pressure:
        for line in pressure:
            line = line.split()
            xs.append(float(line[0]))
            hs.append(float(line[1]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.minorticks_on()
    ax.grid()
    ax.grid(which='minor')
    ax.scatter(xs, hs, s=2)

    ax.set_xlabel(r'x')
    ax.set_ylabel('h')
    plt.savefig('figs/h')
    print('h fig saved')


def l1():
    l1s = []
    f = []
    with open('data' + os.sep + 'l1.txt', 'r') as pressure:
        for line in pressure:
            line = line.split()
            l1s.append(float(line[0]))
            f.append(float(line[1]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    ax.scatter(l1s, f)
    ax.set_xlabel(r'$L_1$')
    ax.set_ylabel('Force')
    plt.savefig('figs/l1')
    print('L1 fig saved')


def delta():
    deltas = []
    f = []
    with open('data' + os.sep + 'delta.txt', 'r') as pressure:
        for line in pressure:
            line = line.split()
            deltas.append(float(line[0]))
            f.append(float(line[1]))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    ax.scatter(deltas, f)
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel('Force')
    plt.savefig('figs/delta')
    print('Delta fig saved')

