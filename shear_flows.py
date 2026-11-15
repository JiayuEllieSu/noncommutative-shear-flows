from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
DATA = OUT / 'data'
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

BLUE, ORANGE, GRAY = '#365F70', '#AA6A4F', '#747B7D'
A = np.array([[1, 1], [0, 1]])
B = np.array([[1, 0], [1, 1]])
M = A @ B
C = A @ np.array([[1, 0], [-1, 1]])
MAPS = {'single': A @ A, 'alternating': M, 'opposite': C}
CYCLES = [0, 2, 4, 6]
N = 16000

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix',
                     'font.size': 12, 'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', dpi=250)
    if name in ('shear_geometry', 'growth_regimes', 'stretching_comparison',
                'tracer_protocols', 'count_variance_wide'):
        fig.savefig(FIG / f'{name}.png', bbox_inches='tight', dpi=160)
    plt.close(fig)


def disk(n=N, seed=630):
    rng = np.random.default_rng(seed)
    r = .12 * np.sqrt(rng.random(n))
    theta = 2 * np.pi * rng.random(n)
    return np.column_stack((.27 + r*np.cos(theta), .39 + r*np.sin(theta)))


def move(points, matrix):
    return (points @ matrix.T) % 1


def variance(points, bins=32):
    counts = np.histogram2d(points[:, 0], points[:, 1], bins=bins,
                            range=[[0, 1], [0, 1]])[0]
    return np.mean((counts / (len(points)/bins**2) - 1)**2)


def stretch(matrix):
    return np.linalg.svd(matrix, compute_uv=False)[0]


def write_csv(filename, header, rows):
    with (DATA / filename).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def shear_figures():
    rectangle = np.array([[-.30, -.18], [.30, -.18], [.30, .18], [-.30, .18]])
    cases = [('initial', np.eye(2)), ('A', A), ('B', B), ('AB', M), ('BA', B @ A)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (name, matrix) in zip(axes, cases):
        p = rectangle @ matrix.T
        loop = np.vstack((p, p[0]))
        color = ORANGE if name == 'BA' else BLUE
        ax.fill(p[:, 0], p[:, 1], color=color, alpha=.16)
        ax.plot(loop[:, 0], loop[:, 1], color=color, lw=2.2)
        ax.axhline(0, color='#E0E0E0', lw=.6)
        ax.axvline(0, color='#E0E0E0', lw=.6)
        ax.set(xlim=(-.85, .85), ylim=(-.85, .85), aspect='equal', title=name)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values(): side.set_visible(False)

        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.fill(p[:, 0], p[:, 1], color=color, alpha=.16)
        ax2.plot(loop[:, 0], loop[:, 1], color=color, lw=2)
        ax2.set(xlim=(-.85, .85), ylim=(-.85, .85), aspect='equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        save(fig2, 'shear_' + name)
    save(fig, 'shear_geometry')


def stretching_figures():
    n = np.arange(13)
    values = {name: np.array([stretch(np.linalg.matrix_power(matrix, int(k)))
                              for k in n]) for name, matrix in MAPS.items()}
    lam = (3 + np.sqrt(5))/2
    write_csv('stretching.csv', ['cycle', 'alternating', 'single', 'opposite', 'exact'],
              zip(n, values['alternating'], values['single'], values['opposite'], lam**n))

    fig, ax = plt.subplots(figsize=(10, 5.3))
    for name, color, style in [('alternating', BLUE, '-'), ('single', ORANGE, '--'),
                               ('opposite', GRAY, ':')]:
        ax.semilogy(n, values[name], color=color, ls=style, lw=2.2, label=name)
    ax.plot(n[::2], lam**n[::2], 'o', mfc='white', mec=BLUE, ms=5)
    ax.set(xlabel='Cycles $n$', ylabel='Maximum stretch $S_n$', xlim=(0, 12), ylim=(.75, 2e5))
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False, ncol=3)
    save(fig, 'stretching_comparison')

    fig, ax = plt.subplots(figsize=(9, 5.0))
    for name, color, style in [('alternating', BLUE, '-'), ('single', ORANGE, '--'),
                               ('opposite', GRAY, ':')]:
        ax.semilogy(n, values[name], color=color, ls=style, lw=2.2, label=name)
    ax.plot(n[::2], lam**n[::2], 'o', mfc='white', mec=BLUE, ms=5)
    ax.set(xlabel='Cycles $n$', ylabel='Maximum stretch $S_n$', xlim=(0, 12), ylim=(.75, 2e5))
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False, ncol=2)
    save(fig, 'stretching')
    return values


def regime_figures():
    q = np.linspace(-6, 4, 1001)
    trace = 2 + q
    chi = np.where(abs(trace) > 2, np.arccosh(np.maximum(1, abs(trace)/2)), 0)
    write_csv('regimes.csv', ['ab', 'trace', 'lyapunov_per_cycle'], zip(q, trace, chi))

    for name, size in [('growth_regimes', (10, 4.2)), ('regimes', (9, 4.8))]:
        fig, ax = plt.subplots(figsize=size)
        ax.plot(q, chi, color=BLUE, lw=2, label='Trace formula')
        sample = np.arange(0, len(q), 50)
        eigen = []
        for i in sample:
            matrix = np.array([[1+q[i], 1], [q[i], 1]])
            eigen.append(max(0, np.log(max(abs(np.linalg.eigvals(matrix))))))
        ax.plot(q[sample], eigen, 'o', ms=4, mfc='white', mec=ORANGE,
                label='Eigenvalue check')
        for x in (-4, 0): ax.axvline(x, color='#C7C7C7', ls=':', lw=1)
        ax.set(xlabel='Product of shear strengths $ab$', ylabel='Largest exponent $\\chi$',
               xlim=(-6, 4), ylim=(-.04, 1.9))
        ax.legend(frameon=False)
        save(fig, name)


def tracer_figures(hist, initial):
    labels = {'single': r'One direction $A^{2n}$', 'alternating': r'Alternating $(AB)^n$',
              'opposite': r'Opposite sign $C^n$'}
    mask = initial[:, 0] < .27
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.3), sharex=True, sharey=True)
    for row, name in enumerate(MAPS):
        for col, n in enumerate(CYCLES):
            ax = axes[row, col]
            p = hist[name][n]
            ax.scatter(p[mask, 0], p[mask, 1], s=.7, c=BLUE, lw=0, rasterized=True)
            ax.scatter(p[~mask, 0], p[~mask, 1], s=.7, c=ORANGE, lw=0, rasterized=True)
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal')
            ax.set_xticks([0, .5, 1])
            ax.set_yticks([0, .5, 1])
            if row == 0: ax.set_title(f'{n} cycles')
            if col == 0: ax.set_ylabel(labels[name])
            ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, 'tracer_protocols')

    for name in MAPS:
        for n in [0, 2, 4, 6, 8]:
            fig, ax = plt.subplots(figsize=(3.1, 3.1))
            p = hist[name][n]
            ax.scatter(p[mask, 0], p[mask, 1], s=.5, c=BLUE, lw=0, rasterized=True)
            ax.scatter(p[~mask, 0], p[~mask, 1], s=.5, c=ORANGE, lw=0, rasterized=True)
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal')
            ax.set_xticks([0, .5, 1])
            ax.set_yticks([0, .5, 1])
            save(fig, f'report_tracer_{name}_{n}')


def coarse_figures(hist):
    n = np.arange(13)
    v = {name: np.array([variance(p) for p in positions]) for name, positions in hist.items()}
    write_csv('coarse_counts.csv', ['cycle', 'alternating', 'single', 'opposite', 'uniform_reference'],
              zip(n, v['alternating'], v['single'], v['opposite'], np.full(13, 1023/16000)))

    for name, size in [('count_variance_wide', (15, 4.4)), ('coarse_counts', (9, 5.0))]:
        fig, ax = plt.subplots(figsize=size)
        for label, color, style in [('alternating', BLUE, '-'), ('single', ORANGE, '--'),
                                    ('opposite', GRAY, ':')]:
            ax.semilogy(n, v[label], ls=style, lw=2, marker='o', ms=3,
                        color=color, label=label)
        ax.axhline(1023/16000, ls='-.', color='black', lw=1,
                   label='Uniform sampling expectation')
        ax.set(xlabel='Cycles $n$', ylabel='Coarse count variance $V_n$', xlim=(0, 12), ylim=(.04, 30))
        ax.grid(axis='y', alpha=.18)
        ax.legend(frameon=False, ncol=2 if name == 'coarse_counts' else 4)
        save(fig, name)
    return v


def main():
    assert np.array_equal(np.linalg.matrix_power(C, 6), np.eye(2))
    initial = disk()
    hist = {}
    for name, matrix in MAPS.items():
        positions = [initial.copy()]
        for _ in range(12): positions.append(move(positions[-1], matrix))
        hist[name] = np.array(positions)
    np.savez_compressed(DATA / 'trajectories.npz', **hist)

    s = stretching_figures()
    regime_figures()
    shear_figures()
    tracer_figures(hist, initial)
    v = coarse_figures(hist)

    expected = [8.727448, 0.066544, 19.320624]
    actual = [v[name][6] for name in ('single', 'alternating', 'opposite')]
    assert np.allclose(actual, expected, atol=1e-8)
    assert np.isclose(s['alternating'][10], 15126.999933893041)
    assert np.isclose(s['single'][10], 20.04987562112089)

    print('Eigenvalues:', np.linalg.eigvals(M))
    print('Largest Lyapunov exponent:', np.log((3 + np.sqrt(5))/2))
    print('Stretch at cycle 10:', s['alternating'][10], s['single'][10])
    print('Coarse variance at cycle 0:', v['alternating'][0])
    print('Coarse variance at cycle 6 (single, alternating, opposite):', *actual)
    print('Coarse variance at cycle 8:', *(v[name][8] for name in MAPS))
    print('Uniform-sampling expectation:', 1023/16000)
    print('Saved figures and data to:', OUT)


if __name__ == '__main__':
    main()
