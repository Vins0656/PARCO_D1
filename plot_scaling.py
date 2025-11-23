#!/usr/bin/env python3
# coding: utf-8
import re
import sys
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THREADSLIST_DEFAULT = [1, 2, 4, 8, 16, 32, 64]

def to_ms(val_str, unit_str):
    v = float(val_str)
    if not unit_str:
        return v  # assume ms
    u = unit_str.lower()
    if u in ('µs', 'us'):
        return v / 1000.0
    return v  # ms

def deduce_mode_from_filename(path):
    name = os.path.basename(path).lower()
    mode = []
    if 'spread' in name: mode.append('spread')
    if 'close' in name:  mode.append('close')
    if 'noopt' in name or 'no_opt' in name or 'senza' in name: mode.append('noopt')
    if 'opt' in name and 'noopt' not in name: mode.append('opt')
    return '+'.join(mode) if mode else 'default'

def parse_outputs(files):
    # data[matrix][mode][algo][T] = ms
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    re_matrix = re.compile(r'^Matrix:\s*(.+?)\s*$')
    re_hdr = re.compile(r'^Threads:\s*(\d+)\s*\|\s*Iterations:\s*([0-9]+)\s*\|\s*Chunk:\s*([0-9]+)', re.IGNORECASE)

    # Casi delle righe tempi:
    # - static: "Average time OMP static (chunk N, ...): VAL UNIT"
    re_omp_static = re.compile(
        r'^Average time OMP\s+static\s*\((.*?)\):\s*([0-9]*\.?[0-9]+)\s*([µu]?s|ms)?\s*$',
        re.IGNORECASE)

    # - guided/dynamic/auto: accetta o meno "(chunk N, ...)" in parentesi
    re_omp_other = re.compile(
        r'^Average time OMP\s+(guided|dynamic|auto)\s*\((.*?)\):\s*([0-9]*\.?[0-9]+)\s*([µu]?s|ms)?\s*$',
        re.IGNORECASE)
    # variante senza parentesi (alcuni log): "... auto : VAL UNIT"
    re_omp_other_noparen = re.compile(
        r'^Average time OMP\s+(guided|dynamic|auto)\s*:\s*([0-9]*\.?[0-9]+)\s*([µu]?s|ms)?\s*$',
        re.IGNORECASE)

    re_albus = re.compile(
        r'^Average time ALBUS\s*(?:\((.*?)\))?:\s*([0-9]*\.?[0-9]+)\s*([µu]?s|ms)?\s*$',
        re.IGNORECASE)

    # Legacy
    re_old = re.compile(r'Average time taken.*?:\s*([0-9]*\.?[0-9]+)\s*ms', re.IGNORECASE)

    current_matrix = None
    current_threads = None
    current_chunk = None

    for path in files:
        mode = deduce_mode_from_filename(path)
        current_matrix = None
        current_threads = None
        current_chunk = None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue

                    mM = re_matrix.match(line)
                    if mM:
                        current_matrix = mM.group(1)
                        continue

                    mH = re_hdr.match(line)
                    if mH:
                        try:
                            current_threads = int(mH.group(1))
                        except:
                            current_threads = None
                        try:
                            current_chunk = int(mH.group(3))
                        except:
                            current_chunk = None
                        continue

                    # OMP static: chunk dentro le parentesi
                    mS = re_omp_static.match(line)
                    if mS:
                        paren = mS.group(1) or ''
                        val = mS.group(2); unit = mS.group(3)
                        t_ms = to_ms(val, unit)
                        # estrai "chunk N" dalla parentesi
                        ch = None
                        mch = re.search(r'chunk\s*([0-9]+)', paren, re.IGNORECASE)
                        if mch:
                            ch = int(mch.group(1))
                        if ch is None:
                            ch = current_chunk
                        mat = current_matrix if current_matrix else os.path.splitext(os.path.basename(path))[0]
                        T = current_threads if current_threads else 1
                        data[mat][mode]['OMP-static'][T] = t_ms
                        continue

                    # OMP guided/dynamic/auto con parentesi
                    mO = re_omp_other.match(line)
                    if mO:
                        sched = mO.group(1).lower()
                        paren = mO.group(2) or ''
                        val = mO.group(3); unit = mO.group(4)
                        t_ms = to_ms(val, unit)
                        # estrai "chunk N" dalla parentesi se presente
                        ch = None
                        mch = re.search(r'chunk\s*([0-9]+)', paren, re.IGNORECASE)
                        if mch:
                            ch = int(mch.group(1))
                        if ch is None:
                            ch = current_chunk
                        mat = current_matrix if current_matrix else os.path.splitext(os.path.basename(path))[0]
                        T = current_threads if current_threads else 1
                        data[mat][mode][f'OMP-{sched}'][T] = t_ms
                        continue

                    # OMP guided/dynamic/auto senza parentesi (fallback)
                    mOnp = re_omp_other_noparen.match(line)
                    if mOnp:
                        sched = mOnp.group(1).lower()
                        val = mOnp.group(2); unit = mOnp.group(3)
                        t_ms = to_ms(val, unit)
                        mat = current_matrix if current_matrix else os.path.splitext(os.path.basename(path))[0]
                        T = current_threads if current_threads else 1
                        # usa l'header per il chunk (non lo salviamo qui ma il dato resta valido per speedup)
                        data[mat][mode][f'OMP-{sched}'][T] = t_ms
                        continue

                    # ALBUS (nessun chunk da estrarre)
                    mA = re_albus.match(line)
                    if mA:
                        val = mA.group(2); unit = mA.group(3)
                        t_ms = to_ms(val, unit)
                        mat = current_matrix if current_matrix else os.path.splitext(os.path.basename(path))[0]
                        T = current_threads if current_threads else 1
                        data[mat][mode]['ALBUS'][T] = t_ms
                        continue

                    # Legacy
                    if 'Average time taken' in line:
                        m_old = re_old.search(line)
                        if m_old:
                            t_ms = float(m_old.group(1))
                            mat = current_matrix if current_matrix else os.path.splitext(os.path.basename(path))[0]
                            T = current_threads if current_threads else 1
                            data[mat][mode]['OMP'][T] = t_ms
                            continue

        except Exception as e:
            print(f'Errore leggendo {path}: {e}')
            continue

    return data

def compute_speedup(times_by_T):
    if not times_by_T or 1 not in times_by_T or times_by_T[1] <= 0:
        return {}
    base = times_by_T[1]
    return {T: (base / t) for T, t in times_by_T.items() if t > 0}

def compute_efficiency(speedup_by_T):
    return {T: (100.0 * sp / T) for T, sp in speedup_by_T.items()}

def sanitize(s):
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in s)

def plot_2x2(data_for_file, out_prefix, threadslist=None):
    threads = threadslist if threadslist else THREADSLIST_DEFAULT
    matrices = sorted(list(data_for_file.keys()))
    if not matrices:
        print(f'Nessuna matrice per {out_prefix}')
        return
    matrices = matrices[:4]  # 2x2

    # SPEEDUP
    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes_s = axes_s.ravel()
    for ax, mat in zip(axes_s, matrices):
        ax.plot(threads, threads, linestyle='--', color='#888888', label='ideale')
        for mode, algos in data_for_file[mat].items():
            for algo in ['OMP-static','OMP-guided','OMP-dynamic','OMP-auto','ALBUS','OMP']:
                if algo not in algos:
                    continue
                sp = compute_speedup(algos[algo])
                if not sp:
                    continue
                Ts = sorted(sp.keys())
                Ys = [sp[T] for T in Ts]
                ax.plot(Ts, Ys, marker='o', label=f'{mode} {algo}')
        ax.set_title(f'{mat} - Speedup')
        ax.set_xlabel('Threads')
        ax.set_ylabel('Speedup vs T=1')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks(threads)
        ax.legend(fontsize=8, loc='best')
    for k in range(len(matrices), 4):
        axes_s[k].axis('off')
    fig_s.suptitle(f'Scaling (speedup) - {out_prefix}', fontsize=14)
    fig_s.savefig(f'scaling_{sanitize(out_prefix)}.png', dpi=150)
    plt.close(fig_s)

    # EFFICIENZA
    fig_e, axes_e = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes_e = axes_e.ravel()
    for ax, mat in zip(axes_e, matrices):
        for mode, algos in data_for_file[mat].items():
            for algo in ['OMP-static','OMP-guided','OMP-dynamic','OMP-auto','ALBUS','OMP']:
                if algo not in algos:
                    continue
                sp = compute_speedup(algos[algo])
                eff = compute_efficiency(sp)
                if not eff:
                    continue
                Ts = sorted(eff.keys())
                Ys = [eff[T] for T in Ts]
                ax.plot(Ts, Ys, marker='s', label=f'{mode} {algo}')
        ax.set_title(f'{mat} - Efficienza (%)')
        ax.set_xlabel('Threads')
        ax.set_ylabel('Efficienza (%)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks(threads)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8, loc='best')
    for k in range(len(matrices), 4):
        axes_e[k].axis('off')
    fig_e.suptitle(f'Efficienza - {out_prefix}', fontsize=14)
    fig_e.savefig(f'efficiency_{sanitize(out_prefix)}.png', dpi=150)
    plt.close(fig_e)

def process_file(inputfile):
    data_all = parse_outputs([inputfile])
    base = os.path.splitext(os.path.basename(inputfile))[0]
    plot_2x2(data_all, out_prefix=base, threadslist=THREADSLIST_DEFAULT)
    print(f"Creati: scaling_{base}.png, efficiency_{base}.png")

if __name__ == '__main__':
    files = sys.argv[1:]
    if not files:
        print('Uso: python plot_scaling.py risultati_1_close.txt risultati_1_spread.txt ...')
        sys.exit(0)
    for infile in files:
        if os.path.exists(infile):
            process_file(infile)
        else:
            print(f'Attenzione: {infile} non trovato, salto.')
