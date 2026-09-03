"""
geometry/dump_stress_components.py
-----------------------------------------------------------------------------
Dump nodal-averaged stress COMPONENTS for the BASE-1 instance at the end of
Preload_Step and at the end of Load_Step.

model_<N>_results.csv only stores invariants (Mises / principals), which is not
enough to check the analytical model: that model predicts a single normal
component on a specific section, and it contains no preload term. Writing both
steps lets the analysis subtract them:

    DELTA = S(Load_Step) - S(Preload_Step)

which isolates the response to the applied load P alone - the only load case the
analytical model of Chapter 2 represents.

Run with Abaqus python (2.7-compatible, no CAE licence needed):

    abaqus python geometry/dump_stress_components.py model_0

Writes output/<model>/<model>_components.csv with columns
    Node, X, Y, Z, P11..P23 (preload), L11..L23 (preload + P)
-----------------------------------------------------------------------------
"""

import os
import sys

from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL

COMPONENTS = ['11', '22', '33', '12', '13', '23']


def _nodal_stress(odb, instance, step_name):
    """Extrapolate S to nodes and average per node -> {label: [6 components]}."""
    step = odb.steps[step_name]
    nf = len(step.frames)
    if nf == 0:
        raise ValueError('%s has 0 frames' % step_name)
    frame = step.frames[nf - 1]
    print('  %-14s %2d frames, last frameValue = %s'
          % (step_name, nf, frame.frameValue))

    field = frame.fieldOutputs['S'].getSubset(
        region=instance, position=ELEMENT_NODAL)

    sums, counts = {}, {}
    for v in field.values:
        label = v.nodeLabel
        acc = sums.get(label)
        if acc is None:
            acc = [0.0] * 6
            sums[label] = acc
        for i in range(6):
            acc[i] += v.data[i]
        counts[label] = counts.get(label, 0) + 1

    return dict((lab, [s / counts[lab] for s in acc])
                for lab, acc in sums.items())


def dump(model_name, results_dir=None, out_dir=None):
    if results_dir is None:
        results_dir = os.path.join('output', model_name)
    if out_dir is None:
        out_dir = results_dir

    odb_path = os.path.join(results_dir, model_name + '.odb')
    if not os.path.exists(odb_path):
        raise IOError('ODB not found: %s' % odb_path)

    odb = openOdb(path=odb_path, readOnly=True)
    try:
        print('%s: steps = %s' % (model_name, list(odb.steps.keys())))
        instance = odb.rootAssembly.instances['BASE-1']
        coords = dict((n.label, n.coordinates) for n in instance.nodes)

        S_pre = _nodal_stress(odb, instance, 'Preload_Step')
        S_load = _nodal_stress(odb, instance, 'Load_Step')
    finally:
        odb.close()

    out_path = os.path.join(out_dir, model_name + '_components.csv')
    fh = open(out_path, 'w')
    try:
        fh.write('Node,X,Y,Z,' +
                 ','.join(['P' + c for c in COMPONENTS]) + ',' +
                 ','.join(['L' + c for c in COMPONENTS]) + '\n')
        n_written = 0
        for label in sorted(coords.keys()):
            pre, load = S_pre.get(label), S_load.get(label)
            if pre is None or load is None:
                continue          # node carries no stress (e.g. orphan)
            x, y, z = coords[label]
            fh.write('%d,%.6f,%.6f,%.6f,' % (label, x, y, z))
            fh.write(','.join(['%.6g' % s for s in pre]) + ',')
            fh.write(','.join(['%.6g' % s for s in load]) + '\n')
            n_written += 1
    finally:
        fh.close()

    print('wrote %s (%d nodes)' % (out_path, n_written))
    return out_path


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'model_0'
    dump(name)
