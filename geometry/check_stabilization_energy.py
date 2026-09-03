"""Extract stabilization-energy diagnostics from an ODB.

The model applies viscous stabilization in both steps, so part of the
"load-only" field recovered by the preload decomposition
    sigma^P = sigma|end of Load_Step - sigma|end of Preload_Step
may be damping artefact rather than structural response. The subtraction is
only meaningful if the dissipated energy is negligible.

Run with:
    abaqus python check_stabilization_energy.py [odb_path]

Default target is output/model_0/model_0.odb.
"""
import os
import sys

from odbAccess import openOdb

# Energies of interest. ALLSD is the static-dissipation (stabilization)
# energy; ALLIE the total internal energy. Their ratio is the figure of merit.
WANTED = ('ALLSD', 'ALLIE', 'ALLWK', 'ALLKE', 'ALLVD', 'ALLCD',
          'ALLFD', 'ALLPD', 'ALLAE', 'ALLSE', 'ETOTAL')

# Abaqus documentation guidance: stabilization is acceptable when the
# dissipated energy is a small fraction of the internal energy.
TOL_GOOD = 0.01     # <= 1 %  : negligible
TOL_WARN = 0.05     # <= 5 %  : tolerable but should be reported


def collect(odb):
    """Return {step_name: {energy_name: [(time, value), ...]}}."""
    out = {}
    for step_name, step in odb.steps.items():
        series = {}
        for hr in step.historyRegions.values():
            for key, ho in hr.historyOutputs.items():
                name = key.split()[0].upper()
                if name in WANTED:
                    # Whole-model energies live in a single history region;
                    # if several report the same name, keep the longest.
                    if name not in series or len(ho.data) > len(series[name]):
                        series[name] = list(ho.data)
        out[step_name] = series
    return out


def fmt(v):
    return '%12.4e' % v if v is not None else '%12s' % 'n/a'


def main(path):
    if not os.path.isfile(path):
        sys.exit('ODB not found: %s' % path)

    odb = openOdb(path, readOnly=True)
    try:
        data = collect(odb)
        step_names = list(odb.steps.keys())
    finally:
        odb.close()

    print('')
    print('STABILIZATION ENERGY CHECK')
    print('  odb: %s' % path)
    print('')

    if not any(data.values()):
        print('  No whole-model energy history found in this ODB.')
        print('  The builder requests only S, U and RF, and no history output')
        print('  for energies, so ALLSD/ALLIE cannot be recovered from this')
        print('  run. Add a HistoryOutputRequest with variables=(ALLSD,')
        print('  ALLIE, ALLWK) and re-run to enable this check.')
        return 2

    verdict = 0
    for step_name in step_names:
        series = data.get(step_name, {})
        print('-' * 64)
        print('STEP: %s' % step_name)
        if not series:
            print('  no energy history in this step')
            continue

        print('  %-8s %12s %12s' % ('energy', 'final', 'max'))
        for name in WANTED:
            if name in series:
                vals = [v for _, v in series[name]]
                print('  %-8s %s %s' % (name, fmt(vals[-1]), fmt(max(vals))))

        if 'ALLSD' in series and 'ALLIE' in series:
            sd = [v for _, v in series['ALLSD']]
            ie = [v for _, v in series['ALLIE']]
            final = sd[-1] / ie[-1] if ie[-1] else float('inf')
            n = min(len(sd), len(ie))
            worst = max((sd[i] / ie[i]) for i in range(n) if ie[i])

            print('')
            print('  ALLSD/ALLIE  final = %8.4f %%' % (100 * final))
            print('  ALLSD/ALLIE  worst = %8.4f %%' % (100 * worst))

            if final <= TOL_GOOD:
                print('  -> PASS: dissipated energy negligible (<= 1 %).')
            elif final <= TOL_WARN:
                print('  -> MARGINAL: 1-5 %. Usable, but report the value and')
                print('     do not treat small differences as structural.')
                verdict = max(verdict, 1)
            else:
                print('  -> FAIL: > 5 %. A material part of the recovered')
                print('     field is damping, not structural response.')
                verdict = 2
        print('')

    # The decomposition subtracts two states, so what matters for it is the
    # dissipation accumulated during Load_Step alone.
    if 'Load_Step' in data and 'Preload_Step' in data:
        ls, ps = data['Load_Step'], data['Preload_Step']
        if 'ALLSD' in ls and 'ALLSD' in ps and 'ALLIE' in ls:
            sd_end = [v for _, v in ls['ALLSD']][-1]
            sd_pre = [v for _, v in ps['ALLSD']][-1]
            ie_end = [v for _, v in ls['ALLIE']][-1]
            ie_pre = [v for _, v in ps['ALLIE']][-1]
            d_sd, d_ie = sd_end - sd_pre, ie_end - ie_pre
            print('=' * 64)
            print('RELEVANT TO THE PRELOAD DECOMPOSITION')
            print('  dissipation added during Load_Step : %s' % fmt(d_sd))
            print('  internal energy added              : %s' % fmt(d_ie))
            if d_ie:
                r = d_sd / d_ie
                print('  ratio                              : %8.4f %%'
                      % (100 * r))
                print('')
                if abs(r) <= TOL_GOOD:
                    print('  -> The increment recovered by the subtraction is')
                    print('     structural: damping added over Load_Step is')
                    print('     negligible against the work done by P.')
                else:
                    print('  -> Damping added over Load_Step is NOT negligible.')
                    print('     sigma_L - sigma_P contains a damping component,')
                    print('     and that component depends on increment size.')
                    verdict = max(verdict, 2)
            print('')

    return verdict


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        'output', 'model_0', 'model_0.odb')
    sys.exit(main(target))
