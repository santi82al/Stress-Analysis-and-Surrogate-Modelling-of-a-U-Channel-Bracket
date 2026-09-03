"""
Render contour plots straight from an ODB, headless.

The thesis needs FE contour images that can sit beside the surrogate's predicted
fields, so the two must be produced the same way every time rather than by hand
in the GUI. Abaqus/CAE can print a viewport off-screen under noGUI, which is what
this script does.

Run (paths come through the environment, because Abaqus rewrites sys.argv with
its own flags -- the same convention run_models.py uses):

    $env:ODB_PATH   = "Predictor/fem_check/model_9001/model_9001.odb"
    $env:SHOT_OUT   = "Thesis_LaTex/figures"
    $env:SHOT_PREFIX= "fem_model9001"
    abaqus cae noGUI=geometry/odb_screenshot.py

Camera: viewVector (1,1,1) with Y up, matching the axis convention of the
surrogate field plots (X right, Y up, Z toward the front) so the two figures are
read the same way. It is an equivalent viewpoint, not a pixel-identical one --
Abaqus and matplotlib do not parameterise the camera the same way.
"""

import os
import sys

from abaqus import session
from abaqusConstants import (CONTOURS_ON_DEF, INTEGRATION_POINT, INVARIANT,
                             NODAL, COMPONENT, PNG, OFF, ON, FEATURE, SOLID)
import visualization
import displayGroupOdbToolset as dgo


# Each entry: (label, output position, refinement, file suffix)
PLOTS = [
    ("S",  INTEGRATION_POINT, (INVARIANT, "Mises"), "smises"),
    ("U",  NODAL,             (COMPONENT, "U1"),    "u1"),
    ("U",  NODAL,             (COMPONENT, "U2"),    "u2"),
    ("U",  NODAL,             (COMPONENT, "U3"),    "u3"),
]


def main():
    odb_env = os.environ.get("ODB_PATH", "")
    if not odb_env:
        raise SystemExit("set ODB_PATH (and optionally SHOT_OUT, SHOT_PREFIX)")
    odb_path = os.path.abspath(odb_env)
    out_dir = os.path.abspath(os.environ.get("SHOT_OUT") or os.path.dirname(odb_path))
    prefix = os.environ.get("SHOT_PREFIX", "fem")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # session.openOdb takes the file path as its `name` argument; passing only
    # path= raises "expected 1, got 0".
    odb = session.openOdb(name=odb_path, readOnly=True)
    vp = session.Viewport(name="shot", origin=(0.0, 0.0), width=260, height=200)
    vp.makeCurrent()
    vp.setValues(displayedObject=odb)

    # White background, no CAE furniture: the image goes into a report.
    session.graphicsOptions.setValues(backgroundStyle=SOLID,
                                      backgroundColor="#FFFFFF")
    vp.viewportAnnotationOptions.setValues(triad=ON, legend=ON, title=OFF,
                                           state=OFF, annotations=OFF,
                                           compass=OFF)
    vp.odbDisplay.commonOptions.setValues(visibleEdges=FEATURE)
    vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))

    # Show one instance only. The surrogate predicts the BRACKET; leaving the
    # bolt in the view lets its contact peak (~1086 MPa vs the bracket's ~294)
    # take over the legend and flatten the bracket field to a single colour.
    instance = os.environ.get("SHOT_INSTANCE", "")
    if instance:
        vp.odbDisplay.displayGroup.replace(
            leaf=dgo.LeafFromPartInstance(partInstanceName=(instance,)))
        print("restricted display to instance %s" % instance)
    vp.view.setViewpoint(viewVector=(1, 1, 1), cameraUpVector=(0, 1, 0))
    vp.view.fitView()

    session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)

    # Cap the stress legend so the field is legible. The peak sits on a singular
    # node at the bore and, left on auto, absorbs the whole colour ramp -- the
    # same reason the convergence study reports vm_p95 rather than the peak.
    smax = os.environ.get("SHOT_SMISES_MAX", "")

    for label, pos, refinement, suffix in PLOTS:
        try:
            vp.odbDisplay.setPrimaryVariable(variableLabel=label,
                                             outputPosition=pos,
                                             refinement=refinement)
            if suffix == "smises" and smax:
                # Everything above maxValue is painted with a dedicated
                # "outside limits" swatch, grey by default -- recolour just
                # that swatch to a dark red instead. The rest of the ramp
                # (band count, boundaries, all other colours) is untouched.
                vp.odbDisplay.contourOptions.setValues(
                    minAutoCompute=OFF, minValue=0.0,
                    maxAutoCompute=OFF, maxValue=float(smax),
                    outsideLimitsAboveColor="#8B0000")
            else:
                vp.odbDisplay.contourOptions.setValues(minAutoCompute=ON,
                                                       maxAutoCompute=ON)
            vp.view.fitView()
            name = os.path.join(out_dir, "%s_%s" % (prefix, suffix))
            session.printToFile(fileName=name, format=PNG, canvasObjects=(vp,))
            print("wrote %s.png" % name)
        except Exception as exc:                      # keep going on one bad field
            print("SKIPPED %s (%s): %s" % (label, suffix, exc))

    odb.close()
    print("done")


main()
