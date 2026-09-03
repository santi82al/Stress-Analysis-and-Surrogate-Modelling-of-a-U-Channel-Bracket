"""
Render the two Chapter 3 validation figures from the reference ODB.

Chapter 3 validates the analytical model against the reference configuration
(model_0), and carries two figure placeholders for it: the meshed assembly, and
the von Mises field at the end of Load_Step. Both come out of the same solved
ODB, so they are produced here rather than screenshotted, for the same reason
build_stage_shots.py exists -- a hand-taken shot cannot be reproduced after a
re-solve.

Camera and print settings follow odb_screenshot.py so these sit consistently
beside the other Abaqus figures in the thesis.

Run from the repo root:

    abaqus cae noGUI=geometry/validation_shots.py

Optional environment overrides:

    SHOT_OUT       output directory   (default Thesis_LaTex/figures)
    SHOT_PREFIX    filename prefix    (default val)
    SHOT_ODB       source ODB         (default output/model_0/model_0.odb)
    SHOT_SMISES_MAX  legend cap, MPa  (default 44.5, the p95 of this field)
"""

import os

from abaqus import session
from abaqusConstants import (CONTOURS_ON_DEF, UNDEFORMED, INTEGRATION_POINT,
                             INVARIANT, ALL, FEATURE, SOLID, PNG, ON, OFF,
                             EXTRAPOLATE_AVERAGE_COMPUTE)
import visualization                                            # noqa: F401
import displayGroupOdbToolset as dgo

# The bracket is taller than it is wide, so an iso view of it suits a portrait
# frame; the same convention as build_stage_shots.py.
PORTRAIT = (150, 200)
ISO = (1, 1, 1)
UP = (0, 1, 0)

# 95th percentile of S_Mises over the bracket in this ODB. The extremum
# (302.7 MPa) sits on a singular node at the bore edge and, left on auto,
# absorbs the whole colour ramp and flattens the arm to one colour -- the same
# reason the convergence study reports vm_p95 rather than the peak.
DEFAULT_CAP = 44.5


def colour_by_instance(vp):
    try:
        vp.enableMultipleColors()
        vp.setColor(initialColor="#BDBDBD")
        vp.setColor(colorMapping=vp.colorMappings["Part instance"])
        vp.disableMultipleColors()
    except Exception as exc:
        print("per-instance colouring skipped: %s" % exc)


def main():
    out_dir = os.path.abspath(os.environ.get("SHOT_OUT")
                              or os.path.join("Thesis_LaTex", "figures"))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    prefix = os.environ.get("SHOT_PREFIX", "val")
    odb_path = os.path.abspath(os.environ.get("SHOT_ODB") or
                               os.path.join("output", "model_0", "model_0.odb"))
    cap = float(os.environ.get("SHOT_SMISES_MAX", DEFAULT_CAP))

    odb = session.openOdb(name=odb_path, readOnly=True)
    vp = session.Viewport(name="val", origin=(0.0, 0.0),
                          width=PORTRAIT[0], height=PORTRAIT[1])
    vp.makeCurrent()
    vp.setValues(displayedObject=odb)

    session.graphicsOptions.setValues(backgroundStyle=SOLID,
                                      backgroundColor="#FFFFFF")
    session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)

    def write(suffix):
        name = os.path.join(out_dir, "%s_%s" % (prefix, suffix))
        session.printToFile(fileName=name, format=PNG, canvasObjects=(vp,))
        print("wrote %s.png" % name)

    # Both figures describe the END of Load_Step. CAE's default frame is not
    # worth relying on, so pin it explicitly.
    step_names = list(odb.steps.keys())
    step_idx = step_names.index("Load_Step")
    last = len(odb.steps["Load_Step"].frames) - 1
    vp.odbDisplay.setFrame(step=step_idx, frame=last)
    print("frame: step %s (%d), frame %d" % ("Load_Step", step_idx, last))

    # ---------------------------------------------------------------- mesh --
    # The whole assembly, undeformed, every element edge drawn: the figure is
    # about the discretisation, not about a result.
    vp.viewportAnnotationOptions.setValues(triad=ON, legend=OFF, title=OFF,
                                           state=OFF, annotations=OFF,
                                           compass=OFF)
    vp.odbDisplay.display.setValues(plotState=(UNDEFORMED,))
    vp.odbDisplay.commonOptions.setValues(visibleEdges=ALL)
    colour_by_instance(vp)
    vp.view.setViewpoint(viewVector=ISO, cameraUpVector=UP)
    vp.view.fitView()
    write("mesh")

    # Same view framed on the bolted joint, where the bores and the back
    # wall/arm junction are. Fitting on the bolt alone sets the camera, then the
    # full model is restored WITHOUT refitting so the framing survives.
    all_three = ("BASE-1", "BOLT-1", "WALL-1")
    try:
        vp.odbDisplay.displayGroup.replace(
            leaf=dgo.LeafFromPartInstance(partInstanceName=("BOLT-1",)))
        vp.view.fitView()
        vp.view.zoom(float(os.environ.get("SHOT_JOINT_ZOOM", "0.30")))
        vp.odbDisplay.displayGroup.replace(
            leaf=dgo.LeafFromPartInstance(partInstanceName=all_three))
    except Exception as exc:
        print("joint framing skipped (%s)" % exc)
        vp.view.fitView()
    write("mesh_joint")

    # ------------------------------------------------------------- contour --
    # Bracket only. The bolt's contact peak is several times the bracket's and
    # would take over the legend (odb_screenshot.py hits the same problem).
    vp.odbDisplay.displayGroup.replace(
        leaf=dgo.LeafFromPartInstance(partInstanceName=("BASE-1",)))
    vp.viewportAnnotationOptions.setValues(legend=ON)
    vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
    vp.odbDisplay.commonOptions.setValues(visibleEdges=FEATURE)
    vp.odbDisplay.setPrimaryVariable(variableLabel="S",
                                     outputPosition=INTEGRATION_POINT,
                                     refinement=(INVARIANT, "Mises"))

    # CAE averages nodal contributions only where they agree to within 75% by
    # default, so at the bore -- a contact discontinuity -- it leaves them
    # unaveraged and reports 312.3 MPa. _save_results_csv() averages every
    # contribution unconditionally and gets 302.7, which is the number the text
    # quotes. Average at 100% so the figure and the text agree.
    #
    # The order matters too: CAE extrapolates, computes the invariant, then
    # averages the scalars (303.9); _save_results_csv() extrapolates, averages
    # the six tensor components, then computes the invariant (302.7). Ask CAE
    # for the second order so the two agree.
    try:
        vp.odbDisplay.basicOptions.setValues(
            averageElementOutput=True, averagingThreshold=100,
            computeOrder=EXTRAPOLATE_AVERAGE_COMPUTE)
    except Exception as exc:
        print("averaging options not fully set (%s) -- legend max may read "
              "slightly high against the text" % exc)
    try:
        vp.odbDisplay.contourOptions.setValues(showMaxLocation=ON,
                                               showMinLocation=OFF)
    except Exception as exc:
        print("max-location marker skipped: %s" % exc)

    vp.view.setViewpoint(viewVector=ISO, cameraUpVector=UP)
    vp.view.fitView()

    # Capped: everything above the cap goes to a dedicated dark-red swatch, so
    # the arm keeps most of the ramp and stays legible.
    vp.odbDisplay.contourOptions.setValues(minAutoCompute=OFF, minValue=0.0,
                                           maxAutoCompute=OFF, maxValue=cap,
                                           outsideLimitsAboveColor="#8B0000")
    write("vm")

    # Auto-scaled alternative: the peak dominates and the arm flattens to one
    # colour. Kept because it makes the "hot spot belongs to the joint" point
    # bluntly, if that is the reading wanted.
    vp.odbDisplay.contourOptions.setValues(minAutoCompute=ON, maxAutoCompute=ON)
    write("vm_auto")

    odb.close()
    print("done -- %s" % out_dir)


main()
