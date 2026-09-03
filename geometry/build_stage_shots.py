"""
Capture the model_0 build sequence of Sec. 5.3 as figure-ready PNGs.

Section 5.3 of the thesis walks through the eight stages the builder issues, and
carries five figure placeholders for them. Screenshotting those by hand gives a
different camera, window size and background every time, and cannot be redone
after a geometry tweak. This script replays the same build on the reference
parameter set and prints each stage off-screen instead, so the whole set can be
regenerated with one command.

It mirrors the geometry section of build_model_4walls() rather than calling it,
because that function meshes, submits the job and extracts results in one go and
offers no hook to stop at an intermediate feature. The bolt/wall/contact helpers
ARE imported from the builder, so only the bracket-solid sequence is duplicated
here; keep it in step with CAE_models_builder.py if the solid changes.

No job is created and nothing is submitted. The two sectioned views come from the
already-solved output/model_0/model_0.odb, because view cuts are a Visualization
feature and cannot be applied to a CAE assembly.

Run from the repo root:

    abaqus cae noGUI=geometry/build_stage_shots.py

or, to watch it build and keep the model open afterwards for extra manual shots:

    abaqus cae script=geometry/build_stage_shots.py

Optional environment overrides (Abaqus rewrites sys.argv with its own flags, so
configuration comes through the environment -- the same convention run_models.py
and odb_screenshot.py use):

    SHOT_OUT     output directory        (default Thesis_LaTex/figures)
    SHOT_PREFIX  filename prefix         (default build)
    SHOT_COLUMN  model_params.csv column (default model_0)
    SHOT_ODB     solved ODB for sections (default output/model_0/model_0.odb)
"""

import csv
import os

from abaqus import mdb, session
from abaqusConstants import (THREE_D, DEFORMABLE_BODY, SIDE1, BOTTOM, RIGHT,
                             COPLANAR_EDGES, XYPLANE, YZPLANE, MIDDLE_SURFACE,
                             FROM_SECTION, CARTESIAN, HEX_DOMINATED, SWEEP,
                             C3D8R, STANDARD, WHOLE_SURFACE, DISTRIBUTING,
                             UNDEFORMED, FEATURE, SOLID, PNG, ON, OFF)
from mesh import ElemType
import regionToolset

from geometry.CAE_models_builder import add_bolt_and_wall, setup_contact_and_preload

PARAMS = ["b_geo", "m_geo", "n_geo", "t_geo", "f_geo", "e_geo", "i_geo",
          "P_load", "wall_thk"]

MODEL_NAME = "fig_stages"
MESH_DIV = 24.0                    # production default of the builder

# Same camera as odb_screenshot.py: X right, Y up, Z toward the front, so the
# build figures are read the same way as the result contours already in Ch. 5.
ISO_VIEW = (1, 1, 1)
UP = (0, 1, 0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_params(csv_path, column):
    """Pull one model column out of model_params.csv, as run_models.py does."""
    with open(csv_path) as f:
        rows = {}
        for row in csv.DictReader(f):
            key = row["Parameter"].strip().split()[0]
            rows[key] = dict((k.strip(), (v or "").strip())
                             for k, v in row.items())
    missing = [p for p in PARAMS if p not in rows]
    if missing:
        raise SystemExit("missing rows in %s: %s" % (csv_path, missing))
    if column not in rows[PARAMS[0]]:
        raise SystemExit("no column '%s' in %s" % (column, csv_path))
    return dict((p, float(rows[p][column])) for p in PARAMS)


class Shooter(object):
    """Owns the viewport and writes one PNG per call."""

    # The bracket is tall (Y=120 against W=60), so an iso view of it fits a 4:3
    # frame by height and wastes a third of the width. Portrait for those,
    # landscape for the square-on section, which is wider than it is high.
    PORTRAIT = (150, 200)
    LANDSCAPE = (260, 200)

    def __init__(self, out_dir, prefix):
        self.out_dir = out_dir
        self.prefix = prefix
        self.vp = session.Viewport(name="stages", origin=(0.0, 0.0),
                                   width=self.PORTRAIT[0],
                                   height=self.PORTRAIT[1])
        self.vp.makeCurrent()

        # White background, no CAE furniture: these go straight into the report.
        session.graphicsOptions.setValues(backgroundStyle=SOLID,
                                          backgroundColor="#FFFFFF")
        self.vp.viewportAnnotationOptions.setValues(
            triad=ON, legend=OFF, title=OFF, state=OFF, annotations=OFF,
            compass=OFF)
        session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)

    def frame(self, wh):
        try:
            self.vp.setValues(width=wh[0], height=wh[1])
        except Exception as exc:
            print("viewport resize to %sx%s skipped: %s" % (wh[0], wh[1], exc))

    def iso(self):
        self.vp.view.setViewpoint(viewVector=ISO_VIEW, cameraUpVector=UP)
        self.vp.view.fitView()

    def write(self, suffix):
        name = os.path.join(self.out_dir, "%s_%s" % (self.prefix, suffix))
        session.printToFile(fileName=name, format=PNG, canvasObjects=(self.vp,))
        print("wrote %s.png" % name)

    def shot_part(self, part, suffix, mesh=OFF, datums=OFF):
        self.vp.setValues(displayedObject=part)
        self.vp.partDisplay.setValues(mesh=mesh)
        try:
            self.vp.partDisplay.geometryOptions.setValues(datumPlanes=datums)
        except Exception as exc:            # display option, never worth failing on
            print("datum-plane display skipped: %s" % exc)
        self.iso()
        self.write(suffix)


def colour_by_instance(vp):
    """Give each part instance its own colour so the three parts read apart."""
    try:
        vp.enableMultipleColors()
        vp.setColor(initialColor="#BDBDBD")
        vp.setColor(colorMapping=vp.colorMappings["Part instance"])
        vp.disableMultipleColors()
    except Exception as exc:
        print("per-instance colouring skipped: %s" % exc)


# ---------------------------------------------------------------------------
# stages 1-8, mirroring build_model_4walls()
# ---------------------------------------------------------------------------
def build_with_shots(shooter, p):
    b_geo, m_geo, n_geo = p["b_geo"], p["m_geo"], p["n_geo"]
    t_geo, f_geo, e_geo = p["t_geo"], p["f_geo"], p["e_geo"]
    i_geo, P_load, wall_thk = p["i_geo"], p["P_load"], p["wall_thk"]

    # -- stage 1: derived dimensions --------------------------------------
    Y_geo = 4 * m_geo - 2 * n_geo + 4 * e_geo
    Z_geo = 2 * m_geo
    W_geo = b_geo + 2 * t_geo
    if Y_geo <= 0:
        raise SystemExit("Invalid geometry: Y_geo=%.1f" % Y_geo)
    print("W=%.2f  Y=%.2f  Z=%.2f" % (W_geo, Y_geo, Z_geo))

    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]
    if "Model-1" in mdb.models:
        mdb.models.changeKey(fromName="Model-1", toName=MODEL_NAME)
    else:
        mdb.Model(name=MODEL_NAME)
    model = mdb.models[MODEL_NAME]

    # -- stage 2a: back wall ----------------------------------------------
    sk = model.ConstrainedSketch(name="base_profile", sheetSize=200.0)
    sk.rectangle(point1=(0.0, 0.0), point2=(W_geo, Y_geo))
    part = model.Part(name="base", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=f_geo)
    shooter.shot_part(part, "geom_a_backwall")

    # -- stage 2b: U-channel arm ------------------------------------------
    flange_face = part.faces.findAt(((W_geo / 2, Y_geo / 2, f_geo),))[0].index
    flange_up_edge = part.edges.findAt(((W_geo / 2, 0, f_geo),))[0].index

    sk2 = model.ConstrainedSketch(
        name="wall_profile", sheetSize=200.0,
        transform=part.MakeSketchTransform(
            sketchPlane=part.faces[flange_face],
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part.edges[flange_up_edge],
            sketchOrientation=BOTTOM,
            origin=(W_geo / 2, Y_geo / 2, f_geo)))
    part.projectReferencesOntoSketch(sketch=sk2, filter=COPLANAR_EDGES)

    xL, xR = -W_geo / 2, W_geo / 2
    yT, yB = Y_geo / 2, -Y_geo / 2
    xLi, xRi = xL + t_geo, xR - t_geo
    yBi = yB + e_geo

    sk2.Line((xL, yT), (xL, yB))
    sk2.Line((xL, yB), (xR, yB))
    sk2.Line((xR, yB), (xR, yT))
    sk2.Line(point1=(xR, yT), point2=(xRi, yT))
    sk2.Line(point1=(xRi, yT), point2=(xRi, yBi))
    sk2.Line(point1=(xRi, yBi), point2=(xLi, yBi))
    sk2.Line(point1=(xLi, yBi), point2=(xLi, yT))
    sk2.Line(point1=(xLi, yT), point2=(xL, yT))

    part.SolidExtrude(
        sketch=sk2,
        sketchPlane=part.faces[flange_face],
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part.edges[flange_up_edge],
        sketchOrientation=BOTTOM,
        depth=Z_geo - f_geo)
    shooter.shot_part(part, "geom_b_arm")

    # -- stage 2c: triangular lightening cut ------------------------------
    cut_face = part.faces.findAt(((W_geo, Y_geo / 2, Z_geo / 2),))[0].index
    cut_up_edge = part.edges.findAt(((W_geo, 0, Z_geo / 2),))[0].index

    sk3 = model.ConstrainedSketch(
        name="cut_profile", sheetSize=200.0,
        transform=part.MakeSketchTransform(
            sketchPlane=part.faces[cut_face],
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part.edges[cut_up_edge],
            sketchOrientation=BOTTOM,
            origin=(W_geo / 2, Y_geo / 2, Z_geo / 2)))
    part.projectReferencesOntoSketch(sketch=sk3, filter=COPLANAR_EDGES)

    x_left = -Z_geo / 2
    x_right = Z_geo / 2 - f_geo * 2
    y_top = Y_geo / 2
    y_bottom = -Y_geo / 2 + e_geo * 2

    sk3.Line((x_left, y_top), (x_left, y_bottom))
    sk3.Line((x_left, y_bottom), (x_right, y_top))
    sk3.Line((x_right, y_top), (x_left, y_top))

    part.CutExtrude(
        sketch=sk3,
        sketchPlane=part.faces[cut_face],
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part.edges[cut_up_edge],
        sketchOrientation=BOTTOM)
    shooter.shot_part(part, "geom_c_cut")

    # -- stage 2d: the two bores ------------------------------------------
    base_bore_face = part.faces.findAt(((W_geo / 2, 0, Z_geo / 2),))[0].index
    base_bore_up_edge = part.edges.findAt(((W_geo / 2, 0, 0),))[0].index

    sk_bore_base = model.ConstrainedSketch(
        name="bore_base", sheetSize=200.0,
        transform=part.MakeSketchTransform(
            sketchPlane=part.faces[base_bore_face],
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part.edges[base_bore_up_edge],
            sketchOrientation=BOTTOM,
            origin=(W_geo / 2, 0.0, 0.0)))
    part.projectReferencesOntoSketch(sketch=sk_bore_base, filter=COPLANAR_EDGES)
    sk_bore_base.CircleByCenterPerimeter(center=(0.0, -m_geo),
                                         point1=(i_geo, -m_geo))
    part.CutExtrude(
        sketch=sk_bore_base,
        sketchPlane=part.faces[base_bore_face],
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part.edges[base_bore_up_edge],
        sketchOrientation=BOTTOM)

    wall_bore_face = part.faces.findAt(((W_geo / 2, Y_geo / 2, 0),))[0].index
    wall_bore_up_edge = part.edges.findAt(((W_geo / 2, Y_geo, 0),))[0].index

    sk_bore_wall = model.ConstrainedSketch(
        name="bore_wall", sheetSize=200.0,
        transform=part.MakeSketchTransform(
            sketchPlane=part.faces[wall_bore_face],
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part.edges[wall_bore_up_edge],
            sketchOrientation=RIGHT,
            origin=(W_geo / 2, Y_geo / 2, 0.0)))
    part.projectReferencesOntoSketch(sketch=sk_bore_wall, filter=COPLANAR_EDGES)
    sk_bore_wall.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(i_geo, 0.0))
    part.CutExtrude(
        sketch=sk_bore_wall,
        sketchPlane=part.faces[wall_bore_face],
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part.edges[wall_bore_up_edge],
        sketchOrientation=RIGHT)
    shooter.shot_part(part, "geom_d_bores")

    # -- stage 3: material and partitioning -------------------------------
    model.Material(name="Al_7075")
    model.materials["Al_7075"].Elastic(table=((70000.0, 0.33),))

    dz = part.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE,
                                         offset=f_geo).id
    dxL = part.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE,
                                          offset=t_geo).id
    dxR = part.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE,
                                          offset=W_geo - t_geo).id
    for did in (dz, dxL, dxR):
        try:
            part.PartitionCellByDatumPlane(datumPlane=part.datums[did],
                                           cells=part.cells)
        except Exception as err:
            print("Partition skipped (%s): %s" % (did, err))

    model.HomogeneousSolidSection(name="Section-1", material="Al_7075")
    part.SectionAssignment(
        region=part.Set(cells=part.cells, name="Set-1"),
        sectionName="Section-1", offset=0.0, offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION)
    shooter.shot_part(part, "partition", datums=ON)

    # -- stages 4-7: bolt, wall, assembly, contact, steps ------------------
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    assembly.Instance(name="base-1", part=part, dependent=ON)

    add_bolt_and_wall(model, W_geo, Y_geo, i_geo, f_geo, wall_thk)
    setup_contact_and_preload(model, W_geo, Y_geo, i_geo, f_geo, wall_thk)

    shooter.vp.setValues(displayedObject=assembly)
    # The partition datums stay switched on into the assembly view, where they
    # are only clutter -- the cutting planes belong to the partition figure.
    try:
        shooter.vp.assemblyDisplay.geometryOptions.setValues(
            datumPlanes=OFF, datumAxes=OFF, datumPoints=OFF,
            datumCoordSystems=OFF)
    except Exception as exc:
        print("assembly datum display skipped: %s" % exc)
    colour_by_instance(shooter.vp)
    shooter.iso()
    shooter.write("assembly_iso")

    # -- stage 8: mesh -----------------------------------------------------
    cells = part.cells
    part.setMeshControls(regions=cells, elemShape=HEX_DOMINATED, technique=SWEEP)
    part.setElementType(
        regions=(cells,),
        elemTypes=(ElemType(elemCode=C3D8R, elemLibrary=STANDARD),))
    mesh_size = min(W_geo, Y_geo, Z_geo) / MESH_DIV
    part.seedPart(size=mesh_size, deviationFactor=0.05, minSizeFactor=0.1)
    part.generateMesh()
    print("mesh_div=%.2f  seed=%.4f  elements=%d"
          % (MESH_DIV, mesh_size, len(part.elements)))

    shooter.shot_part(part, "mesh", mesh=ON)

    return W_geo, Y_geo, Z_geo


# ---------------------------------------------------------------------------
# sectioned views, from the solved ODB (view cuts are Visualization-only)
# ---------------------------------------------------------------------------
def section_shots(shooter, odb_path, W_geo):
    if not os.path.isfile(odb_path):
        print("no ODB at %s -- skipping the two sectioned views. Solve model_0 "
              "first, or point SHOT_ODB at an equivalent .odb." % odb_path)
        return

    # A viewport rejects an Odb as displayedObject until visualization is
    # loaded, and displayGroupOdbToolset only resolves after it.
    import visualization                                        # noqa: F401
    import displayGroupOdbToolset as dgo

    vp = shooter.vp
    odb = session.openOdb(name=odb_path, readOnly=True)
    vp.setValues(displayedObject=odb)
    vp.odbDisplay.display.setValues(plotState=(UNDEFORMED,))
    vp.odbDisplay.commonOptions.setValues(visibleEdges=FEATURE)
    colour_by_instance(vp)

    # Cut on the plane X = W/2, which carries the bolt axis, and keep the half
    # below it so the camera looks into the section from +X.
    try:
        vp.odbDisplay.setValues(viewCutNames=("X-Plane",), viewCut=ON)
        vp.odbDisplay.viewCuts["X-Plane"].setValues(
            position=W_geo / 2.0,
            showModelOnCut=True, showModelBelowCut=True, showModelAboveCut=False)
    except Exception as exc:
        print("view cut failed (%s) -- sections will show the whole model" % exc)

    # Three-quarter section: the parts still read as solids.
    vp.view.setViewpoint(viewVector=(1, 0.45, 1.1), cameraUpVector=UP)
    vp.view.fitView()
    shooter.write("assembly")

    # Square-on section framed about the bolt. fitView on the bolt alone sets the
    # camera, then the full model is put back WITHOUT refitting, so the framing
    # stays on the joint instead of on the whole bracket.
    all_three = ("BASE-1", "BOLT-1", "WALL-1")
    shooter.frame(Shooter.LANDSCAPE)
    try:
        vp.odbDisplay.displayGroup.replace(
            leaf=dgo.LeafFromPartInstance(partInstanceName=("BOLT-1",)))
        vp.view.setViewpoint(viewVector=(1, 0, 0), cameraUpVector=UP)
        vp.view.fitView()
        vp.view.zoom(float(os.environ.get("SHOT_CONTACT_ZOOM", "0.32")))
        vp.odbDisplay.displayGroup.replace(
            leaf=dgo.LeafFromPartInstance(partInstanceName=all_three))
    except Exception as exc:
        print("bolt framing skipped (%s) -- falling back to a full fit" % exc)
        vp.view.setViewpoint(viewVector=(1, 0, 0), cameraUpVector=UP)
        vp.view.fitView()
    shooter.write("contact")

    odb.close()


def main():
    root = os.getcwd()
    out_dir = os.path.abspath(os.environ.get("SHOT_OUT")
                              or os.path.join("Thesis_LaTex", "figures"))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    prefix = os.environ.get("SHOT_PREFIX", "build")
    column = os.environ.get("SHOT_COLUMN", "model_0")
    odb_path = os.path.abspath(os.environ.get("SHOT_ODB") or
                               os.path.join("output", "model_0", "model_0.odb"))

    p = read_params(os.path.join(root, "model_params.csv"), column)
    print("reference set %s: %s" % (column, p))

    shooter = Shooter(out_dir, prefix)
    W_geo, _, _ = build_with_shots(shooter, p)
    section_shots(shooter, odb_path, W_geo)
    print("done -- %s" % out_dir)


main()
