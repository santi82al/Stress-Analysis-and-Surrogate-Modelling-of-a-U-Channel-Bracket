# -*- coding: mbcs -*-                     # Encoding used by Abaqus Python environment

from tkinter import ON
from math import pi
from xml.parsers.expat import model


from abaqus import *
from abaqusConstants import *
from part import *                         # Import Abaqus part module
from material import *                     # Import material definition tools
from section import *                      # Import section definition tools
from assembly import *                     # Import assembly module
from step import *                         # Import analysis step module
from interaction import *                  # Import interaction tools
from load import *                         # Import loads and boundary conditions
from sketch import *                       # Import sketching tools
from mesh import *                         # Import meshing tools
import regionToolset
import csv
import os




# =============================================================================
# COORDINATE SYSTEM REFERENCE
# =============================================================================
#
#   BaseSolidExtrude: rectangle (0,0) -> (W_geo, Y_geo) in the XY plane,
#   extruded in the +Z direction by depth f_geo.
#
#   After extrusion the back wall solid occupies:
#       X : [0,    W_geo]   (width)
#       Y : [0,    Y_geo]   (height)
#       Z : [0,    f_geo]   (depth, back wall thickness)
#
#   The flange SolidExtrude then extends the U-channel arm in +Z:
#       Z : [f_geo, n_geo]  (arm depth beyond the back wall)
#
#   Key planes:
#       Z = 0       - back face of the fitting (mounting face)
#       Z = f_geo   - front face of back wall / base of arm
#       Z = n_geo   - open end of the arm
#       Y = 0       - bottom (ground) wall of the arm
#       Y = Y_geo   - top (open) face of the arm
#       X = 0       - left outer face of arm
#       X = W_geo   - right outer face of arm
#
# =============================================================================



 
# =============================================================================
# RESULTS EXTRACTION
# =============================================================================
def _save_results_csv(odb, model_name):
    """
    Extract full nodal field results from the last frame of 'Load_Step'
    and write to {model_name}_results.csv.

    Stress invariants (Von Mises, principal) are computed manually from the
    6-component tensor [S11,S22,S33,S12,S13,S23] to avoid Abaqus API issues
    with getScalarField(invariant=...) on nodal-extrapolated fields.

    Columns
    -------
    Node  X  Y  Z  U1  U2  U3  U_mag  RF1  RF2  RF3  RF_mag
    S_Mises  S_MaxPrincipal  S_MinPrincipal
    """
    print("*** RESULTS EXTRACTION: model=%s, cwd=%s ***" % (model_name, os.getcwd()))
    from math import acos, cos, pi as _pi

    def _invariants(data):
        """Von Mises + max/min principal from [S11,S22,S33,S12,S13,S23]."""
        s11,s22,s33,s12,s13,s23 = [float(x) for x in data]

        # Von Mises
        mises = (((s11-s22)**2+(s22-s33)**2+(s33-s11)**2)/2.0
                 + 3.0*(s12**2+s13**2+s23**2))**0.5

        # Principal stresses - eigenvalues of symmetric 3x3
        # (Smith, Boyle & Sherrill algorithm for real symmetric matrices)
        p1 = s12**2 + s13**2 + s23**2
        if p1 < 1e-14:                         # already diagonal
            eigs = sorted([s11, s22, s33], reverse=True)
            return mises, eigs[0], eigs[2]

        q  = (s11+s22+s33)/3.0
        p2 = (s11-q)**2+(s22-q)**2+(s33-q)**2+2.0*p1
        p  = (p2/6.0)**0.5

        b11=(s11-q)/p; b22=(s22-q)/p; b33=(s33-q)/p
        b12=s12/p;     b13=s13/p;     b23=s23/p

        r = (b11*(b22*b33-b23**2)
             - b12*(b12*b33-b23*b13)
             + b13*(b12*b23-b22*b13)) / 2.0
        r = max(-1.0, min(1.0, r))   # clamp for numerical safety

        phi  = acos(r) / 3.0
        eig1 = q + 2.0*p*cos(phi)
        eig3 = q + 2.0*p*cos(phi + 2.0*_pi/3.0)
        eig2 = 3.0*q - eig1 - eig3

        return mises, max(eig1,eig2,eig3), min(eig1,eig2,eig3)

    step = odb.steps['Load_Step']
    nf   = len(step.frames)
    if nf == 0:
        raise ValueError('Load_Step has 0 frames - check field output requests or solver convergence')
    frame    = step.frames[nf - 1]
    instance = odb.rootAssembly.instances['BASE-1']

    # -- Node coordinates ------------------------------------------------------
    coords  = {n.label: n.coordinates for n in instance.nodes}

    # -- Displacement - naturally nodal ------------------------------------
    # getSubset(region=instance) is required: node labels are per-instance,
    # not globally unique, so an unscoped lookup can silently pick up a
    # same-numbered node from 'bolt-1' or 'wall-1' instead of 'BASE-1'.
    U_field = frame.fieldOutputs['U'].getSubset(region=instance)
    U_vals  = {v.nodeLabel: v for v in U_field.values}

    # -- Reaction force - naturally nodal (non-zero only at BC nodes) ------
    RF_field = frame.fieldOutputs['RF'].getSubset(region=instance)
    RF_vals  = {v.nodeLabel: v for v in RF_field.values}

    # -- Stress - extrapolate to nodes (ELEMENT_NODAL), average per node ---
    # Each node gets one ELEMENT_NODAL value per contributing element;
    # average them to reproduce the usual "extrapolate and average" nodal
    # stress, then feed the 6-component tensor to _invariants().
    S_field = frame.fieldOutputs['S'].getSubset(
        region=instance, position=ELEMENT_NODAL)
    S_sums, S_counts = {}, {}
    for v in S_field.values:
        label = v.nodeLabel
        acc = S_sums.setdefault(label, [0.0] * 6)
        for i in range(6):
            acc[i] += v.data[i]
        S_counts[label] = S_counts.get(label, 0) + 1
    S_vals = {label: [s / S_counts[label] for s in acc]
              for label, acc in S_sums.items()}

    # -- Write CSV -------------------------------------------------------------
    csv_path = model_name + '_results.csv'
    print("DEBUG: CSV path = %s (abs = %s)" % (csv_path, os.path.abspath(csv_path)))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Node', 'X', 'Y', 'Z',
            'U1', 'U2', 'U3', 'U_mag',
            'RF1', 'RF2', 'RF3', 'RF_mag',
            'S_Mises', 'S_MaxPrincipal', 'S_MinPrincipal'
        ])

        for label in sorted(coords.keys()):
            x, y, z = coords[label]

            u = U_vals.get(label)
            u1,u2,u3 = (u.data[0],u.data[1],u.data[2]) if u else (0.,0.,0.)
            u_mag = u.magnitude if u else 0.

            rf = RF_vals.get(label)
            rf1,rf2,rf3 = (rf.data[0],rf.data[1],rf.data[2]) if rf else (0.,0.,0.)
            rf_mag = (rf1**2+rf2**2+rf3**2)**0.5

            s_data = S_vals.get(label)
            if s_data:
                s_mises, s_maxp, s_minp = _invariants(s_data)
            else:
                s_mises, s_maxp, s_minp = 0., 0., 0.

            writer.writerow([label, x, y, z,
                             u1, u2, u3, u_mag,
                             rf1, rf2, rf3, rf_mag,
                             s_mises, s_maxp, s_minp])
 
    import time; time.sleep(0.1)  # Ensure file is fully written before move
    print('Results saved: %s' % os.path.abspath(csv_path))



# =============================================================================
# BOLT + RIGID SUPPORTING WALL  - parametric builders
# =============================================================================

HEAD_DIA_RATIO    = 1.6    # (head/nut diameter) / shank diameter
HEAD_HT_RATIO     = 0.7    # (head/nut height)   / shank diameter
FRICTION_COEFF    = 0.15   # Coulomb mu  (steel-on-Al)
BOLT_PROOF_STRESS = 720.0  # MPa - grade 8.8 proof stress
PRELOAD_RATIO     = 0.60   # fraction of proof load applied as preload


def build_bolt_part(model, i_geo, f_geo, wall_thk, shank_clearance=0.0):
    """
    Build the stepped bolt (nut | shank | head) as a single revolved solid.
    Three datum-plane partitions create four sweepable hex cells.

    Part-local axial coordinate v (sketch-Y after revolve):
        v = 0               nut back face
        v = hh              nut bearing face / shank start
        v = hh + wall_thk   preload cut plane  (-> global Z = 0)
        v = hh + Ls         head bearing face  (Ls = f_geo + wall_thk)
        v = 2*hh + Ls       head top face
    """
    d   = 2.0 * i_geo
    rs  = i_geo - shank_clearance
    rh  = HEAD_DIA_RATIO * d / 2.0
    hh  = HEAD_HT_RATIO  * d
    Ls  = f_geo + wall_thk

    sk = model.ConstrainedSketch(name='bolt_profile', sheetSize=4.0 * (2*hh + Ls))
    sk.Line(point1=(0.0,  0.0),          point2=(rh,  0.0))
    sk.Line(point1=(rh,   0.0),          point2=(rh,  hh))
    sk.Line(point1=(rh,   hh),           point2=(rs,  hh))
    sk.Line(point1=(rs,   hh),           point2=(rs,  hh + Ls))
    sk.Line(point1=(rs,   hh + Ls),      point2=(rh,  hh + Ls))
    sk.Line(point1=(rh,   hh + Ls),      point2=(rh,  2*hh + Ls))
    sk.Line(point1=(rh,   2*hh + Ls),    point2=(0.0, 2*hh + Ls))
    sk.Line(point1=(0.0,  2*hh + Ls),    point2=(0.0, 0.0))
    axis = sk.ConstructionLine(point1=(0.0, -1.0), point2=(0.0, 1.0))
    sk.assignCenterline(line=axis)

    part = model.Part(name='bolt', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidRevolve(sketch=sk, angle=360.0, flipRevolveDirection=OFF)

    # Three XZPLANE cuts: nut|shank, preload cut, shank|head
    for v_cut in (hh, hh + wall_thk, hh + Ls):
        try:
            dp = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=v_cut)
            part.PartitionCellByDatumPlane(datumPlane=part.datums[dp.id], cells=part.cells)
        except Exception as e:
            print('Bolt partition at v=%.3f skipped: %s' % (v_cut, e))

    if 'Steel_bolt' not in model.materials.keys():
        model.Material(name='Steel_bolt')
        model.materials['Steel_bolt'].Elastic(table=((210000.0, 0.30),))
    if 'Sec_bolt' not in model.sections.keys():
        model.HomogeneousSolidSection(name='Sec_bolt', material='Steel_bolt')
    part.SectionAssignment(
        region=part.Set(cells=part.cells, name='bolt_all'),
        sectionName='Sec_bolt',
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION)

    part.setMeshControls(regions=part.cells, elemShape=HEX_DOMINATED, technique=SWEEP)
    part.setElementType(
        regions=(part.cells,),
        elemTypes=(ElemType(elemCode=C3D8R, elemLibrary=STANDARD),
                   ElemType(elemCode=C3D6,  elemLibrary=STANDARD)))
    seed_bolt = max(i_geo / 3.0, 1.0)
    part.seedPart(size=seed_bolt, deviationFactor=0.05, minSizeFactor=0.1)
    part.generateMesh()

    return part, hh


def build_wall_part(model, W_geo, Y_geo, i_geo, wall_thk):
    """
    Rigid supporting wall: solid block X[0,W_geo] x Y[0,Y_geo] x Z[0,wall_thk]
    with a coaxial through-bore of radius i_geo.
    Positioned to global Z[-wall_thk, 0] by add_bolt_and_wall().
    """
    sk = model.ConstrainedSketch(name='wall_profile_solid',
                                 sheetSize=2.0 * max(W_geo, Y_geo))
    sk.rectangle(point1=(0.0, 0.0), point2=(W_geo, Y_geo))

    part = model.Part(name='wall', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=wall_thk)

    front_face_idx = part.faces.findAt(((W_geo/2, Y_geo/2, wall_thk),))[0].index
    up_edge_idx    = part.edges.findAt(((W_geo/2, Y_geo,   wall_thk),))[0].index

    sk_bore = model.ConstrainedSketch(
        name='wall_bore',
        sheetSize=2.0 * max(W_geo, Y_geo),
        transform=part.MakeSketchTransform(
            sketchPlane=part.faces[front_face_idx],
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part.edges[up_edge_idx],
            sketchOrientation=RIGHT,
            origin=(W_geo/2, Y_geo/2, wall_thk)))
    part.projectReferencesOntoSketch(sketch=sk_bore, filter=COPLANAR_EDGES)
    sk_bore.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(i_geo, 0.0))
    part.CutExtrude(
        sketch=sk_bore,
        sketchPlane=part.faces[front_face_idx],
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part.edges[up_edge_idx],
        sketchOrientation=RIGHT)

    if 'Steel_bolt' not in model.materials.keys():
        model.Material(name='Steel_bolt')
        model.materials['Steel_bolt'].Elastic(table=((210000.0, 0.30),))
    if 'Sec_bolt' not in model.sections.keys():
        model.HomogeneousSolidSection(name='Sec_bolt', material='Steel_bolt')
    part.SectionAssignment(
        region=part.Set(cells=part.cells, name='wall_all'),
        sectionName='Sec_bolt',
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION)

    # Free tet mesh - rigid body, only geometry matters for contact
    part.setElementType(
        regions=(part.cells,),
        elemTypes=(ElemType(elemCode=C3D4, elemLibrary=STANDARD),))
    part.seedPart(size=i_geo, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    return part


def add_bolt_and_wall(model, W_geo, Y_geo, i_geo, f_geo, wall_thk,
                      shank_clearance=0.0):
    """
    Build + instance + position bolt and wall; make wall a fixed rigid body.
    Call AFTER assembly.Instance(name='base-1') and BEFORE setup_contact_and_preload().
    """
    d  = 2.0 * i_geo
    rh = HEAD_DIA_RATIO * d / 2.0
    hh = HEAD_HT_RATIO  * d

    assembly = model.rootAssembly

    bolt, _ = build_bolt_part(model, i_geo, f_geo, wall_thk, shank_clearance)
    wall     = build_wall_part(model, W_geo, Y_geo, i_geo, wall_thk)

    # Bolt: revolve axis is part-local Y; rotate +90 deg about global X -> part-Y maps to +Z
    assembly.Instance(name='bolt-1', part=bolt, dependent=ON)
    assembly.rotate(instanceList=('bolt-1',),
                    axisPoint=(0.0, 0.0, 0.0),
                    axisDirection=(1.0, 0.0, 0.0),
                    angle=90.0)
    assembly.translate(instanceList=('bolt-1',),
                       vector=(W_geo/2.0, Y_geo/2.0, -(wall_thk + hh)))

    # Wall: local Z[0,wall_thk] -> global Z[-wall_thk, 0]
    assembly.Instance(name='wall-1', part=wall, dependent=ON)
    assembly.translate(instanceList=('wall-1',),
                       vector=(0.0, 0.0, -wall_thk))

    rp_wall = assembly.ReferencePoint(point=(W_geo/2.0, Y_geo/2.0, -wall_thk/2.0))
    rp_wall_region = regionToolset.Region(
        referencePoints=(assembly.referencePoints[rp_wall.id],))
    model.RigidBody(
        name='WallRigid',
        refPointRegion=rp_wall_region,
        bodyRegion=regionToolset.Region(cells=assembly.instances['wall-1'].cells))
    model.DisplacementBC(
        name='wall_fixed',
        createStepName='Initial',
        region=rp_wall_region,
        u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)

    return {'bolt_head_height': hh, 'head_radius': rh}


def setup_contact_and_preload(model, W_geo, Y_geo, i_geo, f_geo, wall_thk,
                               preload_force=None):
    """
    (1) Create steps:  Initial -> Preload_Step -> Load_Step  (nlgeom=ON)
    (2) One shared contact property: hard normal + penalty Coulomb friction
    (3) Four surface-to-surface contact pairs:
            Cont_ShankBracketBore  shank OD vs bracket bore ID   Z[0, f_geo]
            Cont_ShankWallBore     shank OD vs wall bore ID       Z[-wall_thk, 0]
            Cont_HeadPad           head underside vs bracket pad  Z = f_geo
            Cont_NutWall           nut underside vs wall back     Z = -wall_thk
    (4) Pre-tension section through shank at Z = 0
    (5) Preload force in Preload_Step; fix length in Load_Step

    Call AFTER add_bolt_and_wall(). The line
        model.StaticStep(name='Load_Step', previous='Initial')
    must be REMOVED from build_model_4walls() - this function creates both steps.
    """
    d     = 2.0 * i_geo
    rs    = i_geo
    rh    = HEAD_DIA_RATIO * d / 2.0
    hh    = HEAD_HT_RATIO  * d
    r_mid = (rs + rh) / 2.0

    assembly = model.rootAssembly
    bracket  = assembly.instances['base-1']
    bolt     = assembly.instances['bolt-1']
    wall     = assembly.instances['wall-1']

    # -- Steps ----------------------------------------------------------------
    # Automatic stabilization on the PRELOAD step. The bracket mount face and the
    # rigid wall front face are coincident at Z=0 (zero initial gap), so seating
    # the Cont_BracketWall pair as preload is applied makes it chatter: the
    # bracket has a near rigid-body mode until the clamp locks. Without damping
    # this pair dominates the residual (contact-force errors ~100x the bolt pairs)
    # and tips some geometries into 'numerical singularity / too many attempts'
    # in Preload_Step, increment 1 (observed on model_3 and model_7; model_4
    # barely converged in ~19 increments). A small dissipated-energy fraction
    # (same magnitude already used on Load_Step) bleeds off that transient and is
    # negligible for the final result - verify ALLSD << ALLIE in the .dat.
    # initialInc lowered 0.1 -> 0.02: a broad 100-point LHS at
    # div=24 left 8 borderline geometries diverging here at t~0.913 as the
    # bracket/wall seat snapped open<->closed across a single large increment.
    # Seating the clamp over more, smaller increments resolves that transition.
    model.StaticStep(
        name='Preload_Step', previous='Initial',
        description='Bolt pre-tension application',
        maxNumInc=100, initialInc=0.02, minInc=1e-5, maxInc=1.0, nlgeom=ON,
        stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
        stabilizationMagnitude=2e-4,
        adaptiveDampingRatio=0.05,
        continueDampingFactors=False)
    model.StaticStep(
        name='Load_Step', previous='Preload_Step',
        description='Service load application',
        maxNumInc=500, initialInc=0.02, minInc=1e-8, maxInc=0.2, nlgeom=ON,
        stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
        stabilizationMagnitude=2e-4,
        adaptiveDampingRatio=0.05,
        continueDampingFactors=False)

    # Patch all auto-created field output requests to write every increment
    # and cover the correct variables. F-Output-1 is created for Preload_Step
    # when that step is built; patching it here ensures it propagates to Load_Step.
    for _req in model.fieldOutputRequests.values():
        try:
            _req.setValues(variables=('S', 'U', 'RF'), frequency=1)
        except Exception as _e:
            print('WARNING: could not patch field output request: %s' % _e)
    # As a second safety, an explicit request anchored to Load_Step
    try:
        model.FieldOutputRequest(
            name='F-Output-Load',
            createStepName='Load_Step',
            variables=('S', 'U', 'RF'),
            frequency=1)
    except Exception as _e:
        print('WARNING: F-Output-Load creation failed: %s' % _e)

    # -- Contact property -----------------------------------------------------
    model.ContactProperty('BoltContactProp')
    model.interactionProperties['BoltContactProp'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON)
    model.interactionProperties['BoltContactProp'].TangentialBehavior(
        formulation=PENALTY,
        maximumElasticSlip=FRACTION,
        fraction=0.005,
        table=((FRICTION_COEFF,),))

    # -- Contact surfaces -----------------------------------------------------
    s_shank_upper = assembly.Surface(
        name='Surf_ShankUpper',
        side1Faces=bolt.faces.findAt(((W_geo/2 + rs, Y_geo/2, f_geo * 0.5),)))
    s_bracket_bore = assembly.Surface(
        name='Surf_BracketBore',
        side1Faces=bracket.faces.findAt(((W_geo/2, Y_geo/2 - i_geo, f_geo * 0.5),)))
    s_shank_lower = assembly.Surface(
        name='Surf_ShankLower',
        side1Faces=bolt.faces.findAt(((W_geo/2 + rs, Y_geo/2, -wall_thk * 0.5),)))
    s_wall_bore = assembly.Surface(
        name='Surf_WallBore',
        side1Faces=wall.faces.findAt(((W_geo/2, Y_geo/2 - i_geo, -wall_thk * 0.5),)))
    s_head_under = assembly.Surface(
        name='Surf_HeadUnder',
        side1Faces=bolt.faces.findAt(((W_geo/2 + r_mid, Y_geo/2, f_geo),)))
    s_bracket_pad = assembly.Surface(
        name='Surf_BracketPad',
        side1Faces=bracket.faces.findAt(((W_geo/2 + r_mid, Y_geo/2, f_geo),)))
    s_nut_under = assembly.Surface(
        name='Surf_NutUnder',
        side1Faces=bolt.faces.findAt(((W_geo/2 + r_mid, Y_geo/2, -wall_thk),)))
    s_wall_back = assembly.Surface(
        name='Surf_WallBack',
        side1Faces=wall.faces.findAt(((W_geo/2 + r_mid, Y_geo/2, -wall_thk),)))

    # -- 3i.  Bracket mounting face vs wall front face  (Z = 0) ---------------
    # This is the clamping bearing contact: bolt head pushes bracket in -Z;
    # wall front face reacts in +Z. Without this the bracket has a Z rigid-body mode.
    _tol = 1e-3
    s_bracket_mount = assembly.Surface(
        name='Surf_BracketMount',
        side1Faces=bracket.faces.getByBoundingBox(
            xMin=-_tol, xMax=W_geo+_tol,
            yMin=-_tol, yMax=Y_geo+_tol,
            zMin=-_tol, zMax=_tol))
    s_wall_front = assembly.Surface(
        name='Surf_WallFront',
        side1Faces=wall.faces.getByBoundingBox(
            xMin=-_tol, xMax=W_geo+_tol,
            yMin=-_tol, yMax=Y_geo+_tol,
            zMin=-_tol, zMax=_tol))

    # -- Contact pairs ---------------------------------------------------------
    model.SurfaceToSurfaceContactStd(
        name='Cont_ShankBracketBore', createStepName='Initial',
        main=s_shank_upper, secondary=s_bracket_bore,
        sliding=FINITE,
        interactionProperty='BoltContactProp',
        adjustMethod=OVERCLOSED)
    model.SurfaceToSurfaceContactStd(
        name='Cont_ShankWallBore', createStepName='Initial',
        main=s_wall_bore, secondary=s_shank_lower,
        sliding=FINITE,
        interactionProperty='BoltContactProp',
        adjustMethod=OVERCLOSED)
    model.SurfaceToSurfaceContactStd(
        name='Cont_HeadPad', createStepName='Initial',
        main=s_head_under, secondary=s_bracket_pad,
        sliding=SMALL,
        interactionProperty='BoltContactProp',
        adjustMethod=OVERCLOSED)
    model.SurfaceToSurfaceContactStd(
        name='Cont_NutWall', createStepName='Initial',
        main=s_wall_back, secondary=s_nut_under,
        sliding=SMALL,
        interactionProperty='BoltContactProp',
        adjustMethod=OVERCLOSED)
    model.SurfaceToSurfaceContactStd(
        name='Cont_BracketWall', createStepName='Initial',
        main=s_wall_front, secondary=s_bracket_mount,
        sliding=SMALL,
        interactionProperty='BoltContactProp',
        adjustMethod=OVERCLOSED)

    # -- Automatic contact stabilization on the clamp pairs -------------------
    # The bracket is held only by contact until the bolt clamp locks, so it has a
    # near rigid-body mode in Z during preload. The global dissipated-energy
    # damping above is calibrated per-mesh on increment 1 and under-damps that
    # mode at some densities (div=30 diverged in Preload_Step, inc 2 - residual
    # force GREW while the contact constraints themselves converged). Automatic
    # contact stabilization damps relative CONTACT velocity instead, auto-ramps
    # to ~0 as the clamp seats, and is largely mesh-insensitive. Applied to the
    # two pairs that carry the Z mode: bracket<->wall clamp and shank<->bore.
    # Contact controls cannot be attached in the Initial step (where the pairs
    # are created), so apply them from Preload_Step onward via setValuesInStep.
    #
    # dampFactor scales the automatically-computed contact stabilization
    # (keyword *CONTACT CONTROLS, STABILIZE=<factor>). Raised 1.0 -> 10.0:
    # the default AUTOMATIC amount converged the single
    # convergence-study geometry across all meshes, but a broad 100-point LHS at
    # div=24 left 8 borderline geometries (lower n_geo / higher e_geo) still
    # chattering open<->closed on the BracketMount/WallFront seat. A 10x factor
    # recovers them and still auto-ramps to ~0 as the clamp seats, so it stays
    # inert on already-converged models (verify ALLSD/ALLIE << 1%).
    model.StdContactControl(name='ClampStab', stabilizeChoice=AUTOMATIC,
                            dampFactor=10.0)
    model.interactions['Cont_BracketWall'].setValuesInStep(
        stepName='Preload_Step', contactControls='ClampStab')
    model.interactions['Cont_ShankBracketBore'].setValuesInStep(
        stepName='Preload_Step', contactControls='ClampStab')

    # -- Pre-tension section at Z = 0 (preload cut plane) ---------------------
    cut_faces  = bolt.faces.findAt(((W_geo/2 + rs * 0.5, Y_geo/2, 0.0),))
    cut_region = regionToolset.Region(side1Faces=cut_faces)
    # -- Bolt preload ----------------------------------------------------------
    if preload_force is None:
        F_preload = PRELOAD_RATIO * BOLT_PROOF_STRESS * pi * rs ** 2
    else:
        F_preload = float(preload_force)
    print('Bolt preload: %.1f N  (rs=%.2f mm, %.0f%% of grade 8.8 proof)'
          % (F_preload, rs, PRELOAD_RATIO * 100))

    model.BoltLoad(
        name='BoltPreload', createStepName='Preload_Step',
        region=cut_region,
        boltMethod=APPLY_FORCE, magnitude=F_preload)
    model.loads['BoltPreload'].setValuesInStep(
        stepName='Load_Step', boltMethod=FIX_LENGTH)

    return {'preload_force': F_preload, 'n_contact_pairs': 4,
            'pt_section': 'BoltPreTension'}


# =============================================================================
# GEOMETRY BUILD FUNCTIONS
# =============================================================================
def build_model_4walls(MODEL_NAME, b_geo, m_geo, n_geo, t_geo, f_geo, e_geo, i_geo, P_load, wall_thk, mesh_div=24.0):

    # DERIVED GEOMETRY
    Y_geo  = 4*m_geo - 2*n_geo + 4*e_geo   # Back wall height (equation designed to make oblique cut of flanges consistent)
    Z_geo = 2*m_geo    # Total height of the U-channel arm, which extends from the base thickness f_geo to the top of the arm (or the edge of the pad)
    W_geo  = b_geo + 2*t_geo   # Total outer width-to-depth

    # Geometry validation
    if Y_geo <= 0:
        raise ValueError(
            "Invalid geometry: Y_geo=%.1f (must be > 0). "
            "Check that 4*m_geo + 4*e_geo > 2*n_geo. "
            "Got: 4*%.1f + 4*%.1f = %.1f, 2*%.1f = %.1f"
            % (Y_geo, m_geo, e_geo, 4*m_geo+4*e_geo, n_geo, 2*n_geo)
        )
    
    # -------------------------------------------------
    # BASE SKETCH AND EXTRUSION
    # -------------------------------------------------

    # Set up a clean model for this run
    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]
    if 'Model-1' in mdb.models:
        mdb.models.changeKey(fromName='Model-1', toName=MODEL_NAME)
    else:
        mdb.Model(name=MODEL_NAME)

    model = mdb.models[MODEL_NAME]

    sk = model.ConstrainedSketch(name='base_profile', sheetSize=200.0)
    sk.rectangle(point1=(0.0, 0.0), point2=(W_geo, Y_geo))   # XY footprint

    part = model.Part(name='base', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=f_geo)
    # Result: solid box  X[0,W_geo]  Y[0,Y_geo]  Z[0,f_geo]
    part_base = model.parts['base']

    # -------------------------------------------------
    # FLANGE EXTRUSION
    # Sketch plane  : front face of back-wall block, Z = f_geo
    # Up edge       : left edge of that face, running along Y at X = 0
    # -------------------------------------------------

    # Front face of the back-wall block - the plane at Z = f_geo.
    # Sample point: centre of the face.
    flange_face = part_base.faces.findAt(((W_geo/2, Y_geo/2, f_geo),))[0].index

    # Left edge of the Z = f_geo face, running vertically along Y at X = 0.
    # This edge defines the "up" direction for the sketch (BOTTOM orientation).
    # Sample point: mid-height of that edge.
    flange_up_edge = part_base.edges.findAt(((W_geo/2, 0, f_geo),))[0].index

    sk2 = model.ConstrainedSketch(
        name='wall_profile',
        sheetSize=200.0,
        transform=part.MakeSketchTransform(
            sketchPlane=part_base.faces[flange_face],           # Front face of back wall at Z = f_geo
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part_base.edges[flange_up_edge],       # Left edge of Z = f_geo face, along Y at X = 0
            sketchOrientation=BOTTOM,
            origin=(W_geo/2, Y_geo/2, f_geo)
        )
    )

    part.projectReferencesOntoSketch(sketch=sk2, filter=COPLANAR_EDGES)

    sk2.Line((-W_geo/2,  Y_geo/2), (-W_geo/2, -Y_geo/2))   # Left outer vertical
    sk2.Line((-W_geo/2, -Y_geo/2), ( W_geo/2, -Y_geo/2))   # Bottom outer horizontal
    sk2.Line(( W_geo/2, -Y_geo/2), ( W_geo/2,  Y_geo/2))   # Right outer vertical

    xL  = -W_geo/2;  xR  =  W_geo/2
    yT  =  Y_geo/2;  yB  = -Y_geo/2
    xLi = xL + t_geo; xRi = xR - t_geo
    yBi = yB + e_geo

    sk2.Line(point1=(xR,  yT), point2=(xRi, yT))    # Top right: outer -> inner
    sk2.Line(point1=(xRi, yT), point2=(xRi, yBi))   # Right inner vertical
    sk2.Line(point1=(xRi, yBi), point2=(xLi, yBi))  # Inner bottom horizontal
    sk2.Line(point1=(xLi, yBi), point2=(xLi, yT))   # Left inner vertical
    sk2.Line(point1=(xLi, yT), point2=(xL,  yT))    # Top left: inner -> outer

    part.SolidExtrude(
        sketch=sk2,
        sketchPlane=part_base.faces[flange_face],           # Front face of back wall at Z = f_geo
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part_base.edges[flange_up_edge],       # Left edge of Z = f_geo face, along Y at X = 0
        sketchOrientation=BOTTOM,
        depth=Z_geo - f_geo                # Arm extends from Z = f_geo to Z = Z_geo
    )
    # Result: U-channel arm added, arm walls span Z[f_geo, Z_geo]


    # -------------------------------------------------
    # TRIANGULAR CUT FEATURE
    # Sketch plane  : right outer face of the arm, X = W_geo
    # Up edge       : rear vertical edge of that face, running along Y at
    #                 X = W_geo, Z = 0 (intersection with back wall)
    # -------------------------------------------------

    # Right outer face of the bracket arm - the plane at X = W_geo.
    # Spans Y[0, Y_geo], Z[0, Z_geo] after the flange extrusion.
    # Sample point: centre of that large face.
    cut_face = part_base.faces.findAt(((W_geo, Y_geo/2, Z_geo/2),))[0].index

    # Rear vertical edge of the X = W_geo face, running along Y at Z = 0.
    # Defines the "up" direction for the triangular-cut sketch (BOTTOM orientation).
    # Sample point: mid-height of that edge.
    cut_up_edge = part_base.edges.findAt(((W_geo, 0, Z_geo/2),))[0].index

    sk3 = model.ConstrainedSketch(
        name='cut_profile',
        sheetSize=200.0,
        transform=part_base.MakeSketchTransform(
            sketchPlane=part_base.faces[cut_face],          # Right outer face of arm at X = W_geo
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part_base.edges[cut_up_edge],      # Rear vertical edge of X = W_geo face, along Y at Z = 0
            sketchOrientation=BOTTOM,
            origin=(W_geo/2, Y_geo/2, Z_geo/2)
        )
    )

    part.projectReferencesOntoSketch(sketch=sk3, filter=COPLANAR_EDGES)

    x_left  = -Z_geo/2
    x_right =  Z_geo/2 - f_geo*2
    y_top   =  Y_geo/2
    y_bottom= -Y_geo/2 + e_geo*2

    sk3.Line((x_left,  y_top),    (x_left,  y_bottom))  # Vertical left edge of triangle
    sk3.Line((x_left,  y_bottom), (x_right, y_top))     # Diagonal hypotenuse
    sk3.Line((x_right, y_top),    (x_left,  y_top))     # Horizontal top edge (closes triangle)

    part.CutExtrude(
        sketch=sk3,
        sketchPlane=part_base.faces[cut_face],              # Right outer face of arm at X = W_geo
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part_base.edges[cut_up_edge],          # Rear vertical edge of X = W_geo face, along Y at Z = 0
        sketchOrientation=BOTTOM
    )
    # Result: triangular pocket removed from the right web of the arm


    # -------------------------------------------------
    # BORE IN BASE (BOTTOM WALL)
    # Sketch plane  : bottom face of the arm, Y = 0
    # Up edge       : rear edge of bottom face, running along X at Y = 0, Z = 0
    # -------------------------------------------------

    # Bottom face of the bracket - the plane at Y = 0.
    # Spans X[0, W_geo], Z[0, Z_geo].
    # Sample point: centre of the bottom face.
    base_bore_face = part_base.faces.findAt(((W_geo/2, 0, Z_geo/2),))[0].index

    # Rear edge of the Y = 0 face, running along X at Z = 0.
    # Defines the "up" (depth) direction for the bore sketch (BOTTOM orientation).
    # Sample point: mid-width of that edge.
    base_bore_up_edge = part_base.edges.findAt(((W_geo/2, 0, 0),))[0].index

    sk_bore_base = model.ConstrainedSketch(
        name='bore_base',
        sheetSize=200.0,
        transform=part.MakeSketchTransform(
            sketchPlane=part_base.faces[base_bore_face],        # Bottom face at Y = 0
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part_base.edges[base_bore_up_edge],    # Rear edge of Y = 0 face, along X at Z = 0
            sketchOrientation=BOTTOM,
            origin=(W_geo/2, 0.0, 0.0)
        )
    )

    part.projectReferencesOntoSketch(sketch=sk_bore_base, filter=COPLANAR_EDGES)

    sk_bore_base.CircleByCenterPerimeter(
        center=(0.0, -m_geo),
        point1=(i_geo, -m_geo)
    )

    part.CutExtrude(
        sketch=sk_bore_base,
        sketchPlane=part_base.faces[base_bore_face],            # Bottom face at Y = 0
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part_base.edges[base_bore_up_edge],        # Rear edge of Y = 0 face, along X at Z = 0
        sketchOrientation=BOTTOM,
        # depth=e_geo
    )
    # Result: through-hole cut in the bottom (ground) wall


    # -------------------------------------------------
    # BORE IN BACK WALL
    # Sketch plane  : back face of the fitting, Z = 0
    # Up edge       : top horizontal edge of back face, running along X at
    #                 Y = Y_geo, Z = 0
    # -------------------------------------------------

    # Back face of the fitting - the plane at Z = 0 (the mounting face).
    # Spans X[0, W_geo], Y[0, Y_geo].
    # Sample point: centre of the back face.
    wall_bore_face = part_base.faces.findAt(((W_geo/2, Y_geo/2, 0),))[0].index

    # Top horizontal edge of the Z = 0 face, running along X at Y = Y_geo.
    # Defines the "right" direction for the bore sketch (RIGHT orientation).
    # Sample point: mid-width of that edge.
    wall_bore_up_edge = part_base.edges.findAt(((W_geo/2, Y_geo, 0),))[0].index

    sk_bore_wall = model.ConstrainedSketch(
        name='bore_wall',
        sheetSize=200.0,
        transform=part.MakeSketchTransform(
            sketchPlane=part_base.faces[wall_bore_face],        # Back face of fitting at Z = 0
            sketchPlaneSide=SIDE1,
            sketchUpEdge=part_base.edges[wall_bore_up_edge],    # Top edge of Z = 0 face, along X at Y = Y_geo
            sketchOrientation=RIGHT,
            origin=(W_geo/2, Y_geo/2, 0.0)
        )
    )

    part.projectReferencesOntoSketch(sketch=sk_bore_wall, filter=COPLANAR_EDGES)

    sk_bore_wall.CircleByCenterPerimeter(
        center=(0.0, 0.0),
        point1=(i_geo, 0.0)
    )

    part.CutExtrude(
        sketch=sk_bore_wall,
        sketchPlane=part_base.faces[wall_bore_face],            # Back face of fitting at Z = 0
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part_base.edges[wall_bore_up_edge],        # Top edge of Z = 0 face, along X at Y = Y_geo
        sketchOrientation=RIGHT
        # depth=f_geo
    )
    # Result: through-hole cut in the back wall


    # -------------------------------------------------
    # MATERIAL DEFINITION
    # -------------------------------------------------

    model.Material(name='Al_7075')
    model.materials['Al_7075'].Elastic(table=((70000.0, 0.33),))


    # =================================================
    # PARTITIONING - split body into sweepable cells
    # =================================================
    # Must run AFTER all geometry features, BEFORE meshing.
    # Three principal-plane cuts produce cells that are each
    # individually sweepable (Abaqus picks the direction per cell).

    dz  = part.DatumPlaneByPrincipalPlane(
              principalPlane=XYPLANE, offset=f_geo).id           # back wall | arm
    dxL = part.DatumPlaneByPrincipalPlane(
              principalPlane=YZPLANE, offset=t_geo).id           # left flange | core
    dxR = part.DatumPlaneByPrincipalPlane(
              principalPlane=YZPLANE, offset=W_geo - t_geo).id   # core | right flange

    for did in (dz, dxL, dxR):
        try:
            part.PartitionCellByDatumPlane(
                datumPlane=part.datums[did],
                cells=part.cells
            )
        except Exception as err:
            print('Partition skipped (%s): %s' % (did, err))

    # -------------------------------------------------
    # SECTION DEFINITION
    # -------------------------------------------------

    model.HomogeneousSolidSection(name='Section-1', material='Al_7075')

    cells  = part.cells
    region = part.Set(cells=cells, name='Set-1')

    part.SectionAssignment(
        region=region,
        sectionName='Section-1',
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION
    )


    # -------------------------------------------------
    # ASSEMBLY
    # -------------------------------------------------

    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    assembly.Instance(name='base-1', part=part, dependent=ON)

    add_bolt_and_wall(model, W_geo, Y_geo, i_geo, f_geo, wall_thk)
    setup_contact_and_preload(model, W_geo, Y_geo, i_geo, f_geo, wall_thk)

    # =================================================
    # MESHING  - HEX-DOMINATED via SWEEP
    # =================================================
    cells = part.cells                       # re-grab AFTER partitioning
    part.setMeshControls(
        regions=cells,
        elemShape=HEX_DOMINATED,
        technique=SWEEP
    )


    elem_type_hex   = ElemType(elemCode=C3D8R,  elemLibrary=STANDARD)
    elem_type_wedge = ElemType(elemCode=C3D6,   elemLibrary=STANDARD)
    elem_type_tet   = ElemType(elemCode=C3D4,   elemLibrary=STANDARD)

    part.setElementType(
        regions=(cells,),
        elemTypes=(elem_type_hex,)
    )

    # Production default mesh_div=24: converged-stress sweet spot from the
    # convergence study, safely above the div=18 validity floor and off the
    # div=30 element-count threshold.
    mesh_size = min(W_geo, Y_geo, Z_geo) / mesh_div
    part.seedPart(size=mesh_size, deviationFactor=0.05, minSizeFactor=0.1)
    part.generateMesh()

    # -- Mesh metadata for convergence studies --------------------------------
    # Record seed / element count so a convergence study can plot peak stress
    # against mesh density. File name starts with MODEL_NAME so run_models'
    # _move_model_files() relocates it to output/<model>/ automatically.
    n_el = len(part.elements)
    print('[%s] mesh_div=%.2f  seed=%.4f  elements=%d'
          % (MODEL_NAME, mesh_div, mesh_size, n_el))
    with open(MODEL_NAME + '_mesh.csv', 'w') as _mf:
        _mf.write('mesh_div,mesh_size,n_elements\n')
        _mf.write('%g,%g,%d\n' % (mesh_div, mesh_size, n_el))


    # -------------------------------------------------
    # LOAD
    # Cylindrical surface of the base (bottom-wall) bore hole.
    # Point on that surface: offset from hole centre by i_geo in X,
    # at Y = e_geo/2 (mid-thickness of bottom wall),
    # at Z = mid-depth of arm.
    # -------------------------------------------------

    loaded_bore_face = assembly.instances['base-1'].faces.findAt(
        ((W_geo/2 + i_geo, e_geo/2, m_geo),)
        # Cylindrical face of bottom-wall bore - point on hole surface at mid-arm depth
    )

    rp = assembly.ReferencePoint(
        point=(W_geo/2, e_geo/2, m_geo)
    )

    rp_region = regionToolset.Region(
        referencePoints=(assembly.referencePoints[rp.id],)
    )
    
    surf = assembly.Surface(name='BoreSurf', side1Faces=loaded_bore_face)

    model.Coupling(
        name='BoltCoupling',
        controlPoint=rp_region,
        surface=surf,
        influenceRadius=WHOLE_SURFACE,
        couplingType=DISTRIBUTING
    )

    model.ConcentratedForce(
        name='BoltForce',
        createStepName='Load_Step',
        region=rp_region,
        cf2=-P_load
    )


    # # -------------------------------------------------
    # # DISPLAY
    # # -------------------------------------------------
    # vp = session.viewports['Viewport: 1']
    # vp.setValues(displayedObject=part)
    # vp.partDisplay.setValues(mesh=ON)
    # vp.assemblyDisplay.meshOptions.setValues(meshTechnique=ON)
    # vp.view.fitView()


    # =================================================
    # CREATE AND SUBMIT JOB
    # =================================================
    mdb.Job(
        name=MODEL_NAME,
        model=MODEL_NAME,
        description='U-channel bracket analysis',
        type=ANALYSIS,
        atTime=None,
        waitMinutes=0,
        waitHours=0,
        queue=None,
        memory=90,                      # % of available RAM
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        userSubroutine='',
        scratch='',
        resultsFormat=ODB,              # generates the .odb file
        multiprocessingMode=DEFAULT,
        numCpus=4,
        numDomains=4,
        numGPUs=0
    )

    # Submit and wait for completion
    mdb.jobs[MODEL_NAME].submit(consistencyChecking=OFF)
    try:
        mdb.jobs[MODEL_NAME].waitForCompletion()
    except Exception as e:
        print('WARNING: waitForCompletion raised: %s' % e)
    _status = mdb.jobs[MODEL_NAME].status
    # In noGUI/script mode mdb.jobs[X].status is unreliable: it comes back None
    # even for jobs that completed successfully, so it must NOT be used as the
    # pass/fail signal. Using it here failed EVERY model (None != COMPLETED) and
    # skipped CSV extraction entirely. Use the .sta file as ground truth instead:
    # a fully converged 2-step job ends with 'THE ANALYSIS HAS COMPLETED
    # SUCCESSFULLY'; an aborted / diverged / partial job does not -- so this
    # still refuses to extract from a bad ODB (preserving the original intent).
    _sta_ok = False
    try:
        with open(MODEL_NAME + '.sta') as _staf:
            _sta_ok = 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in _staf.read()
    except (IOError, OSError):
        _sta_ok = False
    print('Job finished: status=%s, sta_completed=%s -> %s.odb'
          % (_status, _sta_ok, MODEL_NAME))
    if not _sta_ok:
        # Dump the last few job messages for diagnosis. NOTE: slice a LIST copy,
        # not the Abaqus MessageArray directly -- messages[-8:] raises
        # 'IndexError: 0' on the native array.
        try:
            for _m in list(mdb.jobs[MODEL_NAME].messages)[-8:]:
                print('  [%s] %s' % (_m.type, _m.data))
        except Exception as _e:
            print('  (could not read job messages: %s)' % _e)
        raise RuntimeError(
            'Job %s did not complete (status=%s; no COMPLETED SUCCESSFULLY in '
            '%s.sta). Solver did not converge - see %s.msg / %s.sta.'
            % (MODEL_NAME, _status, MODEL_NAME, MODEL_NAME, MODEL_NAME))

    # Open ODB and extract results
    odb_path = MODEL_NAME + '.odb'
    odb = session.openOdb(name=odb_path)

    # Viewport display - skipped automatically in noGUI mode
    try:
        vp = session.viewports['Viewport: 1']
        vp.setValues(displayedObject=odb)
        vp.odbDisplay.deformedShapeOptions.setValues(
            deformationScaling=UNIFORM,
            uniformScaleFactor=1.0
        )
        vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
    except Exception as e:
        print('Viewport display skipped: %s' % e)

    # -- Save full nodal results to CSV ----------------------------------------
    # Do NOT swallow extraction errors: a failure here used to be printed and
    # ignored, so run_models.py reported success while the previous (stale)
    # <model>_results.csv silently survived. Let it propagate (run_models marks
    # the model FAILED and it is retried next run); close the ODB either way.
    try:
        _save_results_csv(odb, MODEL_NAME)
    finally:
        odb.close()
        print("ODB closed: %s.odb" % MODEL_NAME)