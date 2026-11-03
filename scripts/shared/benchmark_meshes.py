"""The ten benchmark meshes as Appendix A lists them: name, source and preparation for the table, and the view of each
in the gallery."""
from dataclasses import dataclass

from shared import render

STANFORD = "Stanford~\\cite{stanford3dscanrep}"
COMMON_3D = "\\cite{jacobson2020common3d}"


@dataclass(frozen=True)
class Mesh:
    name: str  # file in meshes/
    label: str  # name in the thesis
    source: str  # where the original comes from
    note: str = ""  # preparation beyond cleaning and scaling
    view: str = ""  # mesh-screenshot --look-at (eye, target[, up]); empty = default camera
    detail: tuple = ()  # preferred detail centre as fractions (x, y) of the full view; empty = centre of mass
    camera: str = ""  # mesh-screenshot --camera; overrides view
    detail_side: float = 0.0  # fixed detail window side as a fraction of the longer side of the full view


ROCKER_CAMERA = render.camera([-0.026951789855957, 0.0627247095108032, 0.176916003227234],
                             [0.356904417276382, 0.0, 0.934140861034393, -0.155645221471786,
                              -0.474825650453568, 0.861178874969482, 0.181415215134621, -0.0989099442958832,
                              -0.804462432861328, -0.508301973342896, 0.30735856294632, -1.72133219242096,
                              0.0, 0.0, 0.0, 1.0])
HOMER_CAMERA = render.camera([1.19209289550781e-07, 6.55651092529297e-07, 7.15255737304688e-07],
                            [0.890308082103729, 0.0, 0.455358743667603, -4.76837158203125e-07,
                             0.127006903290749, 0.960315465927124, -0.248321309685707, -4.76837158203125e-07,
                             -0.437288045883179, 0.278916150331497, 0.85497659444809, -2.87254476547241,
                             0.0, 0.0, 0.0, 1.0])
ELK_CAMERA = render.camera([0.17758572101593, -0.0320601463317871, -0.0586133003234863],
                          [0.880911767482758, 0.0, -0.473280549049377, -0.184177875518799,
                           -0.353293687105179, 0.665409624576569, -0.657581567764282, 0.0455300807952881,
                           0.3149254322052, 0.746478378772736, 0.58616715669632, -3.30361127853394,
                           0.0, 0.0, 0.0, 1.0])
MESHES = [
    Mesh("stanford-bunny-low", "Bunny (low)", STANFORD + ", decimated", "5 holes filled"),
    # The zoom sits on the uniform grid of the back.
    Mesh("stanford-bunny", "Bunny", STANFORD, "5 holes filled", detail=(0.47, 0.46)),
    Mesh("armadillo-r128", "Armadillo", STANFORD, "marching cubes", "0.8,0.3,-3.5,0,0,0"),
    Mesh("stanford-dragon-r128", "Dragon", STANFORD, "marching cubes"),
    Mesh("fandisk", "Fandisk", "Hoppe et al.~\\cite{hoppe1994piecewise,pmp23}"),
    Mesh("max-planck", "Max Planck", "MPI~" + COMMON_3D, "1 hole filled", "1.2,0.5,-3.8,0,0,0,0,1,0"),
    Mesh("horse", "Horse", "Cyberware~" + COMMON_3D, "", "-4,0,0.5,0,0,0,0,0,1"),
    # The zoom sits on the very large triangles of the patch that the pure queues leave unsplit (Fig. rocker-arm-objectives).
    Mesh("rocker-arm", "Rocker arm", "INRIA~" + COMMON_3D, "", camera=ROCKER_CAMERA, detail=(0.393, 0.778),
         detail_side=0.273),
    Mesh("homer", "Homer", "unknown~" + COMMON_3D, camera=HOMER_CAMERA, detail=(0.079, 0.499), detail_side=0.165),
    # The zoom sits on the toothed rim of the antler plate, where both sides of the thin sheet are visible.
    Mesh("elk", "Elk", "CGAL~\\cite{cgaldata}", camera=ELK_CAMERA, detail=(0.737, 0.157), detail_side=0.175),
]
