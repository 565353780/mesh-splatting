import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def degenerated_triagles(_triangle_indices, vertices):
    min_deg, max_deg = 1.0, 179.0

    ti = _triangle_indices.long()
    A, B, C = vertices[ti[:,0]], vertices[ti[:,1]], vertices[ti[:,2]]

    # edge vectors opposite to vertices A,B,C
    a = B - C      # |a| opposite A
    b = C - A      # |b| opposite B
    c = A - B      # |c| opposite C

    # lengths with epsilon
    eps = 1e-12
    la = torch.linalg.norm(a, dim=1).clamp_min(eps)
    lb = torch.linalg.norm(b, dim=1).clamp_min(eps)
    lc = torch.linalg.norm(c, dim=1).clamp_min(eps)

    # cosines via law of cosines (numerically stable with clamp)
    cosA = ((lb**2 + lc**2 - la**2) / (2*lb*lc)).clamp(-1.0, 1.0)
    cosB = ((lc**2 + la**2 - lb**2) / (2*lc*la)).clamp(-1.0, 1.0)
    cosC = ((la**2 + lb**2 - lc**2) / (2*la*lb)).clamp(-1.0, 1.0)

    # angles in degrees
    rad2deg = 180.0 / torch.pi
    Adeg = torch.arccos(cosA) * rad2deg
    Bdeg = torch.arccos(cosB) * rad2deg
    Cdeg = torch.arccos(cosC) * rad2deg

    # mask triangles with any angle < min or > max
    angle_mask = (Adeg < min_deg) | (Bdeg < min_deg) | (Cdeg < min_deg) | (Adeg > max_deg) | (Bdeg > max_deg) | (Cdeg > max_deg)

    return angle_mask