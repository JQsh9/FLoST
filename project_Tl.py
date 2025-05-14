import sys
sys.path.append("tensor_toolbox")
from t_prod import t_prod
from conjugate_transpose import conjugate_transpose

def project_Tl(A, U, V):
    """Project a tensor A onto the tangent space of the fixed transformed multi-rank manifold at X."""
    UT = conjugate_transpose(U)
    VT = conjugate_transpose(V)
    UUTA = t_prod(U, t_prod(UT, A))
    return UUTA + t_prod(t_prod(A, V),VT) - t_prod(t_prod(UUTA, V),VT)