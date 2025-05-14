def ROmega(X, 
           missing_indices, # missing indexes of tensor
           ):
    """Restrict the tensor X to the observed indices in Omega."""
    Xomega = X.copy()
    Xomega[missing_indices[0], missing_indices[1],missing_indices[2]] = 0
    return Xomega