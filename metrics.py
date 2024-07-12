def metrics(cnf_matrix):
    if cnf_matrix.shape != (2, 2):
        raise ValueError("Confusion matrix should be 2x2.")
    
    TN = cnf_matrix[0, 0]
    FN = cnf_matrix[1, 0]
    TP = cnf_matrix[1, 1]
    FP = cnf_matrix[0, 1]
    
    if (2 * TP + FP + FN) == 0:
        DSC = 1.0  # Avoid division by zero
    else:
        DSC = (2 * TP) / ((2 * TP) + FP + FN)
    
    if (2 - DSC) == 0:
        JC = 1.0  # Avoid division by zero
    else:
        JC = DSC / (2 - DSC)
    
    return DSC, JC
