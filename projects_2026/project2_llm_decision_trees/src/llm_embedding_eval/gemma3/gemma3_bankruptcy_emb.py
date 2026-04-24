from dataclasses import dataclass

def dt_func_0(row):
    """
    Predicts bankruptcy based on financial ratios.

    Args:
        row (dict): A dictionary containing the financial ratios.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node values.
    """
    node1 = int(row['WC/TA'] < 5)
    node2 = int(row['RE/TA'] < 2)
    node3 = int(row['EBIT/TA'] < 5)
    node4 = int(row['S/TA'] < 2)
    node5 = int(row['BVE/BVL'] < 1)
    
    emb = [node1, node2, node3, node4, node5]

    if node1 == 1:
        if node2 == 1:
            return 1, emb
        else:
            return 0, emb
    else:
        if node3 == 1:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node5 == 1:
                return 1, emb
            else:
                return 0, emb
            
def dt_func_1(row):
    """
    Predicts bankruptcy based on financial ratios.

    Args:
        row (dict): A dictionary containing the financial ratios.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node values.
    """

    node1 = int(row['WC/TA'] < 5)
    node2 = int(row['RE/TA'] < 2)
    node3 = int(row['EBIT/TA'] < 5)
    node4 = int(row['S/TA'] < 2)
    node5 = int(row['BVE/BVL'] < 1)

    emb = [node1, node2, node3, node4, node5]

    if node1 == 1:
        if node2 == 1:
            return 1, emb
        else:
            return 0, emb
    else:
        if node3 == 1:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node5 == 1:
                return 1, emb
            else:
                return 0, emb
            
def dt_func_2(row):
    """
    Predicts bankruptcy based on financial ratios.

    Args:
        row (dict): A dictionary containing the financial ratios.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node values.
    """

    node1 = int(row['WC/TA'] < 10)
    node2 = int(row['RE/TA'] < 5)
    node3 = int(row['EBIT/TA'] < 5)
    node4 = int(row['S/TA'] < 2)
    node5 = int(row['BVE/BVL'] < 0.5)

    emb = [node1, node2, node3, node4, node5]

    if node1 == 1:
        if node2 == 1:
            return 1, emb
        else:
            return 0, emb
    else:
        if node3 == 1:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node5 == 1:
                return 1, emb
            else:
                return 0, emb
            
def dt_func_3(row):
    """
    Predicts bankruptcy based on financial ratios.

    Args:
        row (dict): A dictionary containing the financial ratios.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node values.
    """
    node1 = int(row['WC/TA'] < 10)
    node2 = int(row['RE/TA'] < 5)
    node3 = int(row['EBIT/TA'] < 5)
    node4 = int(row['S/TA'] < 2)
    node5 = int(row['BVE/BVL'] < 0.5)
    
    emb = [node1, node2, node3, node4, node5]

    if node1 == 1:
        if node2 == 1:
            return 1, emb
        else:
            return 0, emb
    else:
        if node3 == 1:
            return 1, emb
        else:
            if node4 == 1:
                return 1, emb
            else:
                if node5 == 1:
                    return 1, emb
                else:
                    return 0, emb
                
def dt_func_4(row):
    """
    Predicts bankruptcy based on financial ratios.

    Args:
        row (dict): A dictionary containing the feature values.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node embeddings.
    """

    node1 = int(row['WC/TA'] < 0.10)
    node2 = int(row['RE/TA'] < 0.05)
    node3 = int(row['EBIT/TA'] < 0.03)
    node4 = int(row['S/TA'] < 0.50)
    node5 = int(row['BVE/BVL'] < 0.50)

    emb = [node1, node2, node3, node4, node5]

    if node1 == 1:
        if node2 == 1:
            prediction = 1
        else:
            if node3 == 1:
                prediction = 1
            else:
                prediction = 0
    else:
        if node4 == 1:
            if node5 == 1:
                prediction = 1
            else:
                prediction = 0
        else:
            prediction = 0

    return prediction, emb


@dataclass
class Gemma3BankruptcyEmbedding:
        runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]