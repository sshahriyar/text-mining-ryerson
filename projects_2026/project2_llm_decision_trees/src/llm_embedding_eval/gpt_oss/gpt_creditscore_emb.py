from dataclasses import dataclass


def dt_func_0(row):
    # Precompute binary node evaluations
    node1 = int(row['Derogatory.reports'] <= 1)          # low derogatory reports
    node2 = int(row['Own.home'] == 1)                   # owns a home
    node3 = int(row['Income.per.dependent'] >= 4)        # decent income per dependent
    node4 = int(row['Age'] >= 35)                       # older applicant
    node5 = int(row['Monthly.credit.card.exp'] <= 200)  # low credit card spend

    emb = [node1, node2, node3, node4, node5]

    # Decision tree logic
    if node1:            # Derogatory.reports <= 1
        if node2:        # Own.home == 1
            if node3:    # Income.per.dependent >= 4
                prediction = 1
            else:        # Income.per.dependent < 4
                if node4:  # Age >= 35
                    prediction = 1
                else:
                    prediction = 0
        else:            # Own.home == 0
            if node5:    # Monthly.credit.card.exp <= 200
                prediction = 1
            else:
                prediction = 0
    else:                # Derogatory.reports > 1
        prediction = 0

    return prediction, emb

def dt_func_1(row):
    node1 = int(row['Derogatory.reports'] <= 1)
    node2 = int(row['Own.home'] == 1)
    node3 = int(row['Self.employed'] == 1)
    node4 = int(row['Age'] >= 30)
    node5 = int(row['Monthly.credit.card.exp'] <= 300)

    emb = [node1, node2, node3, node4, node5]

    if node1:
        if node2:
            if node3:
                pred = 0
            else:
                pred = 1
        else:
            if node4:
                pred = 1
            else:
                pred = 0
    else:
        if node5:
            pred = 1
        else:
            pred = 0

    return pred, emb


def dt_func_2(row):
    # Precompute node conditions (binary embeddings)
    node1 = int(row['Derogatory.reports'] <= 1)          # 1 if <=1 reports
    node2 = int(row['Own.home'] == 1)                    # 1 if owns a home
    node3 = int(row['Income.per.dependent'] >= 5)         # 1 if high income per dependent
    node4 = int(row['Monthly.credit.card.exp'] <= 100)   # 1 if credit card expenses are low
    node5 = int(row['Monthly.credit.card.exp'] <= 50)    # 1 if credit card expenses are very low
    node6 = int(row['Self.employed'] == 1)               # 1 if self‑employed

    emb = [node1, node2, node3, node4, node5, node6]

    # Decision tree logic
    if node1:                      # <=1 derogatory reports
        if node2:                  # owns a home
            if node3:              # high income per dependent
                prediction = 1    # accepted
            else:                  # low income per dependent
                if node4:          # low credit card expenses
                    prediction = 1
                else:
                    prediction = 0
        else:                      # does not own a home
            if node5:              # very low credit card expenses
                prediction = 1
            else:
                prediction = 0
    else:                          # >1 derogatory reports
        if node6:                  # self‑employed
            prediction = 0
        else:
            prediction = 1

    return prediction, emb


def dt_func_3(row):
    """
    Decision tree classifier for credit application acceptance.
    Returns a tuple (prediction, emb) where:
        prediction: 0 (not accepted) or 1 (accepted)
        emb: list of binary node evaluations (1 if true, 0 if false)
    Expected keys in `row`:
        'Derogatory.reports'   (int)
        'Income.per.dependent'  (float)
        'Own.home'              (int, 1=yes, 0=no)
        'Self.employed'         (int, 1=yes, 0=no)
        'Age'                   (int)
    """
    # Binary node evaluations
    node1 = int(row['Derogatory.reports'] <= 1)          # Few or no derogatory reports
    node2 = int(row['Income.per.dependent'] > 6.5)        # High income per dependent
    node3 = int(row['Own.home'] == 1)                    # Owns a home
    node4 = int(row['Self.employed'] == 1)               # Self‑employed
    node5 = int(row['Age'] > 30)                         # Above 30 years old

    emb = [node1, node2, node3, node4, node5]

    # Decision logic
    if node1:  # Low derogatory reports
        if node2:  # High income per dependent
            if node3:  # Owns a home
                prediction = 1
            else:
                if node4:  # Self‑employed
                    prediction = 1
                else:
                    prediction = 0
        else:
            prediction = 0
    else:
        prediction = 0

    return prediction, emb


def dt_func_4(row):
    # Binary node evaluations
    node1 = int(row['Derogatory.reports'] <= 1)          # 1: few or no reports
    node2 = int(row['Own.home'] == 1)                    # 1: owns a home
    node3 = int(row['Income.per.dependent'] >= 6)         # 1: high income per dependent
    node4 = int(row['Age'] >= 30)                        # 1: mature applicant
    node5 = int(row['Self.employed'] == 1)               # 1: self‑employed

    emb = [node1, node2, node3, node4, node5]

    # Decision tree logic
    if node1:                       # Few or no derogatory reports
        if node2:                   # Owns a home
            if node3:               # High income per dependent
                pred = 1           # Accept
            else:                   # Low income per dependent
                pred = 0           # Reject
        else:                       # Does not own a home
            if node4:               # Mature applicant
                pred = 1
            else:                   # Younger applicant
                pred = 0
    else:                           # One or more derogatory reports
        if node5:                   # Self‑employed
            pred = 1
        else:                       # Not self‑employed
            pred = 0

    return pred, emb

@dataclass
class GptCreditscoreEmbedding:
    runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]