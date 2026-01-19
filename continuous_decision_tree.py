data = [
    (60, 18.4, "Owner"),
    (75, 19.0, "Nonowner"),
    (85.5, 16.8, "Owner"),
    (52.8, 20.8, "Nonowner"),
    (64.8, 21.6, "Owner"),
    (64.8, 17.2, "Nonowner"),
    (61.5, 20.8, "Owner"),
    (43.2, 20.4, "Nonowner"),
    (87, 23.6, "Owner"),
    (84, 17.6, "Nonowner"),
    (110.1, 19.2, "Owner"),
    (49.2, 17.6, "Nonowner"),
    (108, 17.6, "Owner"),
    (59.2, 16.0, "Nonowner"),
    (82.8, 22.4, "Owner"),
    (65, 18.4, "Nonowner"),
    (69, 20.0, "Owner"),
    (47.4, 15.4, "Nonowner"),
    (93, 20.8, "Owner"),
    (33, 18.8, "Nonowner"),
    (51, 22.0, "Owner"),
    (51, 14.0, "Nonowner"),
    (81, 20.0, "Owner"),
    (63, 14.8, "Nonowner"),
]


def gini(labels):
    total = len(labels)
    if total == 0:
        return 0
    p_owner = labels.count("Owner")/total
    p_non = labels.count("Nonowner")/total
    return 1-(p_owner**2+p_non**2)


def candidate_midpoints(data, index):
    data_sorted = sorted(data, key=lambda x: x[index])
    mids = []

    for i in range(len(data_sorted)-1):
        if data_sorted[i][2] != data_sorted[i+1][2]:
            v1 = data_sorted[i][index]
            v2 = data_sorted[i+1][index]
            mids.append((v1+v2)/2)
    return mids


def gini_for_split(data, index, split):
    left = [row[2] for row in data if row[index] <= split]
    right = [row[2] for row in data if row[index] > split]
    total = len(data)
    return (len(left)/total)*gini(left)+(len(right)/total)*gini(right)


def best_split(data, index):
    mids = candidate_midpoints(data, index)
    best = None
    best_gini = 1

    for m in mids:
        g = gini_for_split(data, index, m)
        if g < best_gini:
            best_gini = g
            best = m
    return best, best_gini


income_split, income_gini = best_split(data, 0)
lawn_split, lawn_gini = best_split(data, 1)

print("Best Income Split:", income_split, "Gini:", income_gini)
print("Best Lawn Split:", lawn_split, "Gini:", lawn_gini)

# for speicific data set uniform or normal distrituion
# compare between differnte methods
# slide  comparison graph of accracy precision recall .
# classification and
# regression
# mse and r2
