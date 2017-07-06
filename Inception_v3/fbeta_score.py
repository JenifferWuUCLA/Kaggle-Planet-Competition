import numpy as np
from sklearn.metrics import fbeta_score

weather_label_index = np.array([5, 6, 10, 11])
weather_label_3 = np.array([5, 10, 11])
land_label_index = np.array([0, 1, 2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 16])

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')
def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    type_num = y.shape[1]
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(type_num):
            p2[:,i] = (p[:,i] > x[i]).astype(np.int)
        #p2 = combine_pred(p2, p, x)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score
    x = [0.2]*type_num
    for i in range(type_num):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 = i2 / float(resolution)
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)
    return x
