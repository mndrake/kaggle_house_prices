import numpy as np
from scipy.optimize import fmin_cobyla

def train_regressors(x_vals, x_sub, y_vals, folds, regressors):
    # train classifiers and return out of fold train
    # predictions and blended submission predictions
    blend_train = np.zeros((x_vals.shape[0], len(regressors)))
    blend_test = np.zeros((x_sub.shape[0], len(regressors)))

    for j, model in enumerate(regressors):
        blend_test_j = np.zeros((x_sub.shape[0], folds.n_splits))
        for i, (train, test) in enumerate(folds.split(x_vals)):
            X_train = x_vals[train]
            y_train = y_vals[train]
            X_test = x_vals[test]
            model.fit(X_train, y_train)
            blend_train[test, j] = model.predict(X_test)
            blend_test_j[:, i] = model.predict(x_sub)
        blend_test[:,j] = np.mean(blend_test_j, axis=1)

    return blend_train, blend_test


def blended(c, x):
    """
    :param c: weights
    :param x: values to blend
    :return: blended values
    """
    result = None
    for i in range(len(c)):
        result = result + c[i] * x[i] if result is not None else c[i] * x[i]
    result /= sum(c)
    return result


def constraint(p, *args):
    return min(p) - .0


def blend_predictions(train_preds, test_preds, y_vals, folds, scorer):
    test_index = None
    for _, test_idx in folds.split(train_preds):
        test_index = np.append(test_index, test_idx) if test_index is not None else test_idx
    val_labels = y_vals[test_index]

    val_predictions, val_submission = [], []

    for i in range(np.shape(train_preds)[1]):
        val_predictions.append(train_preds[:, i])

    for i in range(np.shape(test_preds)[1]):
        val_submission.append(test_preds[:, i])

    p0 = np.array([1.] * len(val_predictions)) / len(val_predictions)

    def error(p, x, y):
        preds = blended(p, x)
        err = scorer(y, preds)
        return err

    p = fmin_cobyla(error, p0, args=(val_predictions, val_labels), cons=[constraint], rhoend=1e-5)

    err = error(p, val_predictions, val_labels)

    print 'weights:', p / np.sum(p)

    blended_train = blended(p, val_predictions)
    y_submission = blended(p, val_submission)

    return blended_train, y_submission