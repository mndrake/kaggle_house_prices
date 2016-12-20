# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def plot_actual_predicted(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.show()
