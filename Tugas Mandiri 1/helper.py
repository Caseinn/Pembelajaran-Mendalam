# plot_helper.py
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_scatter_with_boundary(model, X_test_pt, y_test_pt, device=None, title="Decision Boundary with Test Data"):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    X = X_test_pt.detach().cpu().numpy()
    y = y_test_pt.detach().cpu().numpy().ravel()

    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        zz = model(grid_t).cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', edgecolors='k', label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', edgecolors='k', label="Class 1")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
