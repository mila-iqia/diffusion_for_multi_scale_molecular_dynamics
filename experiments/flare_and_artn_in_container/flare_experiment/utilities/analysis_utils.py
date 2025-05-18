import numpy as np

def linear_fit_and_r2(x: np.ndarray, y: np.ndarray):

    coeffs = np.polyfit(x, y, deg=1)
    xfit = np.array([0, 1.5 * x.max()])
    yfit = np.poly1d(coeffs)(xfit)

    # Create the line of best fit
    y_predicted = coeffs[0] * x + coeffs[1]

    # 2. Calculate R-squared manually
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)  # Total sum of squares
    ss_residual = np.sum((y - y_predicted) ** 2)  # Residual sum of squares
    r2 = 1 - (ss_residual / ss_total)

    return xfit, yfit, r2