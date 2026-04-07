"""
fit_ellipse.py
==============
Cloud Regression Ellipse Fitting
Based on Granville (2022) and Halir & Flusser (1998)

Reference:
    Wansouwé, W. (2026). Partial Arc Ellipse Fitting via Cloud
    Regression: Critical Coverage Threshold and Sensitivity to
    Noise and Eccentricity. Pattern Recognition Letters.

    Granville, V. (2022). Machine Learning Cloud Regression:
    The Swiss Army Knife of Optimization. MLTechniques.com.

Author : Wansouwé Wanbitching
Email  : ericwansouwe@gmail.com
Date   : March 2026
"""

import numpy as np


def fit_ellipse(x, y):
    """
    Fit an ellipse to a set of 2D points using the Cloud
    Regression framework (Granville, 2022), implemented via
    the numerically stable algorithm of Halir & Flusser (1998).

    The method minimizes the sum of squared algebraic distances
    between the observed points and the fitted conic, subject to
    the ellipse-specific constraint 4ac - b^2 > 0.

    Parameters
    ----------
    x : array-like of shape (n,)
        x-coordinates of the observed points.
    y : array-like of shape (n,)
        y-coordinates of the observed points.

    Returns
    -------
    coeffs : ndarray of shape (6,)
        Coefficients (a, b, c, d, e, f) of the general conic
        equation: a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0.

    Raises
    ------
    ValueError
        If no ellipse solution is found (all eigenvalues
        yield a non-ellipse conic).

    Notes
    -----
    The artificial variable matrix W is constructed as:
        W = [x^2, x*y, y^2, x, y, 1]
    The solution is the eigenvector of W^T W associated with
    its smallest eigenvalue, following the Cloud Regression
    framework (Granville, 2022, eq. 6).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Build sub-matrices of the artificial variable matrix W
    D1 = np.vstack([x**2, x*y, y**2]).T   # quadratic terms
    D2 = np.vstack([x, y, np.ones(len(x))]).T  # linear terms

    # Block sub-matrices of W^T W
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    # Reduced eigenvalue system (Halir & Flusser, 1998)
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T

    # Ellipse-specific constraint matrix C
    C = np.array([[0, 0, 2],
                  [0, -1, 0],
                  [2, 0, 0]], dtype=float)
    M = np.linalg.inv(C) @ M

    # Eigendecomposition — solution is eigenvector of lambda_min
    eigval, eigvec = np.linalg.eig(M)

    # Select eigenvector satisfying the ellipse constraint
    # 4*a*c - b^2 > 0
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
    valid = np.nonzero(con > 0)[0]

    if len(valid) == 0:
        raise ValueError(
            "No ellipse solution found. The data may not "
            "describe an ellipse (too few points, or arc "
            "coverage below the critical 180-degree threshold)."
        )

    ak = eigvec[:, valid]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """
    Convert algebraic ellipse coefficients to geometric parameters.

    Given the conic equation a*x^2 + b*x*y + c*y^2 + d*x + e*y
    + f = 0 (with b^2 - 4ac < 0 for an ellipse), compute the
    five geometric parameters: center, semi-axes, and orientation.

    Parameters
    ----------
    coeffs : array-like of shape (6,)
        Coefficients (a, b, c, d, e, f) of the conic equation,
        as returned by fit_ellipse().

    Returns
    -------
    x0 : float
        x-coordinate of the ellipse center.
    y0 : float
        y-coordinate of the ellipse center.
    ap : float
        Length of the semi-major axis (ap >= bp).
    bp : float
        Length of the semi-minor axis.
    phi : float
        Orientation angle of the semi-major axis w.r.t. the
        x-axis, in radians, in [0, pi).

    Raises
    ------
    ValueError
        If the coefficients do not represent an ellipse
        (discriminant b^2 - 4ac >= 0).

    References
    ----------
    Formulas follow https://mathworld.wolfram.com/Ellipse.html
    using the convention a*x^2 + 2b*x*y + c*y^2 + 2d*x + 2f*y
    + g = 0 (note the factor-of-2 rescaling of b, d, f).
    """
    # Rescale coefficients to Mathworld convention
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    # Check ellipse condition
    den = b**2 - a*c
    if den > 0:
        raise ValueError(
            "Coefficients do not represent an ellipse: "
            "b^2 - 4ac must be negative."
        )

    # Ellipse center
    x0 = (c*d - b*f) / den
    y0 = (a*f - b*d) / den

    # Semi-axes lengths
    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    ap  = np.sqrt(num / den / ( fac - a - c))
    bp  = np.sqrt(num / den / (-fac - a - c))

    # Enforce convention ap >= bp
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # Orientation angle
    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2.0 * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
        if not width_gt_height:
            phi += np.pi / 2
        phi = phi % np.pi

    return x0, y0, ap, bp, phi


def get_ellipse_points(params, n=200, t_min=0,
                       t_max=2*np.pi):
    """
    Generate points on an ellipse arc.

    Parameters
    ----------
    params : tuple (x0, y0, ap, bp, phi)
        Geometric parameters of the ellipse.
    n : int
        Number of points to generate.
    t_min : float
        Start angle of the arc in radians (default: 0).
    t_max : float
        End angle of the arc in radians (default: 2*pi).

    Returns
    -------
    x, y : ndarray of shape (n,)
        Cartesian coordinates of the ellipse arc points.
    """
    x0, y0, ap, bp, phi = params
    t = np.linspace(t_min, t_max, n)
    x = (x0 + ap * np.cos(t) * np.cos(phi)
            - bp * np.sin(t) * np.sin(phi))
    y = (y0 + ap * np.cos(t) * np.sin(phi)
            + bp * np.sin(t) * np.cos(phi))
    return x, y


def fit_and_evaluate(params_true, arc_deg, sigma,
                     n=200, seed=42):
    """
    Generate a noisy partial arc and fit an ellipse.

    Parameters
    ----------
    params_true : tuple (x0, y0, ap, bp, phi)
        True ellipse parameters.
    arc_deg : float
        Arc coverage angle in degrees.
    sigma : float
        Standard deviation of Gaussian noise.
    n : int
        Number of points to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    params_est : tuple or None
        Estimated parameters (x0, y0, ap, bp, phi),
        or None if fitting fails.
    error_center : float
        Euclidean error on the center estimate.
    """
    np.random.seed(seed)
    x0, y0, ap, bp, phi = params_true
    arc_rad = arc_deg * np.pi / 180

    # Generate noisy arc points
    x, y = get_ellipse_points(params_true, n,
                               t_min=0, t_max=arc_rad)
    x += np.random.normal(0, sigma, n)
    y += np.random.normal(0, sigma, n)

    # Fit ellipse
    try:
        coeffs = fit_ellipse(x, y)
        x0_est, y0_est, ap_est, bp_est, phi_est = \
            cart_to_pol(coeffs)
        error_center = np.sqrt(
            (x0_est - x0)**2 + (y0_est - y0)**2
        )
        return (x0_est, y0_est, ap_est, bp_est, phi_est), \
               error_center
    except (ValueError, np.linalg.LinAlgError):
        return None, np.nan


if __name__ == "__main__":
    # Quick demo
    import matplotlib.pyplot as plt

    # True ellipse parameters
    params_true = (2.0, 3.0, 5.0, 3.0, np.pi / 4)

    # Generate noisy full ellipse
    np.random.seed(42)
    x, y = get_ellipse_points(params_true, n=100)
    x += np.random.normal(0, 0.3, 100)
    y += np.random.normal(0, 0.3, 100)

    # Fit
    coeffs = fit_ellipse(x, y)
    params_est = cart_to_pol(coeffs)

    print("True parameters      :", params_true)
    print("Estimated parameters :",
          tuple(round(p, 4) for p in params_est))

    # Plot
    t = np.linspace(0, 2 * np.pi, 300)
    x0, y0, ap, bp, phi = params_true
    x_true = (x0 + ap*np.cos(t)*np.cos(phi)
                 - bp*np.sin(t)*np.sin(phi))
    y_true = (y0 + ap*np.cos(t)*np.sin(phi)
                 + bp*np.sin(t)*np.cos(phi))

    x0e, y0e, ape, bpe, phie = params_est
    x_est = (x0e + ape*np.cos(t)*np.cos(phie)
                 - bpe*np.sin(t)*np.sin(phie))
    y_est = (y0e + ape*np.cos(t)*np.sin(phie)
                 + bpe*np.sin(t)*np.cos(phie))

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, color='red',
                label='Noisy observations', zorder=3)
    plt.plot(x_true, y_true, 'k-', lw=1.5,
             label='True ellipse')
    plt.plot(x_est,  y_est,  'b--', lw=1.5,
             label='Fitted ellipse')
    plt.axis('equal')
    plt.legend()
    plt.title('Cloud Regression Ellipse Fitting — Demo')
    plt.tight_layout()
    plt.savefig('demo_fit.png', dpi=150)
    plt.show()
    print("Demo figure saved as demo_fit.png")
