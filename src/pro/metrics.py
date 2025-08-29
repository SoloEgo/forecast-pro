import numpy as np
def wape(y, yhat):
    denom = np.clip(np.abs(y).sum(), 1e-9, None)
    return float(np.abs(y - yhat).sum() / denom)
def smape(y, yhat):
    return float((2*np.abs(y - yhat) / np.clip(np.abs(y)+np.abs(yhat), 1e-9, None)).mean())
def mase(y, yhat, seasonal_period=12):
    d = np.abs(y[seasonal_period:] - y[:-seasonal_period]).mean()
    d = max(d, 1e-9)
    return float(np.abs(y - yhat).mean() / d)
def pinball(y, qhat, tau: float):
    u = y - qhat
    return float(np.maximum(tau*u, (tau-1)*u).mean())
