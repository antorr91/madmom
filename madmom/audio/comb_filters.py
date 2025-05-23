import numpy as np
from madmom.processors import Processor

def feed_forward_comb_filter(signal, tau, alpha):
    if tau <= 0:
        raise ValueError('`tau` must be greater than 0')
    y = signal.astype(float)
    y[tau:] += alpha * signal[:-tau]
    return y

def feed_backward_comb_filter(signal, tau, alpha):
    if signal.ndim == 1:
        return _feed_backward_comb_filter_1d(signal.astype(float), tau, alpha)
    elif signal.ndim == 2:
        return _feed_backward_comb_filter_2d(signal.astype(float), tau, alpha)
    else:
        raise ValueError('signal must be 1d or 2d')

def _feed_backward_comb_filter_1d(signal, tau, alpha):
    if tau <= 0:
        raise ValueError('`tau` must be greater than 0')
    y = signal.copy()
    for n in range(tau, len(signal)):
        y[n] += alpha * y[n - tau]
    return y

def _feed_backward_comb_filter_2d(signal, tau, alpha):
    if tau <= 0:
        raise ValueError('`tau` must be greater than 0')
    y = signal.copy()
    for d in range(2):
        for n in range(tau, len(signal)):
            y[n, d] += alpha * y[n - tau, d]
    return y

def comb_filter(signal, filter_function, tau, alpha):
    tau = np.array(tau, dtype=int, ndmin=1)
    if tau.ndim != 1:
        raise ValueError('`tau` must be a 1D numpy array')
    alpha = np.array(alpha, dtype=float, ndmin=1)
    if len(alpha) == 1:
        alpha = np.repeat(alpha, len(tau))
    if alpha.ndim != 1:
        raise ValueError('`alpha` must be a 1D numpy array')
    if len(tau) != len(alpha):
        raise ValueError('`tau` and `alpha` must have the same length')
    y = [filter_function(signal, t, alpha[i]) for i, t in np.ndenumerate(tau)]
    return np.vstack(y).T if signal.ndim == 1 else np.dstack(y)

class CombFilterbankProcessor(Processor):
    def __init__(self, filter_function, tau, alpha):
        self.tau = np.array(tau, dtype=int, ndmin=1)
        self.alpha = np.array(alpha, dtype=float, ndmin=1)
        if filter_function in ['forward', feed_forward_comb_filter]:
            self.filter_function = feed_forward_comb_filter
        elif filter_function in ['backward', feed_backward_comb_filter]:
            self.filter_function = feed_backward_comb_filter
        else:
            raise ValueError('unknown `filter_function`: %s' % filter_function)

    def process(self, data):
        return comb_filter(data, self.filter_function, self.tau, self.alpha)