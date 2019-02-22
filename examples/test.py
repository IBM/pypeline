import numpy as np

from pypeline.util.math.func import SphericalDirichlet
from pypeline.util.math.sphere import fibonacci_sample, FibonacciInterpolator


def gammaN(r, r0, N):
    similarity = np.tensordot(r0, r, axes=1)
    d_func = SphericalDirichlet(N)
    return d_func(similarity) * (N + 1) / (4 * np.pi)


# \gammaN Parameters
N = 3
r0 = np.array([1, 0, 0])

# Solution at Nyquist resolution
R_nyquist = fibonacci_sample(N)
g_nyquist = gammaN(R_nyquist, r0, N)

# Solution at high resolution
R_dense = fibonacci_sample(10 * N)  # dense grid
g_exact = gammaN(R_dense, r0, N)

# Interpolate Nyquist solution to high resolution solution
fib_interp = FibonacciInterpolator(N)
g_interp = fib_interp(support=R_nyquist,
                      f=g_nyquist.reshape(1, -1),
                      r=R_dense)

a = np.allclose(g_interp, g_exact)
import matplotlib.pyplot as plt
b = np.abs((g_interp - g_exact) / g_exact)
plt.plot(b.reshape(-1))
plt.show()

print(a)
