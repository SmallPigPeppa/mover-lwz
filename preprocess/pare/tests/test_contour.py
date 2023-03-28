import numpy as np
import numpy.random
import matplotlib.pyplot as plt

def f(x, y):
  return x*y

xmax = ymax = 100
z = numpy.array([[f(x, y) for x in range(xmax)] for y in range(ymax)])

plt.pcolormesh(z)
plt.colorbar()
curves = 10
m = max([max(row) for row in z])
levels = numpy.arange(0, m, (1 / float(curves)) * m)
plt.contour(z, levels=levels, cmap='jet')
plt.show()