"""
changes means and errors and then run to create interval plots
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

objects = ("Reflectivity", "Velocity", "Rho HV", "Zdr", "Aggregate")
y_pos = np.arange(len(objects))
means = [0.5784, 0.5673, 0.5974, 0.6178, 0.5947]
errors = [0.0165, 0.0357, 0.0248, 0.0139, 0.0231]

plt.errorbar(y_pos, means, yerr=errors, marker="o", linestyle="")
plt.xticks(y_pos, objects)
plt.ylabel("Accuracy")
plt.ylim((0, 1))

plt.show()
