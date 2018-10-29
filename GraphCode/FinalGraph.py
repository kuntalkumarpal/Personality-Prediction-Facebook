import matplotlib.pyplot as plt
import numpy as np


#SVMLINEAR, SVMPOLY, SVMRBF, DECISIONTREE

alpha = [0.1,0.3,0.5,0.7,0.9]

OT_O_L = [0.6628189622,0.6632343011,0.6620320806,0.6613617692,0.6598725924]
OT_O_P = [0.6597131534,0.6597726129,0.6598458825,0.6601425351,0.6599863557]
OT_O_R = [0.6403141895,0.6405185907,0.640102558,0.6398828074,0.6395915887]
OT_O_D = [0.6520371246,0.6521018645,0.6523630134,0.6521105118,0.6523317264]

OT_A_L = [0.7196212577,0.7192288549,0.7193728915,0.7202411764,0.7187166146]
OT_A_P = [0.7089014171,0.7089207497,0.7089463205,0.708974491,0.7090386053]
OT_A_R = [0.6944183256,0.694336735,0.6943328504,0.6941133846,0.6940345463]
OT_A_D = [0.6992169191,0.6991934522,0.6999352746,0.6999523826,0.7002489138]

# Data for plotting
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

# Create 5 subplots sharing y axis
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharey=False)

ax1.set_ylim([0.62,0.67])
ax1.set_xlim([0,1])
ax1.plot(alpha, OT_O_L, 'ko-')
ax1.plot(alpha, OT_O_P, 'ro-')
ax1.plot(alpha, OT_O_R, 'bo-')
ax1.plot(alpha, OT_O_D, 'co-')
ax1.set(title='Openness', ylabel='MSE')


ax2.set_ylim([0.68,0.73])
ax2.set_xlim([0,1])
ax2.plot(alpha, OT_A_L, 'ko-')
ax2.plot(alpha, OT_A_P, 'ro-')
ax2.plot(alpha, OT_A_R, 'bo-')
ax2.plot(alpha, OT_A_D, 'co-')
ax2.set(title='Aggreableness', ylabel='MSE')


ax3.plot(x1, y1, 'ko-')
ax3.set(title='A tale of 2 subplots', ylabel='Damped oscillation')

ax4.plot(x2, y2, 'r.-')
ax4.set(xlabel='time (s)', ylabel='Undamped')

ax5.plot(x1, y1, 'ko-')
ax5.set(title='A tale of 2 subplots', ylabel='Damped oscillation')


plt.show()
