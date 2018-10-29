import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


#SVMLINEAR, SVMPOLY, SVMRBF, DECISIONTREE

alpha = [0.1,0.3,0.5,0.7,0.9]

OT_O_L = [0.6484,       0.6485, 0.6483, 0.6501, 0.6499]
OT_O_P = [0.6364,	0.6374,	0.6388,	0.6412,	0.6437]
OT_O_R = [0.6257,	0.6194,	0.6328,	0.6257,	0.6258]
OT_O_D = [0.6427,	0.6424,	0.6426,	0.6426,	0.6425]

OT_A_L = [0.6983, 0.7055, 0.7005, 0.7013, 0.6989]
OT_A_P = [0.6895,	0.6904,	0.6918,	0.6942,	0.697]
#OT_A_R = [0.6896,	0.685,	0.6952,	0.6895,	0.6896]
OT_A_R = [0.6866,	0.685,	0.6852,	0.6855,	0.6855]
OT_A_D = [0.6872,	0.6871,	0.6871,	0.6872,	0.6874]

OT_E_L = [0.7912, 0.7922, 0.7926, 0.7946, 0.7957]
OT_E_P = [0.7832,	0.7846,	0.7869,	0.7902,	0.7938]
OT_E_R = [0.7678,	0.7793,	0.7779,	0.7678,	0.7678]
OT_E_D = [0.7868,	0.7873,	0.7874,	0.7874,	0.7873]

OT_N_L = [0.8134, 0.8056, 0.8097, 0.8122, 0.8111]
OT_N_P = [0.7906,	0.7916,	0.7931,	0.7954,	0.7978]
OT_N_R = [0.7843,	0.7904,	0.7906,	0.7841,	0.7841]
OT_N_D = [0.7989,	0.799,	0.7986,	0.7991,	0.799]

OT_C_L = [0.7198, 0.7181, 0.7221, 0.7183, 0.7125]
OT_C_P = [0.7006,	0.7034,	0.703,	0.7055,	0.7086]
OT_C_R = [0.6995,	0.6970,	0.6980, 0.7014,	0.7013]
OT_C_D = [0.7076,	0.7071,	0.708,	0.7083,	0.7074]


# Data for plotting
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

# Create 5 subplots sharing y axis
#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharey=False)
fig, (ax1) = plt.subplots(1, sharey=False)
ax1.set_ylim([0.61,0.67])
ax1.set_xlim([0,1])
#ax1.plot(alpha, OT_O_L, 'ko-.')
SVR_LIN = ax1.plot(alpha, OT_O_L, 'ko-.',label = 'SVR_LINEAR')
SVR_POLY = ax1.plot(alpha, OT_O_P, 'ro-',label = 'SVR_POLY')
SVR_RBF = ax1.plot(alpha, OT_O_R, 'bo--',label='SVR_RBF')
DECISION_TREE = ax1.plot(alpha, OT_O_D, 'go:', label='DECISION_TREE')
ax1.set(title='Openness', ylabel='MSE', xlabel = 'Alpha of LDA')
ax1.legend()
ax1.grid()
fig.savefig('1Openness.eps', bbox_inches='tight')


fig2, (ax2) = plt.subplots(1, sharey=False)
ax2.set_ylim([0.67,0.72])
ax2.set_xlim([0,1])
#ax2.plot(alpha, OT_O_L, 'ko-')
SVR_LIN = ax2.plot(alpha, OT_A_L, 'ko-.',label = 'SVR_LINEAR')
SVR_POLY = ax2.plot(alpha, OT_A_P, 'ro-',label = 'SVR_POLY')
SVR_RBF = ax2.plot(alpha, OT_A_R, 'bo--',label='SVR_RBF')
DECISION_TREE = ax2.plot(alpha, OT_A_D, 'go:', label='DECISION_TREE')
ax2.set(title='Aggreableness', ylabel='MSE', xlabel = 'Alpha of LDA')
ax2.legend()
ax2.grid()
fig2.savefig('2Aggreableness.eps', bbox_inches='tight')


fig3, (ax3) = plt.subplots(1, sharey=False)
ax3.set_ylim([0.76,0.82])
ax3.set_xlim([0,1])
#ax3.plot(alpha, OT_O_L, 'ko-')
SVR_LIN = ax3.plot(alpha, OT_E_L, 'ko-.',label = 'SVR_LINEAR')
SVR_POLY = ax3.plot(alpha, OT_E_P, 'ro-',label = 'SVR_POLY')
SVR_RBF = ax3.plot(alpha, OT_E_R, 'bo--',label='SVR_RBF')
DECISION_TREE = ax3.plot(alpha, OT_E_D, 'go:', label='DECISION_TREE')
ax3.set(title='Extrovertion', ylabel='MSE', xlabel = 'Alpha of LDA')
ax3.legend()
ax3.grid()
fig3.savefig('3Extrovertion.eps', bbox_inches='tight')



fig4, (ax4) = plt.subplots(1, sharey=False)
ax4.set_ylim([0.76,0.85])
ax4.set_xlim([0,1])
#ax4.plot(alpha, OT_O_L, 'ko-')
SVR_LIN = ax4.plot(alpha, OT_N_L, 'ko-.',label = 'SVR_LINEAR')
SVR_POLY = ax4.plot(alpha, OT_N_P, 'ro-',label = 'SVR_POLY')
SVR_RBF = ax4.plot(alpha, OT_N_R, 'bo--',label='SVR_RBF')
DECISION_TREE = ax4.plot(alpha, OT_N_D, 'go:', label='DECISION_TREE')
ax4.set(title='Neuroticism', ylabel='MSE', xlabel = 'Alpha of LDA')
ax4.legend()
ax4.grid()
fig4.savefig('4Neuroticism.eps', bbox_inches='tight')



fig5, (ax5) = plt.subplots(1, sharey=False)
ax5.set_ylim([0.69,0.74])
ax5.set_xlim([0,1])
#ax5.plot(alpha, OT_O_L, 'ko-')
SVR_LIN = ax5.plot(alpha, OT_C_L, 'ko-.',label = 'SVR_LINEAR')
SVR_POLY = ax5.plot(alpha, OT_C_P, 'ro-',label = 'SVR_POLY')
SVR_RBF = ax5.plot(alpha, OT_C_R, 'bo--',label='SVR_RBF')
DECISION_TREE = ax5.plot(alpha, OT_C_D, 'go:', label='DECISION_TREE')
ax5.set(title='Conscientiousness', ylabel='MSE', xlabel = 'Alpha of LDA')
ax5.legend()
ax5.grid()
fig5.savefig('5Conscientiousness.eps', bbox_inches='tight')

#plt.show()
