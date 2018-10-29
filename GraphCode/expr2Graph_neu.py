import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime

#import plotly.plotly as py
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api


'''
Openness	0.6618	0.6599	0.64	0.6522
Aggreableness	0.7194	0.7089	0.6942	0.6997
Extrovertion	0.8064	0.8093	0.7814	0.7968
Neuroticism	0.8228	0.81	0.7943	0.8064
Conscientiousness	0.7299	0.7317	0.7033	0.7166
'''
'''
Openness		0.6395	0.6259	0.6426
Aggreableness		0.6926	0.6855	0.6872
Extrovertion		0.7877	0.7721	0.7872
Neuroticism		0.7937	0.7867	0.7989
Conscientiousness		0.7042	0.6944	0.7077
'''
'''
x = [datetime.datetime(2011, 1, 4, 0, 0),
     datetime.datetime(2011, 1, 5, 0, 0),
     datetime.datetime(2011, 1, 6, 0, 0)]
x = date2num(x)

y = [4, 9, 2]
z=[1,2,3]
k=[11,12,13]

ax = plt.subplot(111)
ax.bar(x-0.2, y,width=0.2,color='b',align='center')
ax.bar(x, z,width=0.2,color='g',align='center')
ax.bar(x+0.2, k,width=0.2,color='r',align='center')
ax.xaxis_date()

plt.show()'''

wo = [0.8228, 0.81,	0.7943,	0.8064]
w = [0.8104, 0.7937,	0.7867,	0.7989]
objects = ('SVR-Linear','SVR-Poly','SVR-RBF','Decision Tree')
#y_pos = range(len(objects))
y_pos = [0.5,1,1.5,2]
plt.xticks(y_pos, objects)
ax1 = plt.subplot(111)
xx = ax1.bar([y-0.15 for y in y_pos], wo, width=0.15,color='b',align='center',alpha=0.5 ,label='Without LIWC')
xxx = ax1.bar([y for y in y_pos], w, width=0.15,color='g',align='center',alpha=0.5 ,label='With LIWC' )

ax1.set(title='Neuroticism', ylabel='MSE')
ax1.legend()
#ax1.grid()
plt.ylim((0,1.05))
plt.xlim((0,3))
#ax1.bar([y+0.1 for y in y_pos], dt, width=0.1,color='r',align='center',alpha=0.5 )

#plt.bar(y_pos, svrrbf,width=0.2,color='g',align='center',alpha=0.5 )
#ax1.bar(len(dt), dt,width=0.2,color='r',align='center')
#ax1.xaxis_date()


def autolabel(rects, order):
    for rect in rects:
        h = rect.get_height()
        print h,rect.get_x()
        if order == 1:
            ax1.text(rect.get_x()+rect.get_width()/2., 1.06*h, '%.4f'%h,
                ha='center', va='bottom')
        else:
            ax1.text(rect.get_x()+rect.get_width()/2.+0.02, 1.02*h, '%.4f'%h,
                ha='center', va='bottom')

autolabel(xx,1)
autolabel(xxx,2)

#plt.show()

plt.savefig('9_Neuroticism.eps', bbox_inches='tight')





