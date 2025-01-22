'''Visualize the experimental comparison results of disaster risk temporal prediction as a chart.'''

import matplotlib.pyplot as plt

# Data for the plots
env_T = [1, 2, 3, 4, 5]
env_Accuracy = [0.7232, 0.7320, 0.7315, 0.7302, 0.7320]
env_Loss = [0.6351, 0.6091, 0.6115, 0.6116, 0.6078]

sem_T = [1, 2, 3, 4, 5]
sem_Accuracy = [0.7226, 0.7540, 0.7782, 0.7888, 0.7886]
sem_Loss = [0.6304, 0.5669, 0.5049, 0.4788, 0.4864]

# Create the plots with legend moved further left within the grid area and x-axis labels adjusted
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(env_T, env_Accuracy, 'g-', marker='^', label='Env Accuracy')
ax2.plot(env_T, env_Loss, 'g--', marker='s', label='Env Loss')
ax1.plot(sem_T, sem_Accuracy, 'b-', marker='^', label='Sem Accuracy')
ax2.plot(sem_T, sem_Loss, 'b--',marker='s', label='Sem Loss')

import matplotlib.font_manager as fm
available_fonts = fm.findSystemFonts()
chinese_font = next((font for font in available_fonts if 'SimHei' in font), None)

ax1.set_xlabel('T/days',fontproperties=fm.FontProperties(fname=chinese_font))
ax1.set_ylabel('Accuracy', color='black')
ax2.set_ylabel('Loss', color='black')

# Set x ticks to only integer values 1-5
ax1.set_xticks([1, 2, 3, 4, 5])
ax2.set_xticks([1, 2, 3, 4, 5])

fig.legend(loc='center left', bbox_to_anchor=(0.65, 0.5))

# plt.title('Comparison of Accuracy and Loss for Different Methods')
plt.show()
