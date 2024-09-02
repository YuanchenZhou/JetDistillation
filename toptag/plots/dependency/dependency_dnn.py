import numpy as np
import matplotlib.pyplot as plt
import re
import os

#dnn simple auc save path
dnn_simple_auc_path = '/users/yzhou276/work/toptag/simple/dnn/auc/'

#dnn student auc save path
dnn_student_auc_path = '/users/yzhou276/work/toptag/student/dnn/auc/'

#dnn avg multi student auc save path
dnn_avg_multi_student_auc_path = '/users/yzhou276/work/toptag/student/dnn/avg_auc/'

#dnn lms multi student auc save path
dnn_lms_multi_student_auc_path = '/users/yzhou276/work/toptag/student/dnn/lms_auc/'

#to save path
to_save_path = '/users/yzhou276/work/toptag/plots/dependency/dependency_plot/'

number_pattern = r"[-+]?\d*\.\d+|\d+"

#Load dnn simple auc
dnn_simple_nlayers = [2,3,4,5,10]
dnn_simple_dense_sizes = [1,10,25,50,100,200,300,500]
dense_size = []
pythia_dnn_auc = []
pythia_dnn_dependency = []
herwig_dnn_auc = []
herwig_dnn_dependency = []
mix_dnn_auc = []
mix_dnn_dependency = []

colors = ['blue','green','orange','purple','red']
alpha_values = np.linspace(0,1,2*len(dnn_simple_dense_sizes)+1)

fig,ax = plt.subplots(figsize=(16,6))
for n in range(len(dnn_simple_nlayers)):
    a = dnn_simple_nlayers[n]
    for m in range(len(dnn_simple_dense_sizes)):
        b = dnn_simple_dense_sizes[m]
        dense = f'{a}_{b}'
        dense_size.append(dense)
        with open(dnn_simple_auc_path+f'best_pythia_dnn_nlayers{a}_dense{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            pythia_dnn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            pythia_dnn_dependency.append(float(dependency[0]))

            ax.scatter([float(pythia_auc[1])], [float(dependency[0])], label = f'Pythia {a}_{b}' ,marker = '.', color = colors[n], alpha = alpha_values[2*m+1])

        with open(dnn_simple_auc_path+f'best_herwig_dnn_nlayers{a}_dense{b}.txt', 'r')	as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            herwig_dnn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            herwig_dnn_dependency.append(float(dependency[0]))
            
            ax.scatter([float(pythia_auc[1])], [float(dependency[0])], label = f'Herwig {a}_{b}' ,marker = 'x', color = colors[n], alpha = alpha_values[2*m+1])
            
        with open(dnn_simple_auc_path+f'best_mix_dnn_nlayers{a}_dense{b}.txt', 'r')  as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            mix_dnn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            mix_dnn_dependency.append(float(dependency[0]))
            
            ax.scatter([float(pythia_auc[1])], [float(dependency[0])], label = f'Mix {a}_{b}' ,marker = '*', color = colors[n], alpha = alpha_values[2*m+1])


ax.set_title('DNN Dependency Plot')
ax.set_xlabel('AUC on Pythia dataset')
ax.set_ylabel('AUC_Diff / AUC_on_Pythia')
ax.set_xlim(0.87,0.92)
ax.set_ylim(0,0.02)
ax.legend(ncol=5, loc='upper left', bbox_to_anchor=(1.05,1.1))
fig.subplots_adjust(right=0.45)
plt.show()
plt.savefig(to_save_path + 'dnn_dependency_plot.jpg')

