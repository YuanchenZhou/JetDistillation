import numpy as np
import matplotlib.pyplot as plt
import re
import os

#efn simple auc save path
efn_simple_auc_path = '/users/yzhou276/work/toptag/simple/efn/auc/'

#efn student auc save path
efn_student_auc_path = '/users/yzhou276/work/toptag/student/efn/auc/'

#to save path
to_save_path = '/users/yzhou276/work/toptag/plots/dependency/dependency_plot/'

number_pattern = r"[-+]?\d*\.\d+|\d+"

#Load efn simple auc
efn_latent_sizes = [1,2,4,8,16,32,64,128,256,512]
efn_phi_sizes = [50,100,250,500]
pythia_efn_auc = []
pythia_efn_dependency = []
herwig_efn_auc = []
herwig_efn_dependency = []

colors = ['blue','green','orange','purple','red']
alpha_values = np.linspace(0,1,len(efn_latent_sizes)+1)

fig,ax = plt.subplots(figsize=(16,7))
for n in range(len(efn_phi_sizes)):
    b = efn_phi_sizes[n]
    for m in range(len(efn_latent_sizes)):
        a = efn_latent_sizes[m]
        with open(efn_simple_auc_path+f'best_pythia_efn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            pythia_efn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            pythia_efn_dependency.append(float(dependency[0]))
            
            ax.scatter([float(pythia_auc[1])], [float(dependency[0])], label = f'Pythia {a}_{b}' ,marker = '.', color = colors[n], alpha = alpha_values[m+1])
            
        with open(efn_simple_auc_path+f'best_herwig_efn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            herwig_efn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            herwig_efn_dependency.append(float(dependency[0]))
            
            ax.scatter([float(pythia_auc[1])], [float(dependency[0])], label = f'Herwig {a}_{b}' ,marker = 'x', color = colors[n], alpha = alpha_values[m+1])

ax.set_title('EFN Dependency Plot')
ax.set_xlabel('AUC on Pythia dataset')
ax.set_ylabel('AUC_Diff / AUC_on_Pythia')
ax.set_xlim(0.89,0.94)
ax.set_ylim(0,0.015)
ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(1.05,1))
fig.subplots_adjust(right=0.5)
plt.show()
plt.savefig(to_save_path + 'efn_dependency_plot.jpg')
