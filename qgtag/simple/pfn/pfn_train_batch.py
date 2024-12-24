import os

pfn_latent_sizes = [128]#[1,2,4,8,16,32,64,128,256,512]
pfn_phi_sizes = [250]#[50,100,250,500]

for i in range(10):
    for l in pfn_latent_sizes:
        for phi in pfn_phi_sizes:
            label = "latent"+str(l)+"_phi"+str(phi)+'_pfn'

            with open(f'batch/train_{label}.sh', 'w') as run_script:

                run_script.write('#!/bin/bash\n\n')

                run_script.write('#SBATCH -N 1\n')
                run_script.write('#SBATCH -n 1\n')
                run_script.write('#SBATCH --mem=96G\n')
                run_script.write('#SBATCH -t 12:00:00\n')

                # https://docs.ccv.brown.edu/oscar/gpu-computing/submit-gpu
                run_script.write('#SBATCH -p gpu --gres=gpu:1\n')

                run_script.write('#SBATCH -o logs/slurm-%j.out\n')
                run_script.write('#SBATCH -J PFN_'+label+'\n')
                run_script.write('\n')

                run_script.write('source /users/yzhou276/work/qgtag/qg/bin/activate\n\n')

                cmd =  'python3 pfn_train.py '
                cmd += '-nEpochs=200 '
                cmd += '-batchSize=500 '
                cmd += '-latentSize='+str(l)+' '
                cmd += '-phiSizes='+str(phi)+' '
                cmd += '-doEarlyStopping '
                cmd += '-patience=10 '
                cmd += '-usePIDs '
                cmd += '-ModelNum='+str(i)+' '

                print(cmd)
                run_script.write(cmd+'\n')

            print('sbatch batch/train_'+label+'.sh')
            os.system(f'sbatch batch/train_{label}.sh')
