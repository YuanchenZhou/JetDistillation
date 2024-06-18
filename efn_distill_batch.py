import os

#pfn_latent_sizes = [1,2,4,8,16,32,64,128,256,512]
pfn_latent_sizes = [128]
#pfn_phi_sizes = [50,100,250,500]
pfn_phi_sizes = [250]

efn_latent_sizes = [1,2,4,8,16,32,64,128] #,256,512]
efn_phi_sizes = [50,100,250] #,500]

for l in pfn_latent_sizes:
    for phi in pfn_phi_sizes:
        for n in efn_latent_sizes:
            for d in efn_phi_sizes:
                label = f'latent{n}_phi{d}_EFN_by_latent{l}_phi{phi}_pfn'
                with open(f'batch/train_{label}.sh', 'w') as run_script:

                    run_script.write('#!/bin/bash\n\n')

                    run_script.write('#SBATCH -N 1\n')
                    run_script.write('#SBATCH -n 1\n')
                    run_script.write('#SBATCH --mem=64G\n')
                    run_script.write('#SBATCH -t 12:00:00\n')

                    # https://docs.ccv.brown.edu/oscar/gpu-computing/submit-gpu
                    run_script.write('#SBATCH -p gpu --gres=gpu:1\n')

                    run_script.write('#SBATCH -o logs/slurm-%j.out\n')
                    run_script.write('#SBATCH -J EFN_'+label+'\n')
                    run_script.write('\n')

                    run_script.write('source QG/bin/activate\n\n')

                    cmd =  'python3 efn_distill.py '
                    cmd += '-nEpochs=200 '
                    cmd += '-batchSize=500 '
                    cmd += '-latentSizeTeacher='+str(l)+' '
                    cmd += '-phiSizesTeacher='+str(phi)+' '
                    cmd += '-latentSizeStudent='+str(n)+' '
                    cmd += '-phiSizesStudent='+str(d)+' '
                    cmd += '-doEarlyStopping '
                    cmd += '-patience=20 '
                    cmd += '-usePIDs '

                    print(cmd)
                    run_script.write(cmd+'\n')

                print('sbatch batch/train_'+label+'.sh')
                os.system(f'sbatch batch/train_{label}.sh')
