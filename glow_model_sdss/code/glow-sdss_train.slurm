#!/bin/bash
#SBATCH --job-name=sdss_train    # nom du job
# Il est possible d'utiliser une autre partition que celle par défaut
# en activant l'une des 5 directives suivantes :
##SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
#SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --qos=qos_gpu-dev  # debug 2h max
##SBATCH --qos=qos_gpu-t4    # 100h max 
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH --partition=gpu_p4          # decommenter pour la partition gpu_p4 (GPU A100 40 Go)
##SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
# Ici, reservation de 3x10=30 CPU (pour 3 taches) et de 3 GPU (1 GPU par tache) sur un seul noeud :
#SBATCH --nodes=1                    # nombre de noeud
##SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU par tache (soit 1/4 ou 1/8 des GPU du noeud suivant
# la partition), l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour chaque tache:
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 du noeud ici)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=6           # nombre de CPU par tache pour gpu_p4 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=12:00:00              # temps d’execution maximum demande (HH:MM:SS)
#SBATCH --output=sdss_train_%j.out # nom du fichier de sortie
#SBATCH --error=sdss_train_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition
#module load cpuarch/amd
 
# Chargement des modules
module load pytorch-gpu/py3/2.0.0
 
# Echo des commandes lancees
set -x
 
# Pour la partition "gpu_p5", le code doit etre compile avec les modules compatibles
# Execution du code avec binding via bind_gpu.sh : 1 GPU par tache

# dataset="sdss"
group="B"  # set "A" or "B"

#below ntrain => number of data training size  (ex 100000, 10000, 1000) 
ntrain="100000"
# specify <rootdir>
dataroot='<rootdir>/datasets/sdss_'${nrain}'_'  #the group is glued after to get sdss_0_<group>
output_dir="./results/results_glow_model_"+${ntrain}+"/output_"${ntrain}"_2ndmod" # do not finish by "_" and "/" as  group is glued after to get output_<group>


batch=32
iter=300000
lr=1e-4  # was 1e-4 1st turn, 1e-5 2nd turn and so on
seed=42
img_size=64
n_bits=5

n_block=3
affine="--affine"
temp=1.0

# for a freh optimization
saved_model=""
saved_optimize=""
fresh="--fresh"

# specif <dir> to get right best model so far
#saved_model="--saved_model <dir>/model_best_saved.pt"
#saved_optimizer="--saved_optimizer <dir>/optim_best_saved.pt"
#fresh=""



srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u train.py --group ${group} ${fresh} --dataroot ${dataroot} --seed ${seed} --output_dir ${output_dir} --n_block ${n_block} --n_bits ${n_bits} ${affine}  --batch ${batch} --iter ${iter}  --lr ${lr} --img_size ${img_size} --temp ${temp} ${saved_model}  ${saved_optimizer}




