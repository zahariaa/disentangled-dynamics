#!/bin/bash
# 2018-11-09 AZ Created
# first argument is distribution type
# second argument is experiment type
# e.g.,
# ./train-gammas.sh 1
# ./train-gammas.sh 1-36
#
# To release held jobs:
#
#for (( v = 1448325; v<=1448366; v++)) do qrls $v; done

# parameters
walltime="02:59:00"
memory="3GB" # PER CPU
numnodes=8
parfor=true #false
requestgpu=1 #0
# script-specific parameters (6 parameter grid)
n_latent=4 #10
max_iter=50000

### BUILD PYTHON JOB
for var in "$@"
do
   if [ $var == $1 ]; then
      if [ $(echo $var | grep -e "-") ]; then
         first=${var%-*}
         total=${var##*-}
      else
         first=$var
         total=$var
      fi
   fi
done
# first argument is index of betas x gammas (1-36)
jobname="dynBG"
pyparams="--beta_is_normalized=True --max_iter=$max_iter --n_latent=$n_latent"

### set up directories
basedir=~/results/
function=inertia_bVAE
scriptdir=$basedir$function

if [ ! -d "$scriptdir" ]
then
   mkdir $scriptdir
   mkdir $scriptdir/r
   mkdir $scriptdir/e
   mkdir $scriptdir/o
fi

######### construct SLURM (job array) script
jfile="${scriptdir}/$jobname.s"

pbtxt1="#!/bin/bash\n#"
pbtxt2="#SBATCH --account=nklab\n#SBATCH --job-name=$jobname"
pbtxt3="#SBATCH --nodes=1\n#SBATCH --cpus-per-task=$numnodes\n#SBATCH --gres=gpu:$requestgpu"
pbtxt4="#SBATCH --time=$walltime"
pbtxt5="#SBATCH --mem-per-cpu=$memory"
pbtxt8="#SBATCH --output=${scriptdir}/o/${jobname}%a.txt"
pbtxt9="#SBATCH --error=${scriptdir}/e/${jobname}%a.txt"
pbtxt9a="source activate torchaxon\nbetas=(0.001 0.0055 0.0302 0.1657 0.9103 5.0)\ngammas=(0.0 0.2 0.4 0.6 0.8 1.0)\nnb=\${#betas[@]}\ncd ~/dgm/src/$function/"

# MAIN PYTHON EXECUTION SLURM line
#Command to execute Python code


pbtxt10="a=\$SLURM_ARRAY_TASK_ID\nib=\$(( \$((a-1)) % \$nb))\nig=\$(( \$((a-1)) / \$nb))"
pbtxt11="" #echo \$a,\$ib,\$ig,\${betas[\$ib]},\${gammas[\$ig]}"
pbtxt12="python main.py $pyparams --beta=\${betas[\$ib]} --gamma=\${gammas[\$ig]}" # > ${scriptdir}/r/$jobname.txt"

# WRITE SLURM SCRIPT
echo -e "${pbtxt1}\n${pbtxt2}\n${pbtxt3}\n${pbtxt4}\n${pbtxt5}\n${pbtxt8}\n${pbtxt9}\n${pbtxt9a}\n\n${pbtxt10}\n${pbtxt11}\n${pbtxt12}\n" > ${jfile}

# submit array job
jid=`sbatch --array=$first-$total $jfile`
echo $jid

