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
walltime="00:59:00"
memory="2GB" # PER CPU
numnodes=8
parfor=true #false
requestgpu=1 #0
# script-specific parameters (6x6 parameter grid)
n_latent=4
max_iter=50000
# normalized_beta_values = np.logspace(np.log(.001), np.log(5), 6, base=np.e)
betas=(0.001 0.0055 0.0302 0.1657 0.9103 5.0)
# gamma_values = np.logspace(-4, 4, 5, base=10)
# gamma_values = np.insert(gamma_values,0,0)
gammas=(0.0 0.0001 0.01 1.0 100.0 10000.0)

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
ib=$((($first-1) % ${#betas[@]}))
ig=$((($first-1) / ${#betas[@]}))
jobname="dynB${betas[$ib]}G${gammas[$ig]}"
pyparams="--beta_is_normalized=True --max_iter=$max_iter --n_latent=$n_latent"
pyparams="$pyparams --beta=${betas[$ib]} --gamma=${gammas[$ig]}"

### set up directories
basedir=~/results/
function=dynamic_simple
scriptdir=$basedir$function

if [ ! -d "$scriptdir" ]
then
   mkdir $scriptdir
   mkdir $scriptdir/r
   mkdir $scriptdir/e
#   mkdir $scriptdir/o
fi

######### construct SLURM (job array) script
jfile="${scriptdir}/$jobname.s"

pbtxt1="#!/bin/bash\n#"
pbtxt2="#SBATCH --account=nklab\n#SBATCH --job-name=$jobname"
pbtxt3="#SBATCH --nodes=1\n#SBATCH --cpus-per-task=$numnodes\n#SBATCH --gres=gpu:$requestgpu"
pbtxt4="#SBATCH --time=$walltime"
pbtxt5="#SBATCH --mem-per-cpu=$memory"
pbtxt8="" #SBATCH --output=${scriptdir}/o/${jobname}%a.txt"
pbtxt9="#SBATCH --error=${scriptdir}/e/${jobname}%a.txt"
pbtxt9a="\nmodule load anaconda/3-2018.12\nsource activate pytorchenv\nimport torch\n"

# MAIN PYTHON EXECUTION SLURM line
#Command to execute Python code
pbtxt10="cd ~/projects/dgm/src/dynamic_simple"
pbtxt11="./main.py $pyparams > ${scriptdir}/r/$jobname.txt"
pbtxt12=""

# WRITE SLURM SCRIPT
echo -e "${pbtxt1}\n${pbtxt2}\n${pbtxt3}\n${pbtxt4}\n${pbtxt5}\n${pbtxt8}\n${pbtxt9}\n${pbtxt9a}\n\n${pbtxt10}\n${pbtxt11}\n${pbtxt12}\n" > ${jfile}

# submit array job
jid=`sbatch --array=$first-$total $jfile`
echo $jid

