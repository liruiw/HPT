set -x
set -e
DATE="`date +'%d_%m_%Y_%H_%M_%S'`_$$"
STAT_DIR=${0}
STAT_DIR="${STAT_DIR##*/}"
STAT_DIR="${STAT_DIR%.sh}"

echo "RUNNIGN $STAT_DIR!"
PRETRAINED=${1}
PRETRAINEDCMD=${2}
NUM_RUN=${3-"1"}


# train
ADD_ARGUMENT=${4-""}

# Loop through the arguments starting from the 5th
for arg in "${@:5}"; do
  ADD_ARGUMENT+=" $arg"  # Concatenate each argument
done

CMD="HYDRA_FULL_ERROR=1  time  python   -m hpt.run  \
		script_name=$STAT_DIR \
		env=mujoco_metaworld \
		train.pretrained_dir=output/$PRETRAINED  \
		dataset.episode_cnt=200 \
		domains=mujoco_metaworld_task20		\
		+datasetsize=metaworld_20 \
		train.freeze_trunk=False  \
		output_dir=output/${DATE}_${PRETRAINEDCMD}  \
		$ADD_ARGUMENT"

eval $CMD
# eval
HYDRA_FULL_ERROR=1  time  python   -m hpt.run_eval --multirun \
		--config-name=config \
		--config-path=../output/${DATE}_${PRETRAINEDCMD}   \
		train.pretrained_dir="'output/${DATE}_${PRETRAINEDCMD}'" \
	  	seed="range(3)" \
		hydra.sweep.dir=output/${DATE}_${PRETRAINEDCMD}  \
 		hydra/launcher=joblib \
		hydra.launcher.n_jobs=3