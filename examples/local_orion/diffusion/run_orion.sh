export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

ROOT_DIR=../../../
CONFIG=../../config_files/diffusion/config_diffusion_mlp_orion.yaml
DATA_DIR=${ROOT_DIR}/data/si_diffusion_small
PROCESSED_DATA=${DATA_DIR}/processed
DATA_WORK_DIR=./tmp_work_dir/

orion -v hunt --config orion_config.yaml \
    python ${ROOT_DIR}/crystal_diffusion/train_diffusion.py \
    --config $CONFIG \
    --data $DATA_DIR \
    --processed_datadir $PROCESSED_DATA \
    --dataset_working_dir $DATA_WORK_DIR \
    --output '{exp.working_dir}/{trial.id}/'
