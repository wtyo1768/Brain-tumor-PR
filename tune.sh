
echo '********HyperParams*******'

LR=1e-3
EPOCHS=5
BATCH_SIZE=8

# seed=46
echo "LR : $LR"
echo "EPOCHS : $EPOCHS"
echo "BATCH_SIZE : $BATCH_SIZE"
echo '**************************'


# ['T1', 'T1c', 'T2', 'Flair']
for i in $(seq 0 5);
do
    python3 run.py  \
        --gpu 1 \
        --max_epochs $EPOCHS\
        --BATCH_SIZE $BATCH_SIZE\
        --LR $LR \
        --dtype 'T1c'
done