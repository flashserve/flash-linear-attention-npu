batch=1 #309
seqlen=32768
kNumHead=32
vNumHead=32
kHeadDim=128
vHeadDim=128
isVariedLen=0 #1
chunkSize=64
dtype="bf16"
useActualInput=1
useActualOutput=1
device=2
useInitialState=0
storeFinalState=0
dataPath="/home/z00958757/dump_data/kernel_io_64_1225_1_32768_32_128/chunk_gated_delta_rule_fwd_h_io.pt"

echo 'Case: batch=' $batch ' seqlen=' $seqlen ' kNumHead=' $kNumHead  ' vNumHead=' $vNumHead ' kHeadDim=' $kHeadDim ' vHeadDim=' $vHeadDim ' isVariedLen=' $isVariedLen ' chunkSize=' $chunkSize ' dtype=' $dtype
python3 test_fwd_h.py $batch $seqlen $kNumHead $vNumHead $kHeadDim $vHeadDim $isVariedLen $chunkSize $useInitialState $storeFinalState "$dtype" $useActualInput $useActualOutput $dataPath
python data_compare_h.py $dtype