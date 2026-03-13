shapeBatch=1
seqlen=32768
kNumHead=32
vNumHead=32
kHeadDim=128
vHeadDim=128
isVariedLen=1 #0 #1
tokenBatch=309 #1 #309
chunkSize=64
scale=0.08838834764831845
dtype="bf16"

device=2
useActualInput=1
useActualOutput=1

dataPath="/home/z00958757/dump_data/kernel_io_cu_64_1225_1_32768_32_128/chunk_fwd_o_io.pt"

echo 'Case: batch=' $batch ' seqlen=' $seqlen ' kNumHead=' $kNumHead  ' vNumHead=' $vNumHead ' kHeadDim=' $kHeadDim ' vHeadDim=' $vHeadDim ' isVariedLen=' $isVariedLen ' chunkSize=' $chunkSize ' dtype=' $dtype
python3 test_fwd_o.py $shapeBatch $seqlen $kNumHead $vNumHead $kHeadDim $vHeadDim $isVariedLen $tokenBatch $chunkSize $scale "$dtype" $useActualInput $useActualOutput $dataPath
python data_compare_o.py $dtype