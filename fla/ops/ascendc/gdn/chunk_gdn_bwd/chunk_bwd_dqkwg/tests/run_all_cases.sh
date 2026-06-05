# bash run.sh run case_step2_07
# bash run.sh run case_step2_08
# bash run.sh run case_step2_09
# bash run.sh run case_step2_10
# bash run.sh run case_step2_01
# bash run.sh run case_step2_02
# bash run.sh run case_step2_03
# bash run.sh run case_step2_04
# bash run.sh run case_step2_05
# bash run.sh run case_step2_06

keys=($(python3 -c "
import cases
print(' '.join(cases.cases.keys()))
"))

# 循环执行命令
for i in "${!keys[@]}"; do
    key="${keys[$i]}"
    echo "===================================Processing: $key================================================================"
    
    # if [ $i -eq 0 ]; then
    #     echo "First key - compiling first..."
    #     bash run.sh compile "$key"
    # fi
    
    bash run.sh run "$key"
done

# # bash run.sh run case_21  # zj
# # bash run.sh run case_22  # zj
# # bash run.sh run case_17
# # bash run.sh run case_19
# bash run.sh run case_09
# bash run.sh run case_10
# # bash run.sh run case_01
# # bash run.sh run case_02
# # bash run.sh run case_03
# # bash run.sh run case_04
# # bash run.sh run case_05
# bash run.sh run case_06
# # bash run.sh run case_07
# # bash run.sh run case_08
# bash run.sh run case_11
# bash run.sh run case_12
# bash run.sh run case_13
# bash run.sh run case_14
# bash run.sh run case_15
# bash run.sh run case_16
# bash run.sh run case_18
# bash run.sh run case_20
