#usage: bash run.sh run case_step2_01
import case_extra_info
import torch
import random

def generate_increasing_sequence(start, end, count):
    """
    生成递增随机序列
    start: 起始数字
    end: 结束数字  
    count: 生成个数（包括首尾）
    """
    if count < 2:
        return [start]
    
    # 在 start 和 end 之间生成 count-2 个随机数
    middle = sorted(random.sample(range(start + 1, end), count - 2))
    return [start] + middle + [end]


cases = {   #B,HV,HK,T,chunk_size,dtype,Gtype,scale,cu_seqlens, K,V
        "case_fast": [2,8,4,512,128,torch.bfloat16,torch.float32,0.088,None,128,256],
        "case_01": [64,8,8,1024,64,torch.float16,torch.float16,0.088,None,128,128],
        "case_02": [32,16,16,2048,64,torch.bfloat16,torch.bfloat16,0.0625,None,128,128],
        "case_03": [16,32,32,4096,64,torch.float16,torch.float16,0.0442,None,128,128],
        "case_04": [8,32,32,8192,64,torch.bfloat16,torch.bfloat16,0.03125,None,128,128],
        "case_05": [128,4,4,1024,64,torch.float16,torch.float16,0.088,None,128,128],
        "case_06": [64,4,4,4096,64,torch.bfloat16,torch.bfloat16,0.0625,None,128,128],
        "case_07": [32,16,16,8192,64,torch.float16,torch.float16,0.0442,None,128,128],
        "case_08": [16,32,32,16384,64,torch.bfloat16,torch.bfloat16,0.03125,None,128,128],
        "case_09": [64,8,8,2048,128,torch.float16,torch.float16,0.0625,None,128,128],
        "case_10": [32,16,16,4096,128,torch.bfloat16,torch.bfloat16,0.0442,None,128,128],
        "case_11": [16,32,32,8192,128,torch.float16,torch.float16,0.03125,None,128,128],
        "case_12": [8,32,32,8192,128,torch.bfloat16,torch.bfloat16,0.0221,None,128,128],  #C12
        "case_13": [1,4,4,1024,64,torch.float16,torch.float16,0.088,None,128,128],
        "case_14": [48,8,8,2048,64,torch.bfloat16,torch.bfloat16,0.0625,None,128,128],
        "case_15": [24,16,16,4096,64,torch.float16,torch.float16,0.0442,None,128,128],
        "case_16": [12,32,32,8192,64,torch.bfloat16,torch.bfloat16,0.03125,None,128,128],
        "case_17": [1,16,16,32768,64,torch.float16,torch.float32,0.0625,[0,16,20000,30000,32768],128,128],      # V1
        "case_18": [1,8,8,65536,64,torch.bfloat16,torch.bfloat16,0.0625,[0,16,20000,65536],128,128],
        "case_19": [1,32,32,65536,64,torch.float16,torch.float32,0.0442,[0,16,20000,50000,65536],128,128],
        "case_20": [1,32,32,262144,64,torch.bfloat16,torch.bfloat16,0.03125,[0,16,20000,50000,65536,210000,262144],128,128],
        "case_21": [21,32,32,195,64,torch.float16,torch.float16,0.03125,None,128,128],
        "case_22": [1,32,32,7909,64,torch.bfloat16,torch.bfloat16,0.03125,[0,1024,2168,3087,4096,7909],128,128],
        "case_23": [1,32,32,65536,64,torch.bfloat16,torch.bfloat16,0.32,case_extra_info.pj_cu_seqlens,128,128],
        "case_24": [1,32,32,65536,128,torch.bfloat16,torch.bfloat16,0.32,case_extra_info.pj_cu_seqlens,128,128],
        "case_25": [1,16,8,1024,64,torch.bfloat16,torch.float32,0.088, [0, 57, 143, 187, 197, 227],128,256],
        "case_step2_01": [1,32,16,16384,64,torch.float16,torch.float32,0.03125, generate_increasing_sequence(0, 16384, 128),128,256],
        "case_step2_02": [1,63,21,16384,64,torch.bfloat16,torch.float32,0.03125, [0, 1066, 2048, 5000, 9000, 10000, 12000, 14000, 16384],128,256],
        "case_step2_03": [1,32,8,65536,128,torch.float16,torch.float32,0.03125, generate_increasing_sequence(0, 65536, 172),128,256],
        "case_step2_04": [1,32,16,65536,64,torch.bfloat16,torch.float32,0.03125, generate_increasing_sequence(0, 65536, 668),128,128],
        "case_step2_05": [1,32,4,65536,64,torch.float16,torch.float32,0.03125, generate_increasing_sequence(0, 65536, 17),128,128],
        "case_step2_06": [1,64,2,65519,64,torch.bfloat16,torch.float32,0.03125, generate_increasing_sequence(0, 65519, 30),128,256],
        "case_step2_07": [1,32,16,4096,64,torch.float16,torch.float32,0.03125, None,128,256],
        "case_step2_08": [16,63,21,2048,64,torch.bfloat16,torch.float32,0.03125, None,128,256],
        "case_step2_09": [711,32,4,196,128,torch.float16,torch.float32,0.03125, None,128,128],
        "case_step2_10": [176,64,2,24,64,torch.bfloat16,torch.float32,0.03125, None,128,256],
        "case_step2_11": [1,48,16,65536,64,torch.float16,torch.float32,0.03125, generate_increasing_sequence(0, 65536, 667),128,256],
        "case_step2_12": [1,48,16,65536,128,torch.bfloat16,torch.float32,0.03125, generate_increasing_sequence(0, 65536, 13),128,256],
}
def error_exit(msg):
    print(f"[error_exit] {msg}")
    exit(1)
def verify_case(case_name, case):
    if (not case["cu_seqlens"] is None) and case["B"] != 1:
            error_exit(f"{case_name} varlen only support B = 1")
    if case["HV"] % case["HK"] != 0 or case["HV"] < case["HK"]:
            error_exit(f"{case_name} HV must be divisible by HK")
    if case["K"] != 128 or (case["V"] != 128 and case["V"] != 256):
            error_exit(f"{case_name} K must be 128 and V must be 128 or 256")
    return True
def get_case_info(case_name):
    case = {#B,HV,HK,T,chunk_size,dtype,Gtype,scale,cu_seqlens, K,V
        "B": cases[case_name][0],
        "HV": cases[case_name][1],
        "HK": cases[case_name][2],
        "T": cases[case_name][3],
        "chunk_size": cases[case_name][4],
        "dtype": cases[case_name][5],
        "Gtype": cases[case_name][6],
        "scale": cases[case_name][7],
        "cu_seqlens": cases[case_name][8],
        "K": cases[case_name][9],
        "V": cases[case_name][10]
    }
    if verify_case(case_name,case):
        return case
    else:
        return None
