import torch
from typing import Optional
import hashlib

def generate_cu_seqlens(
    cu_seqlens_len: int,
    total_length: int,
    device: Optional[torch.device] = None
) -> torch.LongTensor:
    """
    生成 cu_seqlens tensor
    
    Args:
        cu_seqlens_len: cu_seqlens tensor 的长度 (等于 batchsize + 1)
        total_length: 总长度 T，tensor 最后一个值
        device: 可选，指定 tensor 的设备
    
    Returns:
        cu_seqlens: [cu_seqlens_len] 的 tensor，其中：
            - cu_seqlens[0] = 0
            - cu_seqlens[i] - cu_seqlens[i-1] = 第 i 个序列的长度
            - cu_seqlens[cu_seqlens_len-1] = total_length
    """
    batchsize = cu_seqlens_len - 1
    import random
    remaining = total_length
    seq_lengths = []
    for i in range(batchsize - 1):
        min_len = 1
        max_len = remaining - (batchsize - 1 - i)
        if max_len < min_len:
            max_len = min_len
        seq_len = random.randint(min_len, max_len)
        seq_lengths.append(seq_len)
        remaining -= seq_len
    seq_lengths.append(remaining)
    
    cu_seqlens = [0]
    for seq_len in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    
    tensor = torch.tensor(cu_seqlens, dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def create_tensor(shape, dtype=torch.float16):
    return torch.rand(shape, dtype=dtype)

def bool_matrix_to_uint8(chunk_size):
    # 创建反上三角矩阵（上三角为0，下三角为1）
    bool_matrix = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
    bool_matrix = ~bool_matrix
    # print(f"==== bool_matrix.shape = {bool_matrix.shape} ",bool_matrix)
    # 将bool矩阵转换为uint8 (0或1)
    uint8_matrix = bool_matrix.to(torch.uint8)
    # 重塑为 (chunk_size, chunk_size//8, 8) 以便每8个bit打包
    reshaped = uint8_matrix.reshape(chunk_size, chunk_size // 8, 8)
    # 将每8个bit打包成一个uint8
    # bit0 * 1 + bit1 * 2 + bit2 * 4 + ... + bit7 * 128
    powers = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8)
    packed = (reshaped * powers).sum(dim=-1).to(torch.uint8)
    return packed

def compare_tensors_by_ratio(tensor1, tensor2, ratio_threshold=0.01, verbose=True):
    """
    对比两个tensor，检查差值与tensor1对应点位值的比值是否超过阈值
    
    Args:
        tensor1: 第一个tensor（作为基准）
        tensor2: 第二个tensor
        ratio_threshold: 比值阈值，默认0.01（1%）
        verbose: 是否打印详细结果，默认True
    
    Returns:
        bool: 是否所有点位都通过精度检查
    """
    # 检查形状是否相同
    if tensor1.shape != tensor2.shape:
        if verbose:
            print(f"错误: tensor形状不匹配!")
            print(f"  tensor1 shape: {tensor1.shape}")
            print(f"  tensor2 shape: {tensor2.shape}")
        return False
    
    # 计算绝对差值
    diff = torch.abs(tensor1 - tensor2)
    
    # 计算比值：差值 / (|tensor1| + epsilon)，避免除以0
    epsilon = 1e-8  # 小值避免除以0
    ratio = diff / (torch.abs(tensor1) + epsilon)
    
    # 找出超过阈值的点位
    mask = ratio > ratio_threshold
    failed_count = torch.sum(mask).item()
    total_count = tensor1.numel()
    
    if failed_count == 0:
        if verbose:
            print(f"✓ 精度对比成功!")
            print(f"  总点数: {total_count}")
            print(f"  最大相对差: {ratio.max().item():.6f}")
            print(f"  平均相对差: {ratio.mean().item():.6f}")
            print(f"  最大绝对差值: {diff.max().item():.6f}")
        return True
    else:
        if verbose:
            print(f"✗ 精度对比失败!")
            print(f"  总点数: {total_count}")
            print(f"  失败点数: {failed_count} ({failed_count/total_count*100:.2f}%)")
            print(f"  最大相对差: {ratio.max().item():.6f}")
            print(f"  平均相对差: {ratio.mean().item():.6f}")
            print(f"  最大绝对差值: {diff.max().item():.6f}")
            
            # 打印部分失败点位
            print(f"\n失败点位示例 (最多显示10个):")
            failed_indices = torch.nonzero(mask, as_tuple=True)
            display_count = min(10, failed_count)
            
            for i in range(display_count):
                idx = tuple(dim[i].item() for dim in failed_indices)
                val1 = tensor1[idx].item()
                val2 = tensor2[idx].item()
                val_diff = diff[idx].item()
                val_ratio = ratio[idx].item()
                print(f"  位置{idx}:")
                print(f"    tensor1={val1:.6f}, tensor2={val2:.6f}")
                print(f"    差值={val_diff:.6f}, 比值={val_ratio:.6f} ({val_ratio*100:.2f}%)")
        
        return False

def create_incremental_tensor(shape, dtype=torch.float16, start=1, step=1):
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    tensor = torch.arange(
        start, 
        start + total_elements * step, 
        step, 
        dtype=dtype
    ).reshape(shape)
    return tensor

def get_tensor_md5(tensor):
    tensor_np = tensor.cpu().numpy()
    md5_hash = hashlib.md5(tensor_np.tobytes()).hexdigest()
    return md5_hash

def compare_tensors_md5(tensor1, tensor2):
    md5_1 = get_tensor_md5(tensor1)
    md5_2 = get_tensor_md5(tensor2)
    
    return md5_1, md5_2, md5_1 == md5_2
