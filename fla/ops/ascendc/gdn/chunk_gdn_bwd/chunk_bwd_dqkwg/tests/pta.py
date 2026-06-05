from chunk_bwd_dqkwg_cpu import *
import torch
import torch_npu
import fla_npu
# -------------------------------------------------------------------------
# 使用示例 / 验证
# -------------------------------------------------------------------------
def tensor_size_gb(tensor, precision=2):
    """
    获取 tensor 的大小（GB）
    
    Args:
        tensor: PyTorch tensor
        precision: 小数点精度
    
    Returns:
        float: 大小（GB）
    """
    size_bytes = tensor.element_size() * tensor.numel()
    size_gb = size_bytes / (1024 ** 3)
    return round(size_gb, precision)
if __name__ == "__main__":
    
    case_name = "case_22"
    if len(sys.argv) > 1:
        case_name = sys.argv[1].rstrip('\r\n')  # 去除末尾的 \r 和 \n
        print(f"[test.py] case name: {case_name}")

    # 简单的形状参数
    K, V = 128, 128
    calc_type = torch.float32
    isVarLen = False
    chunk_size = 128
    from cases import get_case_info
    case = get_case_info(case_name)
    device_id = 6
    

    dtype = torch.float16
    Gtype = torch.float16
    B, H = 4, 8
    T = 1024
    scale = 0.088
    if isVarLen:
        cu_seqlens = torch.cumsum(torch.tensor([0, 3, 64, 63, 66, 260]), dim=0)
    else:
        cu_seqlens = None
    if case != None:
        dtype = case["dtype"]
        Gtype = case["Gtype"]
        B, HV, HK = case["B"], case["HV"], case["HK"]
        chunk_size = case["chunk_size"]
        cu_seqlens = case["cu_seqlens"]
        cu_seqlens_torch = torch.tensor(cu_seqlens) if cu_seqlens is not None else None

        if case["cu_seqlens"] is None:
            isVarLen = False
        else:
            isVarLen = True
        T = case["T"]
        scale = case["scale"]
        K = case["K"]
        V = case["V"]
        # V = 256 ########################################手动修改V，测试二阶段用例


    if isVarLen:
        B = 1  ##变长只支持B=1
        T = cu_seqlens_torch[-1]
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        num_chunks = len(chunk_indices) // 2
        # print("chunk_indices",chunk_indices)
    else:
        chunk_indices = None
        num_chunks = (T + chunk_size - 1) // chunk_size
    
    data_path = "/data/huangjunzhe/GDN/result/result_newg"
    RANDOM_DATA = True
    RUN_CPU = True
    SAVE_FILES = True
    BENCHMARK = True

    if SAVE_FILES:
        os.makedirs(f'{data_path}/{case_name}/in/', exist_ok=True)
        os.makedirs(f'{data_path}/{case_name}/out/', exist_ok=True)
    if RANDOM_DATA:
        q = torch.randn(B,HK,T,K, dtype=dtype, requires_grad=True)  # std≈5e-6#torch.randn([B, T, H, K], dtype=dtype)
        k = torch.randn(B,HK,T,K, dtype=dtype, requires_grad=True)   # torch.randn([B, T, H, K], dtype=dtype)
        v = torch.randn(B,HV,T,V, dtype=dtype, requires_grad=True)  # torch.randn([B, T, H, V], dtype=dtype)

        # g = torch.randn(B,T,H, dtype=dtype, requires_grad=True) * 5e-2   # torch.randn([B, T, H], dtype=Gtype)
        g = -torch.sort(torch.rand(B*T*HV) * 10, descending=False)[0].reshape((B,HV,T)).to(Gtype)    #G必须递减且为负数
        # print("g",g)
        do = torch.randn(B,HV,T,V, dtype=dtype, requires_grad=True)  # torch.randn([B, T, H, V], dtype=dtype)
        
        dv = torch.randn(B,HV,T,V, dtype=dtype, requires_grad=True)  # torch.randn([B, T, H, V], dtype=dtype)
        w = torch.randn(B,HK,T,K, dtype=dtype, requires_grad=True)   # torch.randn([B, T, H, K], dtype=dtype)

        h = torch.randn(B, HV, num_chunks, K, V, dtype=dtype, requires_grad=True)  # torch.randn([B, num_chunks, H, K, V], dtype=dtype)
        dh = torch.randn(B, HV, num_chunks, K, V, dtype=dtype, requires_grad=True)  # torch.randn([B, num_chunks, H, K, V], dtype=dtype)
        if SAVE_FILES:
            torch.save(q, f"{data_path}/{case_name}/in/q_cpu.pt")
            torch.save(k, f"{data_path}/{case_name}/in/k_cpu.pt")
            torch.save(v, f"{data_path}/{case_name}/in/v_cpu.pt")
            torch.save(g, f"{data_path}/{case_name}/in/g_cpu.pt")
            torch.save(do, f"{data_path}/{case_name}/in/do_cpu.pt")
            torch.save(dv, f"{data_path}/{case_name}/in/dv_cpu.pt")
            torch.save(w, f"{data_path}/{case_name}/in/w_cpu.pt")
            torch.save(h, f"{data_path}/{case_name}/in/h_cpu.pt")
            torch.save(dh, f"{data_path}/{case_name}/in/dh_cpu.pt")
    else:
        # q=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/q_cpu.pt", weights_only=False)
        # k=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/k_cpu.pt", weights_only=False)
        # v=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/v_new_cpu.pt", weights_only=False)
        # w=torch.empty_like(q)
        # g=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/g_cpu.pt", weights_only=False)
        # h=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/h_cpu.pt", weights_only=False)
        # dv=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/dv_cpu.pt", weights_only=False)
        # do=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/do_cpu.pt", weights_only=False)
        # dh=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/dh_cpu.pt", weights_only=False)
        q=torch.load(f"{data_path}/{case_name}/in/q_cpu.pt", weights_only=False)
        k=torch.load(f"{data_path}/{case_name}/in/k_cpu.pt", weights_only=False)
        v=torch.load(f"{data_path}/{case_name}/in/v_cpu.pt", weights_only=False)
        w=torch.empty_like(q)
        g=torch.load(f"{data_path}/{case_name}/in/g_cpu.pt", weights_only=False)
        h=torch.load(f"{data_path}/{case_name}/in/h_cpu.pt", weights_only=False)
        dv=torch.load(f"{data_path}/{case_name}/in/dv_cpu.pt", weights_only=False)
        do=torch.load(f"{data_path}/{case_name}/in/do_cpu.pt", weights_only=False)
        dh=torch.load(f"{data_path}/{case_name}/in/dh_cpu.pt", weights_only=False)

    q = q.to(dtype).to(calc_type)
    k = k.to(dtype).to(calc_type)
    v = v.to(dtype).to(calc_type)
    h = h.to(dtype).to(calc_type)
    g = g.to(Gtype).to(calc_type)
    do = do.to(dtype).to(calc_type)
    dh = dh.to(dtype).to(calc_type)
    dv = dv.to(dtype).to(calc_type)
    w = w.to(dtype).to(calc_type)
    print("entering chunk_bwd_dqkwg")
    print(f"q: {q.shape} {dtype} => {q.dtype}")
    print(f"k: {k.shape} {dtype} => {k.dtype}")
    print(f"v: {v.shape} {dtype} => {v.dtype}")
    print(f"w: {w.shape} {dtype} => {w.dtype}")
    print(f"g: {g.shape} {Gtype} => {g.dtype}")
    print(f"h: {h.shape} {dtype} => {h.dtype}")
    print(f"dv: {dv.shape} {dtype} => {dv.dtype}")
    print(f"do: {do.shape} {dtype} => {do.dtype}")
    print(f"dh: {dh.shape} {dtype} => {dh.dtype}")
    if cu_seqlens == None:
        print("cu_seqlens is None")
    else:
        print(f"cu_seqlens: {cu_seqlens_torch.shape} {cu_seqlens_torch.dtype} {cu_seqlens_torch}")
        # print(f"chunk_indices: {chunk_indices.shape} {chunk_indices.dtype} {chunk_indices}")
    print(f"scale: {scale}")
    print(f"chunk_size: {chunk_size}")


    print("==============start NPU=============")
    torch_npu.npu.set_device(device_id)
    q_npu = q.to(dtype).npu()
    k_npu = k.to(dtype).npu()
    v_npu = v.to(dtype).npu()
    # w_npu = w.to(dtype).npu()
    g_npu = g.to(Gtype).npu()
    h_npu = h.to(dtype).npu()
    dv_npu = dv.to(dtype).npu()
    do_npu = do.to(dtype).npu()
    dh_npu = dh.to(dtype).npu()
    print(f"input size(GB): q {tensor_size_gb(q_npu)}, k {tensor_size_gb(k_npu)}, v {tensor_size_gb(v_npu)}, g {tensor_size_gb(g_npu)}, h {tensor_size_gb(h_npu)}, dv {tensor_size_gb(dv_npu)}, do {tensor_size_gb(do_npu)}, dh {tensor_size_gb(dh_npu)}")
    # cu_seqlens_npu = cu_seqlens if cu_seqlens is not None else None
    chunk_indices_npu = chunk_indices if cu_seqlens is not None else None
    down_tri = q_npu
    dq_npu, dk_npu, dw_npu, dg_npu = torch.ops.npu.npu_chunk_bwd_dqkwg(
        q_npu, k_npu, v_npu, g_npu, h_npu, do_npu, dh_npu, dv_npu, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices_npu, w=None, g_gamma=None, scale=scale, transpose_state_layout=None
       #q_npu, k_npu, v_npu, g_npu, h_npu, do_npu, dh_npu, dv_npu, chunk_size, cu_seqlens=cu_seqlens, w=None, g_gamma=None, chunk_indices=chunk_indices_npu, scale=scale, transpose_state_layout=None

    )
    dq_npu = dq_npu.cpu()
    dk_npu = dk_npu.cpu()
    dw_npu = dw_npu.cpu()
    dg_npu = dg_npu.cpu()

    # print("Output shapes:", dq.shape, dk.shape, dg.shape, dw.shape)
    print("dq_npu", dq_npu.shape, dq_npu.dtype)
    print("dk_npu", dk_npu.shape, dk_npu.dtype)
    print("dw_npu", dw_npu.shape, dw_npu.dtype)
    print("dg_npu", dg_npu.shape, dg_npu.dtype, dg_npu[0][0][-1])

    # print("dq_npu[0][0][-1]", dq_npu[0][0][-1])
    if RUN_CPU:
        print("=====start cpu=========")
        q = torch.transpose(q, 1, 2).to(dtype)
        k = torch.transpose(k, 1, 2).to(dtype)
        v = torch.transpose(v, 1, 2).to(dtype)
        w = torch.transpose(w, 1, 2).to(dtype)
        g = torch.transpose(g, 1, 2).to(Gtype)
        h = torch.transpose(h, 1, 2).to(dtype)
        dv = torch.transpose(dv, 1, 2).to(dtype)
        do = torch.transpose(do, 1, 2).to(dtype)
        dh = torch.transpose(dh, 1, 2).to(dtype)

        dq, dk, dw, dg = chunk_bwd_dqkwg_cpu(
            q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens_torch, chunk_size
        )
        if BENCHMARK:
            print("=====start cpu benchmark=========")
            dq_benchmark, dk_benchmark, dw_benchmark, dg_benchmark = chunk_bwd_dqkwg_cpu(
                q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens_torch, chunk_size, benchmark = True
            )
            dq_benchmark = torch.transpose(dq_benchmark, 1, 2).cpu()
            dk_benchmark = torch.transpose(dk_benchmark, 1, 2).cpu()
            dw_benchmark = torch.transpose(dw_benchmark, 1, 2).cpu()
            dg_benchmark = torch.transpose(dg_benchmark, 1, 2).cpu()
        # dq = dq.to(dtype)
        # dk = dk.to(dtype)
        # dw = dw.to(dtype)
        # dg = dg.to(Gtype)
        dq = torch.transpose(dq, 1, 2).cpu()
        dk = torch.transpose(dk, 1, 2).cpu()
        dw = torch.transpose(dw, 1, 2).cpu()
        dg = torch.transpose(dg, 1, 2).cpu()
        # print("dq[0][0][-1]", dq[0][0][-1])
        # print("dk", dk)
        # print("dw", dw)
        # print("dg", dg)

        print("dq", dq.cpu().shape, dq.cpu().dtype)
        print("dk", dk.cpu().shape, dk.cpu().dtype)
        print("dw", dw.cpu().shape, dw.cpu().dtype)
        print("dg", dg.cpu().shape, dg.cpu().dtype)

        type_dict = {torch.float16:"float16", torch.float32:"float32",torch.bfloat16:"bf16"}

        if SAVE_FILES:
            if RANDOM_DATA:
                torch.save(dq.detach(),f"{data_path}/{case_name}/out/dq_cpu.pt")
                torch.save(dk.detach(),f"{data_path}/{case_name}/out/dk_cpu.pt")
                torch.save(dw.detach(),f"{data_path}/{case_name}/out/dw_cpu.pt")
                torch.save(dg.detach(),f"{data_path}/{case_name}/out/dg_cpu.pt")
                print(f"cpu files saved to {data_path}/{case_name}/out/ .")
                if BENCHMARK:
                    torch.save(dq_benchmark.detach(),f"{data_path}/{case_name}/out/dq_cpu_benchmark.pt")
                    torch.save(dk_benchmark.detach(),f"{data_path}/{case_name}/out/dk_cpu_benchmark.pt")
                    torch.save(dw_benchmark.detach(),f"{data_path}/{case_name}/out/dw_cpu_benchmark.pt")
                    torch.save(dg_benchmark.detach(),f"{data_path}/{case_name}/out/dg_cpu_benchmark.pt")
                    print(f"cpu benchmark files saved to {data_path}/{case_name}/out/ .")
                
    if SAVE_FILES:
        torch.save(dq_npu.detach(),f"{data_path}/{case_name}/out/dq_npu.pt")
        torch.save(dk_npu.detach(),f"{data_path}/{case_name}/out/dk_npu.pt")
        torch.save(dw_npu.detach(),f"{data_path}/{case_name}/out/dw_npu.pt")
        torch.save(dg_npu.detach(),f"{data_path}/{case_name}/out/dg_npu.pt")
        print(f"npu files saved to {data_path}/{case_name}/out/ .")