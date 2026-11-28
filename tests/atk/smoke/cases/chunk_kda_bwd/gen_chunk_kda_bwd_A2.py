"""Generate 200 accuracy-only KDA backward CPU dual-benchmark cases.

IDs 198/199 preserve model case1/case28 boundaries, saved/recompute respectively.
Run this file to regenerate JSON; --check verifies the frozen artifact.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
except ModuleNotFoundError as exc:
    if not (exc.name == "atk" or exc.name.startswith("atk.")):
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None

OP_NAME = "chunk_kda_bwd_A2"
PUBLIC_OP = "chunk_kda_bwd"
STANDARD = {"acc": {"cv_fused_double_benchmark": {
    "max_re_ratio": 5, "avg_re_ratio": 1.5, "root_mean_squared_ratio": 1.5}},
    "perf": "not_key", "mem": 1.1}

def _profile(name: str, **updates) -> dict:
    spec = {
        "name": name,
        "op": PUBLIC_OP,
        "route": "ascendc",
        "soc": "ascend910b",
        "mode": "dense",
        "B": 1,
        "H": 1,
        "T": 64,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "layout": "BNSD",
        "with_l2": False,
        "with_bias": True,
        "beta_mode": "plain",
        "beta_dtype": "bf16",
        "raw_g_dtype": "bf16",
        "a_log_dtype": "fp32",
        "disable_recompute": True,
        "implementation": "auto",
        "metadata": "auto",
        "safe_gate": True,
        "use_gate_in_kernel": True,
        "use_exp2": True,
        "state_v_first": False,
        "lower_bound": -0.5,
        "scale": 1 / math.sqrt(128),
        "qk_std": 0.08,
        "v_std": 0.20,
        "do_std": 0.20,
        "raw_g_mean": -1.0,
        "raw_g_std": 0.35,
    }
    spec.update(updates)
    if spec["mode"] == "varlen":
        spec["B"] = 1
        spec["T"] = sum(spec["seqlens"])
    spec["use_beta_sigmoid_in_kernel"] = spec["beta_mode"] != "plain"
    spec["allow_neg_eigval"] = spec["beta_mode"] == "two_sigmoid"
    spec["beta_range"] = [0.15, 1.85] if spec["allow_neg_eigval"] else [0.1, 0.9]
    spec["expected_finalize_key"] = 102 if spec["with_l2"] else 101
    if spec["disable_recompute"]:
        spec["expected_prepare_branch"] = (
            "NTD/varlen" if spec["mode"] == "varlen" else "BNSD/dense"
        )
        spec["expected_recompute_template"] = None
    else:
        spec["expected_prepare_branch"] = (
            "NTD/varlen" if spec["mode"] == "varlen" else "BNSD/dense"
        )
        # Recompute consumes the original gate dtype and converts on chip.
        spec["expected_recompute_template"] = (
            f"g={spec['raw_g_dtype']},beta={spec['beta_dtype']},K=128"
        )
    return spec


MODEL_CU = {1: [0, 684, 1364, 2044, 2716, 3384, 4051, 4716, 5375, 6033, 6686, 7334, 7980, 8619, 9255, 9881, 10506, 11130, 11753, 12375, 12996, 13614, 14224, 14833, 15438, 16040, 16642, 17241, 17831, 18419, 19007, 19595, 20168, 20736, 21304, 21870, 22435, 23000, 23557, 24111, 24664, 25214, 25762, 26309, 26853, 27395, 27937, 28476, 29014, 29551, 30088, 30624, 31158, 31688, 32214, 32739, 33260, 33779, 34298, 34816, 35329, 35839, 36347, 36851, 37354, 37854, 38342, 38826, 39307, 39784, 40260, 40729, 41197, 41664, 42127, 42588, 43045, 43500, 43952, 44404, 44847, 45289, 45730, 46168, 46604, 47040, 47476, 47911, 48345, 48778, 49211, 49644, 50076, 50508, 50940, 51371, 51801, 52230, 52659, 53083, 53499, 53914, 54327, 54738, 55145, 55551, 55955, 56354, 56751, 57146, 57538, 57919, 58299, 58679, 59057, 59435, 59812, 60188, 60561, 60934, 61297, 61658, 62019, 62378, 62736, 63094, 63451, 63805, 64158, 64511, 64864, 65211, 65551, 65891, 66229, 66555, 66878, 67189, 67497, 67803, 68108, 68411, 68713, 69014, 69314, 69614, 69914, 70212, 70510, 70805, 71100, 71393, 71686, 71979, 72272, 72565, 72858, 73151, 73442, 73733, 74023, 74313, 74603, 74893, 75183, 75473, 75763, 76052, 76341, 76630, 76918, 77206, 77494, 77782, 78070, 78358, 78646, 78934, 79222, 79510, 79797, 80084, 80371, 80658, 80944, 81230, 81516, 81802, 82088, 82374, 82660, 82946, 83232, 83517, 83802, 84082, 84354, 84625, 84890, 85147, 85400, 85652, 85898, 86138, 86378, 86617, 86854, 87089, 87323, 87556, 87788, 88017, 88243, 88469, 88693, 88915, 89133, 89338, 89542, 89737, 89929, 90121, 90312, 90503, 90694, 90884, 91072, 91258, 91443, 91626, 91809, 91991, 92173, 92354, 92535, 92715, 92895, 93074, 93252, 93430, 93606, 93781, 93953, 94125, 94291, 94457, 94623, 94787, 94949, 95109, 95269, 95429, 95589, 95747, 95905, 96061, 96216, 96369, 96520, 96669, 96817, 96962, 97107, 97252, 97397, 97541, 97684, 97826, 97965, 98104, 98242, 98379, 98515, 98650, 98785, 98918, 99051, 99184, 99317, 99450, 99582, 99714, 99845, 99976, 100106, 100236, 100366, 100496, 100625, 100752, 100879, 101004, 101129, 101253, 101377, 101501, 101624, 101747, 101870, 101993, 102114, 102233, 102352, 102471, 102589, 102706, 102823, 102937, 103050, 103163, 103275, 103387, 103498, 103609, 103720, 103831, 103942, 104053, 104164, 104275, 104386, 104497, 104608, 104719, 104830, 104941, 105052, 105163, 105274, 105385, 105496, 105607, 105718, 105829, 105940, 106050, 106160, 106270, 106380, 106490, 106600, 106710, 106820, 106930, 107040, 107150, 107260, 107370, 107480, 107590, 107700, 107810, 107920, 108030, 108139, 108248, 108357, 108466, 108575, 108684, 108793, 108902, 109011, 109120, 109229, 109338, 109447, 109556, 109665, 109774, 109883, 109991, 110099, 110207, 110315, 110423, 110531, 110639, 110747, 110855, 110963, 111071, 111179, 111287, 111395, 111503, 111610, 111717, 111824, 111931, 112038, 112145, 112252, 112359, 112466, 112573, 112680, 112787, 112893, 112999, 113105, 113211, 113317, 113423, 113529, 113635, 113741, 113847, 113953, 114059, 114164, 114269, 114374, 114479, 114584, 114689, 114794, 114899, 115004, 115109, 115213, 115317, 115421, 115525, 115629, 115733, 115837, 115941, 116045, 116148, 116251, 116354, 116457, 116560, 116663, 116766, 116869, 116972, 117075, 117178, 117281, 117384, 117487, 117590, 117693, 117796, 117898, 118000, 118102, 118204, 118306, 118408, 118510, 118612, 118714, 118816, 118918, 119020, 119121, 119222, 119323, 119424, 119525, 119626, 119727, 119828, 119929, 120030, 120131, 120232, 120333, 120434, 120535, 120636, 120737, 120837, 120937, 121037, 121137, 121237, 121337, 121437, 121537, 121637, 121737, 121837, 121937, 122037, 122137, 122237, 122337, 122437, 122537, 122637, 122737, 122837, 122937, 123037, 123137, 123237, 123337, 123436, 123535, 123634, 123733, 123832, 123931, 124030, 124129, 124228, 124327, 124426, 124525, 124624, 124723, 124822, 124921, 125020, 125118, 125216, 125314, 125412, 125510, 125608, 125706, 125804, 125902, 126000, 126098, 126196, 126294, 126392, 126490, 126588, 126686, 126784, 126882, 126980, 127078, 127176, 127274, 127372, 127470, 127568, 127666, 127764, 127861, 127958, 128055, 128152, 128249, 128346, 128443, 128540, 128637, 128734, 128831, 128928, 129025, 129122, 129219, 129316, 129413, 129510, 129606, 129702, 129798, 129894, 129990, 130086, 130182, 130278, 130374, 130470, 130566, 130662, 130758, 130854, 130950, 131046], 28: [0, 12892, 25396, 37858, 50241, 62526, 74670, 86779, 98757, 110727, 122492, 130995]}

def accuracy_profiles():
    profiles = []
    short_lengths = [1, 7, 15, 31, 32, 33, 63, 64, 65, 95, 127, 128, 129, 255, 257, 511, 512, 513]
    medium_lengths = [1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193]
    long_lengths = [16383, 16384, 16385, 24575, 24576, 32767, 32768, 32769]
    for i in range(198):
        saved = i % 2 == 0
        variable = (i // 2) % 2 == 1
        group = i // 4
        if i < 132:
            length = short_lengths[group % len(short_lengths)]
            heads = [1, 2, 3, 4, 5, 7, 8, 16, 32, 96][group % 10]
        elif i < 174:
            length = medium_lengths[group % len(medium_lengths)]
            heads = [1, 2, 3, 4, 8][group % 5]
        else:
            length = long_lengths[group % len(long_lengths)]
            heads = [1, 2, 4, 8][group % 4]
        # Both saved/recompute modes cover the single public layout.
        token_input = group % 2 == 1
        layouts = ("NTD", "TND") if variable else ("BNSD", "BSND")
        beta_mode = ["plain", "sigmoid", "two_sigmoid"][group % 3]
        if variable:
            pattern = group % 4
            if pattern == 0:
                lengths = [length]
            elif pattern == 1:
                lengths = [1, 63, length, 65]
            elif pattern == 2:
                lengths = [0, length, 0, 1, 0, 127]
            else:
                lengths = [max(1, length // 3), length, max(1, length // 7), 64, 33]
        else:
            lengths = None
        updates = dict(
            B=([1, 2, 3][group % 3] if not variable and i < 132 else 1),
            H=heads, T=length, mode="varlen" if variable else "dense",
            layout=layouts[int(token_input)],
            disable_recompute=saved, with_l2=(group // 3) % 2 == 1,
            beta_mode=beta_mode,
            beta_dtype="bf16" if beta_mode == "plain" and group % 2 == 0 else "fp32",
            raw_g_dtype="bf16" if group % 2 == 0 else "fp32",
            a_log_dtype="fp32", with_bias=group % 5 != 0,
            lower_bound=[-0.5, -1.0, -2.0, -5.0][(group // 6) % 4],
            scale=[1 / math.sqrt(128), 0.125, 1.0][(group // 4) % 3],
            metadata="explicit" if group % 2 else "auto", implementation="optimized",
        )
        if lengths is not None:
            updates["seqlens"] = lengths
        profiles.append(_profile(f"accuracy_{i:03d}_{'saved' if saved else 'recompute'}", **updates))
    # A nontrivial smoke: two full chunks plus a tail, L2 and sigmoid fused.
    profiles[0] = _profile("accuracy_000_smoke_saved_l2_sigmoid", H=3, T=129,
        with_l2=True, beta_mode="sigmoid", beta_dtype="fp32", raw_g_dtype="fp32",
        layout="BNSD", implementation="optimized")
    for original_id, saved, heads in ((1, True, 4), (28, False, 8)):
        cu = MODEL_CU[original_id]
        profiles.append(_profile(f"model_case{original_id}_{'saved' if saved else 'recompute'}",
            mode="varlen", seqlens=[b-a for a,b in zip(cu, cu[1:])],
            H=heads, layout="NTD", disable_recompute=saved,
            with_l2=True, beta_mode="sigmoid", beta_dtype="fp32", raw_g_dtype="fp32",
            metadata="explicit", implementation="optimized", source_model_case=original_id))
    for i, spec in enumerate(profiles):
        spec.update(case_id=i, package="accuracy", logical_id=i, seed_slot=0, seed=2026102700+i)
    return profiles


ACCURACY_PROFILES = accuracy_profiles()


def validate_profiles():
    assert len(ACCURACY_PROFILES) == 200
    assert [p["case_id"] for p in ACCURACY_PROFILES] == list(range(200))
    assert len({p["seed"] for p in ACCURACY_PROFILES}) == 200
    assert sum(p["disable_recompute"] for p in ACCURACY_PROFILES) == 100
    for p in ACCURACY_PROFILES:
        assert p["K"] == p["V"] == 128 and p["chunk_size"] == 64
        assert p["safe_gate"] and p["use_gate_in_kernel"] and p["use_exp2"]
        assert min(p["B"], p["H"], p["T"]) > 0
        if p["mode"] == "varlen":
            assert sum(p["seqlens"]) == p["T"] and min(p["seqlens"]) >= 0
        assert p["layout"] in (("NTD", "TND") if p["mode"] == "varlen" else ("BNSD", "BSND"))
        if p["use_beta_sigmoid_in_kernel"]:
            assert p["beta_dtype"] == "fp32"
    assert ACCURACY_PROFILES[198]["source_model_case"] == 1
    assert ACCURACY_PROFILES[199]["source_model_case"] == 28


def _input(name: str, dtype: str, value, *, input_type: str = "attr", shape=None) -> dict:
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
    }


def case_payload(spec: dict) -> dict:
    encoded = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "id": int(spec["case_id"]),
        "default_seed": int(spec["seed"]),
        "name": str(spec["name"]),
        "aclnn_name": "ChunkKdaBwd",
        "version": "v2.1",
        "api": "pytorch",
        "api_type": f"executor_{OP_NAME}",
        "expected_error_msg": None,
        "backward": False,
        "standard": STANDARD,
        "outputs": None,
        "inputs": [
            _input("low_precision_marker", "bf16", [0, 0], input_type="tensor", shape=[1]),
            _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
            _input("case_id", "int", int(spec["case_id"])),
            _input("case_spec", "non_param", encoded),
        ],
        "save_name": OP_NAME,
    }


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_kda_bwd_A2")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config):
            index = max(int(self.index) - 1, 0)
            if index >= len(ACCURACY_PROFILES):
                raise ValueError(
                    f"Use GEN_CASES_DTYPE_NUMBERS={len(ACCURACY_PROFILES)} for {OP_NAME}"
                )
            spec = ACCURACY_PROFILES[index]
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = spec["name"]
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "case_id":
                    cfg.range_values = index
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(
                        spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
            return case_config



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    validate_profiles()
    records = [case_payload(p) for p in ACCURACY_PROFILES]
    path = Path(__file__).with_name(f"atk_{OP_NAME}.json")
    if args.check:
        assert json.loads(path.read_text(encoding="utf-8")) == records
    else:
        path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"{path}: {len(records)} accuracy cases")


if __name__ == "__main__":
    main()
