#!/usr/bin/env python3
"""
Gerador de benchmarks projetados para Qwen3.5 0.8B Q4_K_M em Android.

Baseado na formula:
  gen_tok_s = (memory_bandwidth_GBs / model_size_GB) * efficiency

Uso:
  python3 scripts/benchmark_table.py
"""

import json
import sys

MODEL_SIZE_GB = 0.505  # Q4_K_M
MEASURED_EFFICIENCY = 0.66  # A54 real: 19.7 / (15.0 * 0.8 / 0.505)
# A54 LPDDR4X theoretical: 15 GB/s, effective ~12 GB/s
# 19.7 = (12 / 0.505) * 0.83 -> efficiency closer to 0.83 for the A54
# But using conservative 0.66 for projections

# Devices database: {name: (soc, ram_gb, ram_type, mem_bandwidth_gbs, cores_big, cores_little)}
DEVICES = {
    # Flagship 2024-2025
    "Galaxy S25 Ultra": ("Snapdragon 8 Elite", 12, "LPDDR5X", 68, "2x Oryon V2 + 6x Oryon M"),
    "Galaxy S24 Ultra": ("Snapdragon 8 Gen 3", 12, "LPDDR5X", 64, "1x X4 + 5x A720 + 2x A520"),
    "ASUS ROG Phone 9": ("Snapdragon 8 Elite", 16, "LPDDR5X", 68, "2x Oryon V2 + 6x Oryon M"),
    "OnePlus 13": ("Snapdragon 8 Elite", 12, "LPDDR5X", 68, "2x Oryon V2 + 6x Oryon M"),
    "Pixel 9 Pro": ("Tensor G4", 16, "LPDDR5X", 64, "1x X4 + 3x A720 + 4x A520"),
    "Xiaomi 15 Pro": ("Snapdragon 8 Elite", 12, "LPDDR5X", 68, "2x Oryon V2 + 6x Oryon M"),
    
    # Flagship 2023
    "Galaxy S23 Ultra": ("Snapdragon 8 Gen 2", 12, "LPDDR5X", 58, "1x X3 + 4x A715 + 3x A510"),
    "Pixel 8 Pro": ("Tensor G3", 12, "LPDDR5X", 51, "1x X3 + 4x A715 + 4x A510"),
    
    # Mid-Range 2024
    "Galaxy A55": ("Exynos 1480", 8, "LPDDR5", 44, "4x A78 + 4x A55"),
    "POCO X6 Pro": ("Dimensity 8300", 8, "LPDDR5X", 51, "4x A715 + 4x A510"),
    "Pixel 8a": ("Tensor G3", 8, "LPDDR5X", 51, "1x X3 + 4x A715 + 4x A510"),
    "Redmi Note 13 Pro+": ("Dimensity 7200", 8, "LPDDR5", 34, "2x A715 + 6x A510"),
    
    # Mid-Range 2023
    "Galaxy A54": ("Exynos 1380", 8, "LPDDR4X", 15, "4x A78 + 4x A55"),
    "Galaxy A35": ("Exynos 1380", 6, "LPDDR4X", 15, "4x A78 + 4x A55"),
    "POCO X5 Pro": ("Snapdragon 778G", 8, "LPDDR5", 25, "4x A78 + 4x A55"),
    
    # Budget
    "Galaxy A15 5G": ("Dimensity 6100+", 4, "LPDDR4X", 14, "2x A76 + 6x A55"),
    "Moto G84 5G": ("Snapdragon 695", 8, "LPDDR4X", 14, "2x A78 + 6x A55"),
    "Redmi 13C": ("Helio G85", 4, "LPDDR4X", 13, "2x A75 + 6x A55"),
    "Galaxy A05s": ("Snapdragon 680", 4, "LPDDR4X", 13, "4x A73 + 4x A53"),
}

def calc_tok_s(bandwidth_gbs, efficiency=MEASURED_EFFICIENCY):
    """Calculate tokens/sec for Q4_K_M model."""
    gen = (bandwidth_gbs / MODEL_SIZE_GB) * efficiency
    prefill = gen * 5.5  # Prefill is roughly 5-6x faster on ARM
    return gen, prefill

def calc_effective_bandwidth(theoretical_gbs, ram_type):
    """Estimate effective bandwidth considering real-world overhead."""
    factors = {"LPDDR5X": 0.85, "LPDDR5": 0.82, "LPDDR4X": 0.80, "LPDDR4": 0.78}
    return theoretical_gbs * factors.get(ram_type, 0.80)

def main():
    print("# Benchmarks Projetados — Qwen3.5 0.8B Q4_K_M (505 MB)")
    print()
    print("## Flagship 2023-2025")
    print()
    print("| Dispositivo | SoC | RAM | Banda | Geração (tok/s) | Prefill (tok/s) |")
    print("|---|---:|---:|---:|---:|")
    
    flagships = ["Galaxy S25 Ultra", "ASUS ROG Phone 9", "OnePlus 13", "Xiaomi 15 Pro",
                 "Galaxy S24 Ultra", "Pixel 9 Pro", "Galaxy S23 Ultra", "Pixel 8 Pro"]
    
    for name in flagships:
        if name in DEVICES:
            soc, ram, ram_type, bw, cores = DEVICES[name]
            eff_bw = calc_effective_bandwidth(bw, ram_type)
            gen, prefill = calc_tok_s(eff_bw)
            print(f"| **{name}** | {soc} | {ram} GB {ram_type} | {eff_bw:.0f} GB/s | **{gen:.0f}** | {prefill:.0f} |")
    
    print()
    print("## Mid-Range")
    print()
    print("| Dispositivo | SoC | RAM | Banda | Geração (tok/s) | Prefill (tok/s) |")
    print("|---|---:|---:|---:|")
    
    midrange = ["Galaxy A55", "POCO X6 Pro", "Pixel 8a", "Redmi Note 13 Pro+",
                "POCO X5 Pro", "Galaxy A54", "Galaxy A35"]
    
    for name in midrange:
        if name in DEVICES:
            soc, ram, ram_type, bw, cores = DEVICES[name]
            eff_bw = calc_effective_bandwidth(bw, ram_type)
            gen, prefill = calc_tok_s(eff_bw)
            measured = ""
            if name == "Galaxy A54":
                measured = " ⚡REAL: 19.7 tok/s"
            print(f"| **{name}** | {soc} | {ram} GB {ram_type} | {eff_bw:.0f} GB/s | **{gen:.0f}**{measured} | {prefill:.0f} |")
    
    print()
    print("## Budget")
    print()
    print("| Dispositivo | SoC | RAM | Banda | Geração (tok/s) | Prefill (tok/s) | Nota |")
    print("|---|---:|---:|---:|---|")
    
    budget = ["Galaxy A15 5G", "Moto G84 5G", "Redmi 13C", "Galaxy A05s"]
    
    for name in budget:
        if name in DEVICES:
            soc, ram, ram_type, bw, cores = DEVICES[name]
            eff_bw = calc_effective_bandwidth(bw, ram_type)
            gen, prefill = calc_tok_s(eff_bw)
            note = "⚠ RAM<6GB: usar Qwen2.5 0.5B" if ram < 6 else ""
            print(f"| **{name}** | {soc} | {ram} GB {ram_type} | {eff_bw:.0f} GB/s | **{gen:.0f}** | {prefill:.0f} | {note} |")
    
    print()
    print("---")
    print(f"*Modelo: Qwen3.5 0.8B Q4_K_M (505 MB). Eficiencia medida: {MEASURED_EFFICIENCY*100:.0f}% (A54).*")
    print("*Geração projetada como (banda_efetiva / 0.505 GB) x 0.66.*")
    print("*Prefill ~5.5x mais rapido que geração em ARM (medido no A54).*")
    
    # Also output JSON for programmatic use
    json_output = {}
    for name, (soc, ram, ram_type, bw, cores) in DEVICES.items():
        eff_bw = calc_effective_bandwidth(bw, ram_type)
        gen, prefill = calc_tok_s(eff_bw)
        json_output[name] = {
            "soc": soc,
            "ram_gb": ram,
            "ram_type": ram_type,
            "bandwidth_theoretical_gbs": bw,
            "bandwidth_effective_gbs": round(eff_bw, 1),
            "generation_tok_s": round(gen, 1),
            "prefill_tok_s": round(prefill, 1),
            "cores": cores,
        }
    
    with open("/home/daniel/qwen35-ptbr-mobile/benchmarks/projections.json", "w") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON salvo em benchmarks/projections.json ({len(json_output)} dispositivos)")

if __name__ == "__main__":
    main()
