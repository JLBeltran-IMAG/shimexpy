"""This module implements functions to reduce SHI contrast outputs"""

def reduce_outputs(contrasts, reduce_maps, ref_reduced=None):
    out = {}

    # Absorption
    out["absorption"] = contrasts["absorption"]
    if ref_reduced and "absorption" in ref_reduced:
        out["absorption"] -= ref_reduced["absorption"]

    # Scattering
    scat = contrasts["scattering"]
    for key, idxs in reduce_maps.items():
        val = scat[idxs].sum(axis=0)
        if ref_reduced and f"scattering_{key}" in ref_reduced:
            val -= ref_reduced[f"scattering_{key}"]
        out[f"scattering_{key}"] = val

    # Phase wrapped (optional)
    phase = contrasts.get("phase_wrapped", None)

    if phase is not None:
        if "horizontal" in reduce_maps:
            p, n = reduce_maps["horizontal"]
            out["phase_horizontal"] = phase[p] - phase[n]

        if "vertical" in reduce_maps:
            p, n = reduce_maps["vertical"]
            out["phase_vertical"] = phase[p] - phase[n]

        if "bidirectional" in reduce_maps:
            hp, hn, vp, vn = reduce_maps["bidirectional"]
            out["phase_bidirectional"] = (
                phase[hp] - phase[hn] +
                phase[vp] - phase[vn]
            )

    return out
