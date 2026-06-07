# exp110_focused_surgery_rulehit_cropish_sweep

## Hypothesis

exp109でtask048 focused bypass surgeryがsafe micro-deltaを出した。P0 cropish taskとrule-hit taskへ同じconservative bypassを横展開すれば、広いanchor scanより安定したsubmit-safe deltaを拾える可能性がある。

## Result

- campaign #: `12`
- base local estimate: `6282.835302884462`
- targets: `35`
- generated candidates: `1299`
- improved candidates: `22`
- accepted tasks: `[184, 187, 207, 263, 316, 394]`
- local delta: `+0.086886962127`
- new local estimate: `6282.922189846589`
- submission decision: `submit_candidate_after_review`

## Accepted

- task184: `38855 -> 37955`, `+0.023435521192`
- task187: `105313 -> 104413`, `+0.008582679541`
- task207: `1875 -> 1795`, `+0.043603637482`
- task263: `8054 -> 8052`, `+0.000248354652`
- task316: `3903 -> 3893`, `+0.002565419570`
- task394: `4753 -> 4713`, `+0.008451349690`

## Interpretation

focused artifact surgeryはtask048だけの偶然ではなかった。#10後の方針転換により、#11〜#12でlocal `+0.109972` を獲得した。

ただしgainはまだmicro-delta級。LB 7500には桁が足りないため、このlaneはsubmit-safe calibration / post-passとして保持しつつ、#13以降はfused lowering/subgraph extractionで大きいcost reductionを狙う。

## Risk

- leakage risk: low。graph surgeryのみ。
- overfitting risk: low-to-medium。acceptedはfull validation済みだが、LB calibration候補として扱う。
