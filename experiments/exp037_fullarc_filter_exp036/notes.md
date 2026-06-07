# exp037_fullarc_filter_exp036

## Hypothesis

exp036の広いno-op bypass探索はsample20では有効だが、一部はarc-gen後半に過学習している可能性がある。全arc-genを通過した改善だけに絞れば、より堅いlocal estimateを更新できる。

## Result

- base: `experiments\exp035_greedy_logic_surgery_composition`
- source: `experiments\exp036_noop_bypass_and_prune`
- source selected tasks: `16`
- kept after full arc-gen: `12`
- rejected after full arc-gen: `4`
- local estimate: `6480.095137`
- delta vs base: `0.432396`
- gap to 6500: `19.904863`
- submission decision: `no_submit: below 6500 threshold`

## Risks

- leakage risk: medium: selected edits pass all available arc-gen validation, but base still contains high-risk lookup artifacts.
- overfitting risk: medium: full arc-gen filter removes observed sample20-only failures, but private distribution risk remains.
