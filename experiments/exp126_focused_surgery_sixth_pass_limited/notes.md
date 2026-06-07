# exp126_focused_surgery_sixth_pass_limited

## Hypothesis

exp122後のaccepted-task bundleにも、冗長guard/cast/arithmeticがまだ残っており、focused surgery四巡目でsubmit-safe micro-deltaを追加できる可能性がある。

## Result

- targets: `7`
- generated candidates: `525`
- status counts: `{'rejected': 504, 'improved': 21}`
- accepted tasks: `[263, 316]`
- local delta: `0.002847`
- new local estimate: `6282.962381`

## Decision

deltaが出るなら、focused surgeryは最終submission bundleのpost-passとして継続する。ただしLB 1位に必要な桁ではないため、主戦略は低cost primitive compilerのまま。

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。
