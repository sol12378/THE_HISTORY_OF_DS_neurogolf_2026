# exp111_focused_surgery_second_pass

## Hypothesis

exp109/exp110でacceptedしたfocused bypass editsの後に、2周目の局所surgeryを走らせると追加の冗長guard/castが露出する可能性がある。これによりfocused surgeryがcompoundableなpost-passか、ほぼone-shot cleanupかを判定する。

## Result

- campaign index: `13`
- base local estimate: `6282.922189846589`
- generated candidates: `1417`
- status counts: `rejected=1354`, `no_cost_gain=41`, `improved=22`
- accepted tasks: `[184, 187, 263, 316, 394]`
- local delta: `+0.013967011450821687`
- new local estimate: `6282.93615685804`
- zip sha256: `8043ea2d87831b5b9806401b46de7c6c89f57124a97e8afde5b76ac8ae30c096`

## Accepted Details

- task184: `second_bypass_And_010_in1`, cost `37955 -> 37925`, validation `169_pass_0_fail`
- task187: `second_bypass_Or_069_in1`, cost `104413 -> 103513`, validation `266_pass_0_fail`
- task263: `second_bypass_Mul_084_in0`, cost `8052 -> 8050`, validation `267_pass_0_fail`
- task316: `second_bypass_And_009_in0`, cost `3893 -> 3883`, validation `266_pass_0_fail`
- task394: `second_bypass_Add_025_in0`, cost `4713 -> 4705`, validation `266_pass_0_fail`

## Interpretation

focused surgeryは2周目でも改善を出すため完全なone-shotではない。ただし、exp110の `+0.086887` からexp111の `+0.013967` へ強く逓減している。したがって、これはsubmit-safe bundleの標準post-pass / LB較正laneとして有効だが、LB 7500を狙う主経路ではない。

## Next Action

#14以降は、3周目surgeryを限定的に確認する場合でも短く止める。主力はtask185 lattice homogeneous-2x2、task366 object-marker copy、task365 rectangle cropなど、既にPython ruleが解けているtaskのfused loweringまたは大きなsubgraph extractionへ戻す。

## Risk

- leakage risk: low。graph surgeryのみで、全arc-gen validation gateを通している。
- overfitting risk: low-to-medium。LB較正はまだ未提出なので、bundle化後に小delta提出で確認する。
