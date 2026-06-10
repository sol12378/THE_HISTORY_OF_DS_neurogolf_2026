# exp197_one_conv_kernel_miner

## 結果

- generic 1x1/3x3 Conv least-squares minerをgain_to_600上位taskへ試行。
- pixel-loop実装が重すぎ、timeoutしたため停止。
- 1x1限定・上位20への縮小でも遅く、実験としては不採用。

## 判断

3x3 Conv minerを続けるならvectorized実装に作り直す。今はtask-specific solved-rule loweringへ移る。
