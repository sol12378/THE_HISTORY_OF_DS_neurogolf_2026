# exp131_farm_os_cost_probe

## Hypothesis

レビュー指摘に従い、cost extractorを現行cost式寄りの数値proxyへ修正すれば、cheap controlとhigh-risk controlを実測前により妥当に分離できる。

## Result

- identity: band `250-600_plausible`, proxy `0`, official `0`, reject `False`
- channel_gather: band `250-600_plausible`, proxy `10`, official `0`, reject `False`
- conv1x1: band `250-600_plausible`, proxy `100`, official `0`, reject `False`
- full_grid_where: band `high_cost_probe_only`, proxy `18000`, official `0`, reject `False`
- small_where_expand: band `high_cost_probe_only`, proxy `36024`, official `36000`, reject `False`
- static_slice_pad_3x3: band `250-600_plausible`, proxy `381`, official `360`, reject `False`

## Interpretation

これはscore-producing実験ではなくfarm OSの校正実験。accepted/rejectedの判断はcost extractor単体ではなく、official score_network、full-arc validation、bundle ledgerを接続して行う必要がある。

## Leakage / Overfitting Risk

task001に対するcost probeであり、出力正解性は評価していない。LBやlocal estimateへ加算しない。
