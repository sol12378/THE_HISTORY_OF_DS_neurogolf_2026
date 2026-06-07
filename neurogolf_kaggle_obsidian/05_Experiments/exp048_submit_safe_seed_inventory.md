# exp048_submit_safe_seed_inventory

## 目的

exp041のLB崩壊を受け、submit-safe seedとteacher-only artifactを分離する。

## 結果

- strict seed: exp005, local 6282.230228, 400/400 all arc-gen pass
- teacher: exp023, local 6479.398825
- teacher gain vs strict: +197.168597
- P0 teacher-gain tasks: 18
- P1 high strict-cost tasks: 128

## 解釈

exp005を信頼seedにし、exp023/041はteacher/oracleとしてのみ扱う。teacher gainはそのまま提出できないが、rule compression対象として価値がある。

## 次

P0最大gainのtask020から、signature lookupを明示ruleへ圧縮する。
