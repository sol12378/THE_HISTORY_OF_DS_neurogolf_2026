# exp_b026_five_experiment_review_b020_b024

## 目的

5実験ごとのPDCA点検。`exp_b020`〜`exp_b024` が、`cost<=250` / 7700 / local-LB統一に対して本当に有意義だったか確認する。

## Verdict

`useful_but_postpass_limited`

有意義。ただし主な成果はsubmit-safe micro deltaとpublic notebook由来post-passの較正であり、7700に必要な大量taskのcost<=250化そのものはまだ進んでいない。

## 実験別評価

- `exp_b020`: L4上位が単純固定crop/bbox cropではないことを確認。負の結果だが、naive Slice路線を棄却できた。
- `exp_b021`: exp038由来4 task deltaをstrict seedへ載せて提出。LB `5930.02` でstrict seedから `+0.13`。
- `exp_b022`: ユーザー指定notebookをsource/lowering/taxonomy intelligenceへ分類。
- `exp_b023`: Seddik notebookからinitializer surgery patternを抽出。
- `exp_b024`: strict seed上の狭いSeddik-style scalarizationを提出。LB `5929.92`、local deltaとほぼ一致。

## 次の方針

- submit-safe micro deltaは引き続き早めに提出してlocal/LBを揃える。
- Seddik-style post-passは標準後処理化する。
- ただし7700本筋はpost-passではない。次はobject-anchor crop、L3 object move/erase、task020型sparse completionの横展開へ戻す。
