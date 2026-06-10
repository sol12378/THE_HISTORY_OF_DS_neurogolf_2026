# exp161_where_dtype_rewrite_inspection

## 目的

Whereが多いdtype候補task `206/338/328/366` について、boolish-to-FLOAT CastがWhere条件としてだけ使われる安全rewrite候補かを監査する。

## 結果

- task206: WhereはすでにBOOL条件 + FLOAT16 branchが中心。promisingに見えたCastは最終 `output` への `BOOL -> FLOAT`。
- task338: WhereはUINT8 branchが中心で、FLOAT化は数値consumerがある。
- task328: promisingに見えたCastは最終 `output` への `BOOL -> FLOAT`。
- task366: WhereはBOOL条件 + FLOAT16 branchが中心で、既にdtype縮小が多く入っている。

## 解釈

既存高cost artifactの多くは、すでにBOOL/FLOAT16/UINT8を使っている。削れそうに見えたCastは公式出力型をFLOATへ戻す最終Castであり、削除できない。既存graphの単純dtype post-passは、少なくとも今回のWhere候補では即効性が薄い。

## 判断

Phase C-1は「既存artifactの後処理」より、新規lowering/DSL生成時にBOOL/UINT8/FLOAT16中間を意識して設計する方向へ寄せる。既存graph surgeryでは、dtypeよりも具体的な冗長Cast/Where/branch pruningが見つかった場合だけ試す。

## リスク

- leakage risk: low。graph構造検査のみ。
- overfitting risk: low-to-medium。rewrite候補はまだ作っていない。
