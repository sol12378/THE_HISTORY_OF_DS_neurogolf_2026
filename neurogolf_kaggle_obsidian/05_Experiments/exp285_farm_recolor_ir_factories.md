# exp285_farm_recolor_ir_factories

## 目的

recolor supplier のために direct/cast の標準 IR factory を追加する。

## 結果

- `recolor_direct_program()`: `cost_proxy=45`
- `recolor_cast_program()`: `cost_proxy=141`
- 両方とも hard reject なし。

## 判断

成功。今後の recolor candidate supplier は direct-first で生成し、cast は fallback として明示的に扱う。

## Submit

なし。tooling のみ。
