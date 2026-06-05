# Competition Overview

The 2026 NeuroGolf Championship asks participants to solve image grid transformation tasks from examples. Initial work should prioritize reproducing the submission format and building validation discipline.

## 2026-06-05 Update

- ARC-AGI由来の400 taskに対して、taskごとに最小構造のONNX networkを提出するコードゴルフ型コンペ。
- 公式DiscussionのNetwork Synthesis Challenge投稿では、従来のML学習だけでなく、LLM agentic codingやneural architecture searchも選択肢として挙げられている。
- 初回はスコア最大化ではなく、公式仕様確認、データ取得、1 taskの手書きONNX検証を目的にする。
- exp001では `task087` の固定3x3 180度回転を `Gather` 2段で表現し、公開例すべてで完全一致。
