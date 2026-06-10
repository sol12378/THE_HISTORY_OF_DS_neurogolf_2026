# exp234_task300_max_color_submit_probe

## 目的

exp178 current public bestに、exp233のtask300 max-color mask4x3 improved candidateを1 taskだけ差し替え、LB較正提出用zipを作る。

## 結果

- status: `zip_ready`
- validation: `267_pass_0_fail`
- task300 cost: `77546` -> `52653`
- expected_local_delta: `0.3871480916525627`
- expected_lb_if_public_pass: `6006.317148091653`
- zip_sanity: `{'count': 400, 'names_ok': True, 'first': ['task001.onnx', 'task002.onnx', 'task003.onnx'], 'last': ['task398.onnx', 'task399.onnx', 'task400.onnx'], 'bytes': 732764, 'sha256': '799b71d7f9ee7cbc375e5a7703f27544e937dd421a37b369069847b0f9b9a18f'}`

## 判断

zip_readyならKaggleへsingle-task deltaとして提出する。採点待ち中は次のPhase C候補探索を進める。
