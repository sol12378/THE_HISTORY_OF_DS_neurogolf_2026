"""全テンプレートの正しさ・テンソル数・cost概算を検証する。"""
import numpy as np
import templates as T

# テスト用グリッド 4x5（非正方で軸の取り違えを検出）
g = np.array([
    [0, 1, 2, 3, 0],
    [0, 1, 0, 3, 4],
    [5, 0, 0, 3, 4],
    [5, 5, 0, 0, 4],
], dtype=np.uint8)

H, W = g.shape
U8 = 1  # uint8 = 1 byte


def cost_est(model):
    """memory概算 = グラフ内テンソルのバイト総和（uint8前提・概算）。
       入力/出力/中間を全て同程度サイズと仮定した上限見積もり。"""
    ntensor, nnode, ninit = T.count_tensors(model)
    return ntensor, nnode, ninit


def check(name, model, expected, x=None):
    if x is None:
        x = g.astype(np.uint8)
    got = T.run(model, x)
    ok = np.array_equal(got, expected)
    nt, nn, ni = cost_est(model)
    status = "OK " if ok else "NG!"
    print(f"[{status}] {name:18s} tensors={nt} nodes={nn} inits={ni} "
          f"out_shape={got.shape}")
    if not ok:
        print("   expected:\n", expected)
        print("   got:\n", got)
    return ok


results = []

# T1 恒等
results.append(check("identity", T.t_identity(), g))

# T2 左右反転
results.append(check("flip_lr", T.t_flip_lr(), g[:, ::-1]))

# T3 上下反転
results.append(check("flip_ud", T.t_flip_ud(), g[::-1, :]))

# T4 180度回転
results.append(check("rot180", T.t_rot180(), g[::-1, ::-1]))

# T5 90度反時計: np.rot90(g) は反時計回り
results.append(check("rot90_ccw", T.t_rot90_ccw(), np.rot90(g, 1)))

# T6 90度時計: np.rot90(g, -1)
results.append(check("rot90_cw", T.t_rot90_cw(), np.rot90(g, -1)))

# T7 転置
results.append(check("transpose", T.t_transpose(), g.T))

# T8 色置換: 0->0,1->6,2->4,3->8,4->4,5->5,... (長さ10のLUT)
# 色置換は入力int32（グリッド値がGatherのindex）
mp = [0, 6, 4, 8, 4, 5, 6, 7, 8, 9]
exp = np.array([[mp[v] for v in row] for row in g], dtype=np.uint8)
results.append(check("recolor", T.t_recolor(mp), exp, x=g.astype(np.int32)))

# T9 タイル 2x2
results.append(check("tile2x2", T.t_tile(2, 2), np.tile(g, (2, 2))))

# T10 切り出し [1:3, 1:4]
results.append(check("crop", T.t_crop(1, 3, 1, 4), g[1:3, 1:4]))

# T11 パディング 上下左右1, 値0
results.append(check("pad", T.t_pad(1, 1, 1, 1, 0),
                     np.pad(g, ((1, 1), (1, 1)), constant_values=0)))

# T12 自己連結 axis=1（横に2枚）
results.append(check("concat_self_w", T.t_concat_self(1),
                     np.concatenate([g, g], axis=1)))

# T13 鏡像連結 axis=1（入力 + 左右反転を横連結）
results.append(check("mirror_concat_w", T.t_mirror_concat(1),
                     np.concatenate([g, g[:, ::-1]], axis=1)))

# T14 ダウンスケール k=2
results.append(check("downscale2", T.t_downscale(2), g[::2, ::2]))

# T15 アップスケール k=2 (形状確定版)
up = np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)
results.append(check("upscale2", T.make_upscale_for_shape(H, W, 2), up))

print()
print(f"=== 結果: {sum(results)}/{len(results)} テンプレート正常 ===")
