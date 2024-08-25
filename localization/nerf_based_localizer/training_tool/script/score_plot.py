# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# csv_dir = '/home/ryomasakata/result_nerf_full_4096/test_result'

# grid_size = 3

# # CSVファイルを読み込んでデータフレームに変換
# def load_csv_files(csv_dir):
#     csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
#     data_frames = []
#     for f in csv_files:
#         df = pd.read_csv(f, skiprows=1, names=['pose[0][0]', 'pose[0][1]', 'pose[0][2]', 'pose_x', 
#                                                'pose[1][0]', 'pose[1][1]', 'pose[1][2]', 'pose_y', 
#                                                'pose[2][0]', 'pose[2][1]', 'pose[2][2]', 'pose_z', 
#                                                'score'])
#         data_frames.append(df)
#     return data_frames

# # データフレームからスコアを取得し、グリッドに変換
# def create_score_grid(data_frames):
#     score_grids = []
#     for df in data_frames:
#         grid = np.zeros((grid_size, grid_size))
#         for index, row in df.iterrows():
#             x = int((row['pose_x'] + 0.5) * (grid_size - 1))
#             y = int((row['pose_y'] + 0.5) * (grid_size - 1))
#             grid[x, y] = row['score']
#         score_grids.append(grid)
#     return score_grids

# # カラーマップを使ってスコアグリッドをプロット
# def plot_score_grids(score_grids):
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#     axes = axes.ravel()
#     for i, grid in enumerate(score_grids):
#         ax = axes[i]
#         cax = ax.imshow(grid, cmap='viridis')
#         fig.colorbar(cax, ax=ax)
#         ax.set_title(f'Grid {i+1}')
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     data_frames = load_csv_files(csv_dir)
#     score_grids = create_score_grid(data_frames)
#     plot_score_grids(score_grids)

# import os
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt

# # CSVファイルが保存されているディレクトリ
# csv_dir = '/home/ryomasakata/result_nerf_full_4096/test_result'

# # CSVファイルのリストを取得
# csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

# # データを格納するリスト
# data = []

# # CSVファイルを順番に読み込む
# for file in csv_files:
#     df = pd.read_csv(file)
#     data.append(df)

# # プロットのための準備
# plt.figure(figsize=(15, 15))

# # 3x3のグリッドにプロットする
# for i, df in enumerate(data):
#     plt.subplot(3, 3, i+1)
#     # scatterプロット
#     sc = plt.scatter(df['pose_x'], df['pose_y'], c=df['score'], cmap='viridis', s=100, edgecolors='w', linewidth=0.5)
#     plt.title(f'File {i+1}')
#     plt.xlabel('pose_x')
#     plt.ylabel('pose_y')
#     plt.colorbar(sc)

# # グリッドの調整
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # CSVファイルの読み込み
# file_path = '/home/ryomasakata/result_nerf_full_4096/test_result/0.csv'
# df = pd.read_csv(file_path)

# print(df.columns.tolist())
# df.columns = df.columns.str.strip()
# # プロットの設定
# fig, ax = plt.subplots()

# # 各行の処理
# for index, row in df.iterrows():
#     pose_x = row['pose_x']
#     pose_y = row['pose_y']
#     score = row['score']
    
#     # グリッドの中心を表示
#     ax.text(pose_x, pose_y, f'{score:.2f}', ha='center', va='center', color='red', fontsize=12)
    
#     # グリッドの描画
#     for i in range(-1, 2):
#         for j in range(-1, 2):
#             ax.add_patch(plt.Rectangle((pose_x - 0.5 + i, pose_y - 0.5 + j), 1, 1, edgecolor='black', facecolor='none'))

# # 軸の設定
# ax.set_aspect('equal')
# plt.xlabel('pose_x')
# plt.ylabel('pose_y')
# plt.title('3x3 Grid with Scores at Center')
# plt.grid(True)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # CSVファイルの読み込み
# file_path = '/home/ryomasakata/result_nerf_full_4096/test_result/0.csv'  # ファイルパスを適宜変更してください
# df = pd.read_csv(file_path)

# # 列名から余分なスペースを削除
# df.columns = df.columns.str.strip()

# # グリッドのサイズ
# grid_size = 3
# grid_length = 0.1  # 各グリッドの大きさを1に統一

# # プロットの設定
# fig, ax = plt.subplots()

# # 各行の処理
# for index, row in df.iterrows():
#     pose_x = row['pose_x']
#     pose_y = row['pose_y']
#     score = row['score']
    
#     # グリッドの描画
#     for i in range(-1, 2):
#         for j in range(-1, 2):
#             grid_center_x = pose_x + i * grid_length
#             grid_center_y = pose_y + j * grid_length
#             ax.add_patch(plt.Rectangle((grid_center_x - 0.5 * grid_length, grid_center_y - 0.5 * grid_length), 
#                                        grid_length, grid_length, edgecolor='black', facecolor='none'))
    
#     # ベクトル場の描画
#     X, Y = np.meshgrid(np.arange(-1, 2) * grid_length + pose_x, np.arange(-1, 2) * grid_length + pose_y)
#     U = np.zeros_like(X) + score  # X方向のベクトル
#     V = np.zeros_like(Y) + score  # Y方向のベクトル
#     ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red')

# # 描画範囲の設定
# ax.set_xlim(pose_x - 1.5 * grid_length, pose_x + 1.5 * grid_length)
# ax.set_ylim(pose_y - 1.5 * grid_length, pose_y + 1.5 * grid_length)

# # 軸の設定
# ax.set_aspect('equal')
# plt.xlabel('pose_x')
# plt.ylabel('pose_y')
# plt.title('3x3 Grid with Vectors at Center')
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Create a 3x3 grid
# fig, ax = plt.subplots()

# # Draw grid lines
# ax.set_xticks(np.arange(0, 4, 1))
# ax.set_yticks(np.arange(0, 4, 1))
# ax.grid(True)

# # Label each cell
# for i in range(3):
#     for j in range(3):
#         ax.text(i + 0.5, 2.5 - j, f'{3*j+i+1}', ha='center', va='center', fontsize=16)

# # Set limits and hide ticks
# ax.set_xlim(0, 3)
# ax.set_ylim(0, 3)
# ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# # Title
# plt.title("strike zone")

# # Show plot
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import matplotlib.colors as mcolors

# # CSVファイルを読み込む
# csv_file = '/home/ryomasakata/result_nerf_full_4096/test_result/0.csv'
# data = pd.read_csv(csv_file)
# data.columns = data.columns.str.strip()

# # pose_x, pose_y, widthを取得
# pose_x = data['pose_x'].values
# pose_y = data['pose_y'].values
# scores = data['score'].values
# width = 0.01

# # スコアの最小値と最大値を取得
# min_score = min(scores)
# max_score = max(scores)

# # カラーマップを作成
# norm = mcolors.Normalize(vmin=min_score, vmax=max_score)
# cmap = plt.get_cmap('coolwarm')


# # グリッドの描画
# fig, ax = plt.subplots()

# # グリッドの数
# num_grids = len(pose_x)

# # グリッドを描画
# for i in range(num_grids):
#     # カラーを取得し、透明度を設定
#     color = cmap(norm(scores[i]))
#     color_with_alpha = (color[0], color[1], color[2], 0.6)  # 透明度を0.6に設定
#     rect = plt.Rectangle((pose_x[i] - width/2, pose_y[i] - width/2), width, width, 
#                          facecolor=color_with_alpha, edgecolor='black')
#     ax.add_patch(rect)
#     ax.text(pose_x[i], pose_y[i], f'{scores[i]}', ha='center', va='center', fontsize=16)

# # グリッドの範囲を設定
# ax.set_xlim(min(pose_x) - width, max(pose_x) + width)
# ax.set_ylim(min(pose_y) - width, max(pose_y) + width)
# ax.set_aspect('equal', adjustable='box')

# # カラーバーの追加
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Score')

# # タイトル
# plt.title("Grid Centers and Scores with Color Gradient")

# # プロットを表示
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import os
import math

directory = '/home/ryomasakata/result_nerf_full_4096/test_result(x-y_only)'
file_template = '{}.csv'
output_directory = '/home/ryomasakata/result_nerf_full_4096/plots'

os.makedirs(output_directory, exist_ok=True)

file_index = 0
while True:
    csv_file = os.path.join(directory, file_template.format(file_index))
    
    if not os.path.isfile(csv_file):
        break
    
    if os.path.getsize(csv_file) == 0:
        print(f"Skipping empty file: {csv_file}")
        file_index += 100
        continue
    
    data = pd.read_csv(csv_file)
    data.columns = data.columns.str.strip()

    pose_x = data['pose_x'].values
    pose_y = data['pose_y'].values
    scores = data['score'].values
    
    # グリッド数の計算
    num_grids = len(pose_x)
    grid_size = int(math.sqrt(num_grids))  # グリッドの一辺の長さ（正方形を仮定）

    if grid_size * grid_size != num_grids:
        print(f"Unexpected number of entries in {csv_file}, expected a perfect square.")
        file_index += 100
        continue

    width = (max(pose_x) - min(pose_x)) / (grid_size - 1)  # グリッドの幅を計算

    min_score = min(scores)
    max_score = max(scores)

    norm = mcolors.Normalize(vmin=50, vmax=250)
    cmap = plt.get_cmap('coolwarm')

    fig, ax = plt.subplots(figsize=(10, 10)) 

    for i in range(num_grids):
        color = cmap(norm(scores[i]))
        color_with_alpha = (color[0], color[1], color[2], 0.6)  
        rect = plt.Rectangle((pose_x[i] - width/2, pose_y[i] - width/2), width, width, 
                             facecolor=color_with_alpha, edgecolor='black')
        ax.add_patch(rect)
        ax.text(pose_x[i], pose_y[i], f'{scores[i]:.1f}', ha='center', va='center', fontsize=10)  # 文字サイズを小さくする

    ax.set_xlim(min(pose_x) - width, max(pose_x) + width)
    ax.set_ylim(min(pose_y) - width, max(pose_y) + width)
    ax.set_aspect('equal', adjustable='box')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Score')

    plt.title(f"Grid Centers and Scores with Color Gradient - {file_index}")

    output_file = os.path.join(output_directory, f'plot_{file_index}.svg')
    plt.savefig(output_file, format='svg')

    plt.close(fig)

    file_index += 100
