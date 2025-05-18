import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from pyvis.network import Network
import tempfile

warnings.filterwarnings("ignore")

# フォント設定
JAPANESE_FONT = "Hiragino Sans"


# CSVファイルの読み込み
def load_data():
    # 現在のスクリプトの場所を取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "member.csv")

    # CSVファイルを読み込む
    df = pd.read_csv(csv_path)

    # interestsをリストに変換する
    df["interests"] = df["interests"].apply(lambda x: x.split(","))

    return df


# 趣味の類似度を計算
def calculate_interests_similarity(df):
    # 各会員の趣味を文字列に変換
    interests_str = df["interests"].apply(lambda x: " ".join(x))

    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    interests_matrix = vectorizer.fit_transform(interests_str)

    # コサイン類似度を計算
    interests_similarity = cosine_similarity(interests_matrix)

    return interests_similarity


# 自己紹介文の類似度を計算
def calculate_intro_similarity(df):
    # 自己紹介文をベクトル化
    vectorizer = TfidfVectorizer()
    intro_matrix = vectorizer.fit_transform(df["introduction"])

    # コサイン類似度を計算
    intro_similarity = cosine_similarity(intro_matrix)

    return intro_similarity


# 総合的な類似度を計算
def calculate_overall_similarity(interests_sim, intro_sim, interests_weight=0.7):
    # 趣味の類似度と自己紹介の類似度を重み付けして統合
    overall_sim = interests_sim * interests_weight + intro_sim * (1 - interests_weight)
    return overall_sim


# ネットワークグラフを作成
def create_network_graph(df, similarity_matrix, threshold=0.3):
    G = nx.Graph()

    # ノードを追加
    for i, name in enumerate(df["name"]):
        G.add_node(i, name=name, interests=df["interests"][i])

    # エッジを追加（類似度がしきい値を超える場合）
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    return G


# PyVisで可視化（インタラクティブなネットワークグラフ）
def create_pyvis_network(G, df, similarity_matrix):
    # PyVisネットワークを作成
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")

    # ノード同士の反発力を設定
    net.repulsion(node_distance=200, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)

    # 言語設定
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 16
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "springConstant": 0.001
        },
        "minVelocity": 0.75
      }
    }
    """)

    # NetworkXグラフからノードを追加
    for i, node in enumerate(G.nodes()):
        interests_str = ", ".join(df.iloc[i]["interests"])
        # ノードのタイトルに詳細情報を含める
        title = f"<b>{df.iloc[i]['name']}</b><br>興味: {interests_str}<br><br><b>類似度</b>:<br>"

        # 他のメンバーとの類似度を追加
        for j, other_node in enumerate(G.nodes()):
            if i != j:
                title += f"{df.iloc[j]['name']}: {similarity_matrix[i, j]:.2f}<br>"

        net.add_node(i, label=df.iloc[i]["name"], title=title, color="skyblue", size=30)

    # エッジを追加
    for edge in G.edges():
        weight = G[edge[0]][edge[1]]["weight"]
        width = weight * 10  # 線の太さを類似度に基づいて設定
        title = f"類似度: {weight:.2f}"
        net.add_edge(edge[0], edge[1], title=title, width=width, color={"opacity": 0.7})

    # 一時ファイルに保存して読み込む
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        temp_path = tmpfile.name
        net.save_graph(temp_path)
        with open(temp_path, "r", encoding="utf-8") as f:
            html_string = f.read()
        os.unlink(temp_path)  # 一時ファイルを削除

    return html_string


# グラフの描画
def draw_graph(G, df):
    # ノードの位置を春モデルのレイアウトで計算
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(12, 8))

    # エッジの太さを類似度に基づいて設定
    edge_weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]

    # ノードのサイズ
    node_size = 1000

    # グラフを描画
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color="lightgray")
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="skyblue", alpha=0.8)

    # ノードラベルを描画
    labels = {i: df["name"][i] for i in range(len(df))}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family=JAPANESE_FONT)

    plt.axis("off")
    return fig


# ヒートマップを作成
def create_heatmap(similarity_matrix, member_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap="YlOrRd")

    # 軸の目盛りと軸ラベルを設定
    ax.set_xticks(np.arange(len(member_names)))
    ax.set_yticks(np.arange(len(member_names)))
    ax.set_xticklabels(member_names)
    ax.set_yticklabels(member_names)

    # 軸ラベルを回転
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 類似度の値を表示
    for i in range(len(member_names)):
        for j in range(len(member_names)):
            ax.text(
                j,
                i,
                f"{similarity_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if similarity_matrix[i, j] < 0.7 else "white",
            )

    plt.colorbar(im)
    plt.tight_layout()
    return fig
