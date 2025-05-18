import streamlit as st
import streamlit.components.v1 as components
from member_map.main import (
    load_data,
    calculate_interests_similarity,
    calculate_intro_similarity,
    calculate_overall_similarity,
    create_network_graph,
    create_pyvis_network,
    create_heatmap,
)


# Streamlitアプリ
def main():
    st.set_page_config(layout="wide", page_title="会員関係マップ")
    st.title("NIHO会員関係マップ")

    # データ読み込み
    df = load_data()

    # 類似度計算
    interests_sim = calculate_interests_similarity(df)
    intro_sim = calculate_intro_similarity(df)

    # パラメータ設定セクションをページ本体の先頭に配置（より控えめに）
    with st.expander("パラメータ設定", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            interests_weight = st.slider("趣味の重み", 0.0, 1.0, 0.7, 0.1)
        with col2:
            threshold = st.slider("類似度閾値", 0.0, 1.0, 0.3, 0.05)

    # 総合類似度の計算
    overall_sim = calculate_overall_similarity(interests_sim, intro_sim, interests_weight)

    # ネットワークグラフの作成
    G = create_network_graph(df, overall_sim, threshold)

    # タブを作成
    tab1, tab2, tab3 = st.tabs(["ネットワークグラフ", "ヒートマップ", "会員データ"])

    with tab1:
        st.header("会員間の関係ネットワーク")
        st.write(
            "会員間の類似度に基づくネットワーク図です。線の太さは類似度の強さを表します。**ノードをドラッグして自由に配置できます**。"
        )

        # PyVisによるインタラクティブグラフ
        html_string = create_pyvis_network(G, df, overall_sim)
        components.html(html_string, height=600)

        # 会員間の類似度の詳細
        st.subheader("選択された会員の類似メンバー")
        selected_member = st.selectbox("会員を選択", df["name"])
        member_idx = df[df["name"] == selected_member].index[0]

        st.write(f"**{selected_member}** の趣味: {', '.join(df.loc[member_idx, 'interests'])}")

        # 選択された会員と他の会員との類似度
        similarities = []
        for i, name in enumerate(df["name"]):
            if i != member_idx:
                similarities.append((name, overall_sim[member_idx, i], ", ".join(df.loc[i, "interests"])))

        similarities.sort(key=lambda x: x[1], reverse=True)

        for name, sim, interests in similarities:
            st.write(f"- **{name}** (類似度: {sim:.2f}) - 趣味: {interests}")

    with tab2:
        st.header("会員間の類似度ヒートマップ")
        st.write("会員間の類似度をヒートマップで表示します。")

        heatmap_fig = create_heatmap(overall_sim, df["name"])
        st.pyplot(heatmap_fig)

    with tab3:
        st.header("会員データ一覧")
        # 興味タグでフィルタリングするオプション
        all_interests = set()
        for interests_list in df["interests"]:
            all_interests.update(interests_list)

        selected_interest = st.selectbox("興味で絞り込む", ["すべて表示"] + sorted(all_interests))

        if selected_interest == "すべて表示":
            filtered_df = df
        else:
            filtered_df = df[df["interests"].apply(lambda x: selected_interest in x)]

        # 会員データを表示
        for _, row in filtered_df.iterrows():
            st.subheader(row["name"])
            st.write(f"**興味**: {', '.join(row['interests'])}")
            st.write(f"**自己紹介**: {row['introduction']}")
            st.markdown("---")


if __name__ == "__main__":
    main()
