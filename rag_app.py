import streamlit as st
import openai
import numpy as np
import faiss
import pickle
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ✅ 環境変数からOpenAI APIキーを取得
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ ベクトル検索インデックスとFAQデータを読み込み
index = faiss.read_index("faq.index")
with open("faq_data.pkl", "rb") as f:
    faq_data = pickle.load(f)

# ✅ Streamlit タイトル
st.title("社内FAQ検索")

# ✅ FAQ一覧表示（DataFrame形式）
with st.expander("FAQ一覧（検索・ソート可）"):
    df = pd.DataFrame(faq_data)
    st.dataframe(df)

# ✅ ユーザーからの質問入力
user_input = st.text_input("質問を入力してください:")

if user_input:
    # ユーザー質問をベクトルに変換
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_input
    )
    vec = np.array(res.data[0].embedding).astype("float32").reshape(1, -1)

    # 類似検索（上位3件）
    D, I = index.search(vec, k=3)

    # スコア1.3未満のみを抽出
    filtered_results = [(faq_data[idx], D[0][i]) for i, idx in enumerate(I[0]) if D[0][i] < 1.3]

    # すべてのスコアが1.0以上なら補完回答を出す
    should_supplement = all(score >= 1.0 for score in D[0])

    # ✅ 補完 + 限定FAQ表示
    if should_supplement:
        # 類似FAQ（スコア1.3未満のみ）を「参考」として表示
        if filtered_results:
            st.subheader("類似FAQ（参考）:")
            for i, (faq, distance) in enumerate(filtered_results):
                st.markdown(f"**{i+1}. Q: {faq['question']}**")
                st.write(f"→ {faq['answer']}")
                st.caption(f"類似スコア（小さいほど近い）: `{distance:.4f}`")
                st.markdown("---")

        # GPTによる補完回答
        st.subheader("補完回答")
        prompt = f"""
以下の質問に対して、一般的かつ信頼性の高い内容に基づいたビジネス向けの回答を提供してください。
文章は絵文字を使用せず、丁寧かつ簡潔に記述してください。

質問: {user_input}
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(response.choices[0].message.content)
        st.caption("※ ChatGPTは2023年4月時点までの知識に基づき回答しています。")

    else:
        # 通常表示（スコア1.3未満のみ表示）
        if filtered_results:
            st.subheader("類似FAQ（関連度順）:")
            for i, (faq, distance) in enumerate(filtered_results):
                st.markdown(f"**{i+1}. Q: {faq['question']}**")
                st.write(f"→ {faq['answer']}")
                st.caption(f"類似スコア（小さいほど近い）: `{distance:.4f}`")
                st.markdown("---")
        else:
            st.warning("類似するFAQが見つかりませんでした。")
