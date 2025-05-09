import streamlit as st
import openai
import numpy as np
import faiss
import pickle
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 認証ライブラリの追加
import streamlit_authenticator as stauth

# ✅ ユーザー情報（仮）
names = ["User One"]
usernames = ["user1"]
passwords = ["pass123"]  # 本番ではハッシュ化を推奨

# ハッシュ生成（本番では外部ファイルで管理）
hashed_pw = stauth.Hasher(passwords).generate()

# ✅ Authenticatorインスタンス作成
authenticator = stauth.Authenticate(
    names,
    usernames,
    hashed_pw,
    "faq_app",  # cookie名
    "abcdef",   # シークレットキー（適当に変更可）
    cookie_expiry_days=1
)

# ✅ ログインチェック
name, auth_status, username = authenticator.login("ログイン", "main")

if auth_status is False:
    st.error("ユーザー名またはパスワードが違います。")
elif auth_status is None:
    st.warning("ログイン情報を入力してください。")
else:
    # ✅ 認証OKならここから表示
    authenticator.logout("ログアウト", "sidebar")
    st.sidebar.success(f"ログイン中: {name}")

    # ↓ ここから下は元のアプリ内容
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    index = faiss.read_index("faq.index")
    with open("faq_data.pkl", "rb") as f:
        faq_data = pickle.load(f)

    st.title("📚 社内FAQ検索（RAG対応）")

    with st.expander("📄 折りたたみ形式でFAQ一覧を見る"):
        for i, faq in enumerate(faq_data):
            st.markdown(f"**{i+1}. Q: {faq['question']}**")
            st.write(f"A: {faq['answer']}")
            st.markdown("---")

    with st.expander("📋 表形式でFAQ一覧（検索・ソート可）"):
        df = pd.DataFrame(faq_data)
        st.dataframe(df)

    with st.expander("📝 CSVの生テキストを見る"):
        with open("faq.csv", "r", encoding="utf-8-sig") as f:
            st.text(f.read())

    user_input = st.text_input("💬 質問を入力してください:")

    if user_input:
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_input
        )
        vec = np.array(res.data[0].embedding).astype("float32").reshape(1, -1)

        D, I = index.search(vec, k=3)

        st.subheader("🔍 類似FAQ（関連度順）:")
        for i, idx in enumerate(I[0]):
            faq = faq_data[idx]
            distance = D[0][i]
            st.markdown(f"**{i+1}. Q: {faq['question']}**")
            st.write(f"→ {faq['answer']}")
            st.caption(f"📌 類似スコア: `{distance:.4f}`")
            st.markdown("---")
