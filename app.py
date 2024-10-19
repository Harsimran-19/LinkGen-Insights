import streamlit as st
from crawlers.linkedin import fetch_posts, make_post_data
from bytewax.testing import run_main
from src.flow import build as build_flow
import time
import os
from src.embedding import EmbeddingModelSingleton, CrossEncoderModelSingleton
from src.qdrant import build_qdrant_client
from src.retriever import QdrantVectorDBRetriever, LangChainQdrantRetriever
from src.chain import create_rag_chain
from models.settings import settings

st.set_page_config(page_title="LinkGen-Insights")
st.title("🎯 LinkGen-Insights")

def basic_prerequisites():
    linkedin_email = st.sidebar.text_input("LinkedIn Email Address")
    linkedin_password = st.sidebar.text_input("LinkedIn Password", type="password")
    linkedin_username_account = st.sidebar.text_input(
        "Type in the username of the profile whose posts you'd like to get."
    )
    need_data = st.sidebar.button("🧲 Fetch Details")
    if need_data:
        warn = st.sidebar.warning(
            "Please keep in mind that this feature fetches data directly from LinkedIn. It might open LinkedIn in your web browser. Please avoid closing the browser while using this feature."
        )
        time.sleep(2)
        if not linkedin_email:
            st.sidebar.warning("Please enter your linkedin email address for login!", icon="⚠")
        elif not linkedin_password:
            st.sidebar.warning("Please enter your linkedin password for login!", icon="⚠")
        elif not linkedin_username_account:
            st.sidebar.warning(
                "Please enter the linkedin username from which you want to fetch the posts!",
                icon="⚠",
            )
        else:
            account_posts_url = f"https://www.linkedin.com/in/{linkedin_username_account}/recent-activity/all/"
            all_posts = fetch_posts(
                linkedin_email, linkedin_password, account_posts_url
            )
            make_post_data(all_posts, linkedin_username_account)
            warn.empty()
            warn = st.sidebar.success("Success! All posts retrieved.")
            return linkedin_username_account

def migrate_data_to_vectordb(username):
    qdrant_client = build_qdrant_client()
    embedding_model = EmbeddingModelSingleton()
    try:
        qdrant_client.delete_collection(settings.VECTOR_DB_OUTPUT_COLLECTION_NAME)
    except Exception as e:
        # Ignore any exception if the collection doesn't exist
        print(f"Collection '{settings.VECTOR_DB_OUTPUT_COLLECTION_NAME}' not found, skipping deletion. Error: {e}")
    warn = st.sidebar.warning(
        "Hold on tight! We're moving data to a new system (VectorDB) to improve performance. We'll be back soon. ",
        icon="🚀",
    )
    if username:
        data_source_path = [f"data/{username}_data.json"]
    else:
        data_source_path = [f"data/{p}" for p in os.listdir("data")]
    flow = build_flow(in_memory=False, data_source_path=data_source_path)
    run_main(flow)
    warn.empty()
    warn = st.sidebar.success("We're all set! Data transfer to VectorDB is finished.", icon="🎊")

def get_insights_from_posts():
    with st.form("my_form"):
        query = st.text_area(
            "✨ Make a Query:",
            f"",
        )
        submitted = st.form_submit_button("Submit Query")
        if submitted:
            embedding_model = EmbeddingModelSingleton()
            cross_encode_model = CrossEncoderModelSingleton()
            qdrant_client = build_qdrant_client()
            vectordb_retriever = QdrantVectorDBRetriever(
                embedding_model=embedding_model,
                cross_encoder_model=cross_encode_model,
                vector_db_client=qdrant_client,
            )
            retriever = LangChainQdrantRetriever()
            retriever.set_retriever(vectordb_retriever)
            chain = create_rag_chain(retriever)
            with st.spinner("⏳Query Search is in Progress. Please wait..."):
                retrieved_results = chain.invoke(query)
            st.success("Query Search is completed. You can now view the results.", icon="✅")
            st.write(f"{retrieved_results}")

if __name__ == "__main__":
    username = None
    
    # Sidebar
    st.sidebar.title("💡 Unlock LinkedIn Insights")
    username = basic_prerequisites()
    
    st.sidebar.title("🛠️ Data Ingestion to VectorDB")
    migrate_data = st.sidebar.button("🔨 Start Data Ingestion")
    if migrate_data:
        migrate_data_to_vectordb(username)
    
    # Main content
    get_insights_from_posts()