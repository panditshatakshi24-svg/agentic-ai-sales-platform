import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)


# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI Sales Intelligence Platform",
    layout="wide"
)

st.title("AI Sales Intelligence Platform")

st.markdown(
    "Upload sales data and interact with AI-powered analytics."
)


# -----------------------------------
# CHAT MEMORY
# -----------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Sales CSV",
    type=["csv"]
)


# -----------------------------------
# PDF FUNCTION
# -----------------------------------

def create_pdf(text):

    doc = SimpleDocTemplate(
        "ai_report.pdf"
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            text,
            styles["BodyText"]
        )
    ]

    doc.build(story)

    return "ai_report.pdf"


# -----------------------------------
# CREATE RAG DATABASE
# -----------------------------------

def create_rag_database(df):

    # Convert dataframe to text
    text_data = df.to_string()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(
        text_data
    )

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    vector_db = Chroma.from_texts(
        chunks,
        embeddings
    )

    return vector_db


# -----------------------------------
# MAIN APP
# -----------------------------------

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(
        uploaded_file
    )

    # Show dataset
    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(df)

    # -----------------------------------
    # OPTIONAL DATE COLUMN
    # -----------------------------------

    if "order_date" in df.columns:

        df["order_date"] = pd.to_datetime(
            df["order_date"]
        )

    # -----------------------------------
    # CREATE REVENUE COLUMN
    # -----------------------------------

    df["order_revenue"] = (
        df["quantity"]
        * df["unit_price"]
        * (
            1
            - df["discount_percent"] / 100
        )
    )

    # -----------------------------------
    # CREATE VECTOR DATABASE
    # -----------------------------------

    vector_db = create_rag_database(df)

    # -----------------------------------
    # METRICS
    # -----------------------------------

    total_revenue = round(
        df["order_revenue"].sum(),
        2
    )

    average_order_value = round(
        df["order_revenue"].mean(),
        2
    )

    total_orders = len(df)

    best_category = (
        df.groupby("product_category")[
            "order_revenue"
        ]
        .sum()
        .idxmax()
    )

    top_payment_method = (
        df.groupby("payment_method")[
            "order_revenue"
        ]
        .sum()
        .idxmax()
    )

    metrics = {
        "total_revenue": total_revenue,
        "average_order_value": average_order_value,
        "orders": total_orders,
        "best_category": best_category,
        "top_payment_method": top_payment_method
    }

    # -----------------------------------
    # SIDEBAR
    # -----------------------------------

    st.sidebar.header(
        "Business Summary"
    )

    st.sidebar.write(
        f"Total Revenue: {total_revenue}"
    )

    st.sidebar.write(
        f"Best Category: {best_category}"
    )

    st.sidebar.write(
        f"Top Payment Method: {top_payment_method}"
    )

    # -----------------------------------
    # KPI CARDS
    # -----------------------------------

    st.subheader(
        "Business Metrics"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Revenue",
        total_revenue
    )

    col2.metric(
        "Average Order Value",
        average_order_value
    )

    col3.metric(
        "Orders",
        total_orders
    )

    # -----------------------------------
    # CATEGORY CHART
    # -----------------------------------

    st.subheader(
        "Revenue by Category"
    )

    category_sales = df.groupby(
        "product_category"
    )["order_revenue"].sum()

    fig, ax = plt.subplots(
        figsize=(6, 3)
    )

    category_sales.plot(
        kind="bar",
        ax=ax
    )

    plt.xticks(rotation=0)

    st.pyplot(fig)

    # -----------------------------------
    # REGION CHART
    # -----------------------------------

    st.subheader(
        "Revenue by Region"
    )

    region_sales = df.groupby(
        "region"
    )["order_revenue"].sum()

    fig2, ax2 = plt.subplots(
        figsize=(6, 3)
    )

    region_sales.plot(
        kind="bar",
        ax=ax2
    )

    plt.xticks(rotation=0)

    st.pyplot(fig2)

    # -----------------------------------
    # TREND CHART
    # -----------------------------------

    if "order_date" in df.columns:

        st.subheader(
            "Revenue Trend Over Time"
        )

        daily_sales = df.groupby(
            "order_date"
        )["order_revenue"].sum()

        fig3, ax3 = plt.subplots(
            figsize=(7, 3)
        )

        daily_sales.plot(
            kind="line",
            marker="o",
            ax=ax3
        )

        st.pyplot(fig3)

    # -----------------------------------
    # LOAD LOCAL LLM
    # -----------------------------------

    llm = OllamaLLM(
        model="phi3",
        temperature=0
    )

    # -----------------------------------
    # AI REPORT PROMPT
    # -----------------------------------

    prompt = PromptTemplate.from_template("""

    You are a business analyst.

    Analyze these metrics:

    {metrics}

    Give:
    - 3 business insights
    - 3 recommendations

    Keep answers short and professional.
    """)

    # -----------------------------------
    # GENERATE AI REPORT
    # -----------------------------------

    with st.spinner(
        "Generating AI Insights..."
    ):

        response = llm.invoke(

            prompt.format(
                metrics=json.dumps(
                    metrics,
                    indent=2
                )
            )
        )

    # -----------------------------------
    # SHOW AI REPORT
    # -----------------------------------

    st.subheader(
        "AI Insights & Recommendations"
    )

    st.write(response)

    # -----------------------------------
    # RAG AI CHAT
    # -----------------------------------

    st.subheader(
        "Ask AI About Your Data"
    )

    user_question = st.text_input(
        "Ask a business question"
    )

    if user_question:

        # Save user message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        # Search relevant chunks
        docs = vector_db.similarity_search(
            user_question,
            k=3
        )

        retrieved_text = "\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        # Generate answer
        with st.spinner(
            "AI thinking..."
        ):

            answer = llm.invoke(
                f"""

                SALES DATA:
                {retrieved_text}

                CHAT HISTORY:
                {st.session_state.chat_history}

                QUESTION:
                {user_question}

                Give a short business answer.
                """
            )

        # Save AI response
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.write(answer)

    # -----------------------------------
    # DOWNLOAD TXT REPORT
    # -----------------------------------

    st.download_button(
        label="Download AI Report",
        data=response,
        file_name="ai_report.txt",
        mime="text/plain"
    )

    # -----------------------------------
    # PDF EXPORT
    # -----------------------------------

    pdf_file = create_pdf(
        response
    )

    with open(
        pdf_file,
        "rb"
    ) as f:

        st.download_button(
            "Download PDF Report",
            f,
            file_name="AI_Report.pdf"
        )

    # -----------------------------------
    # FOOTER
    # -----------------------------------

    st.markdown("---")

    st.caption(
        "Built with Streamlit, ChromaDB, LangChain, Ollama, and Llama3"
    )
