import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess_transactions, compute_rfm, market_basket_analysis

st.set_page_config(layout="wide", page_title="Retail Sales Pattern Analyzer")

# ===============================
# Helper Plot Functions
# ===============================
def plot_trend(df, period="M"):
    # Aggregate sales (count of items since Sales=1 per item)
    monthly = df.groupby(df["Date"].dt.to_period(period))["Sales"].sum()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_title("Sales Trend (Number of Items Bought)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Items Bought")
    st.pyplot(fig)

def plot_bar(df, col, top_n=10, horizontal=False):
    if col not in df.columns:
        st.warning(f"No '{col}' column found")
        return
    data = df.groupby(col)["Sales"].sum().sort_values(ascending=not horizontal).head(top_n)
    fig, ax = plt.subplots(figsize=(10,5))
    kind = "barh" if horizontal else "bar"
    data.plot(kind=kind, ax=ax)
    ax.set_title(f"Top {top_n} by {col}")
    ax.set_xlabel("Count of Items")
    ax.set_ylabel(col)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ===============================
# Streamlit App
# ===============================
def main():
    st.title("📊 Retail Sales Pattern Analyzer (Groceries Dataset)")
    st.markdown("Analyze customer buying behavior with RFM and Market Basket Analysis.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        try:
            df = preprocess_transactions(df)
        except Exception as e:
            st.error(f"Error preprocessing CSV: {e}")
            return

        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Summary
        st.subheader("Summary Statistics")
        total = df["Sales"].sum()
        avg = df["Sales"].mean()
        max_sale = df["Sales"].max()
        min_sale = df["Sales"].min()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Items Bought", f"{total:,.0f}")
        col2.metric("Average per Record", f"{avg:,.2f}")
        col3.metric("Max Items in One Record", f"{max_sale:,.0f}")
        col4.metric("Min Items in One Record", f"{min_sale:,.0f}")

        # Analysis Options
        st.subheader("Analysis Options")
        choice = st.selectbox("Choose analysis type", 
                              ["Sales Trend", "Top Products", "Top Customers", "RFM Analysis", "Market Basket Analysis"])

        if choice == "Sales Trend":
            period = st.selectbox("Period", ["Monthly", "Yearly"])
            plot_trend(df, period="M" if period=="Monthly" else "Y")

        elif choice == "Top Products":
            n = st.slider("Number of top products:", 5, 20, 10)
            plot_bar(df, "Product", top_n=n)

        elif choice == "Top Customers":
            n = st.slider("Number of top customers:", 5, 20, 10)
            plot_bar(df, "Customer", top_n=n, horizontal=True)

        elif choice == "RFM Analysis":
            rfm = compute_rfm(df)
            st.subheader("RFM Table")
            st.dataframe(rfm.head())
            st.subheader("RFM Score Distribution")
            fig, ax = plt.subplots(figsize=(10,5))
            rfm["RFM_Score"].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_xlabel("RFM Score")
            ax.set_ylabel("Count of Customers")
            st.pyplot(fig)

        elif choice == "Market Basket Analysis":
            st.markdown("⚙️ Adjust support and confidence to generate rules.")
            min_support = st.slider("Minimum Support", 0.0005, 0.05, 0.005, step=0.0005)
            min_confidence = st.slider("Minimum Confidence", 0.05, 1.0, 0.2, step=0.05)

            frequent_itemsets, rules = market_basket_analysis(
                df, min_support=min_support, min_confidence=min_confidence
            )

            st.subheader("Frequent Itemsets")
            if frequent_itemsets.empty:
                st.info("No frequent itemsets found. Try lowering minimum support.")
            else:
                st.dataframe(frequent_itemsets.sort_values("support", ascending=False).head(20))

            st.subheader("Association Rules")
            if rules.empty:
                st.info("No association rules found. Try lowering min support/confidence.")
            else:
                st.dataframe(rules.head(20))

if __name__ == "__main__":
    main()
