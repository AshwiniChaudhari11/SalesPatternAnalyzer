import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ===============================
# Preprocess CSV
# ===============================
def preprocess_transactions(df):
    df.columns = df.columns.str.strip()

    # Handle Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.to_datetime("today")

    # Ensure Sales column (each product counts as 1 item)
    if "Sales" not in df.columns:
        df["Sales"] = 1

    # Ensure Customer
    if "Customer" not in df.columns:
        if "Member_number" in df.columns:
            df["Customer"] = df["Member_number"]
        else:
            df["Customer"] = "Unknown"

    # Ensure Product
    if "Product" not in df.columns:
        if "itemDescription" in df.columns:
            df["Product"] = df["itemDescription"]
        elif "item" in df.columns:
            df["Product"] = df["item"]
        else:
            raise ValueError("CSV must contain a product column (e.g., itemDescription).")

    # Ensure TransactionID
    if "TransactionID" not in df.columns:
        df["TransactionID"] = df.groupby(["Customer", "Date"]).ngroup()

    return df.dropna(subset=["Customer", "Date", "Product", "TransactionID"])

# ===============================
# RFM Analysis
# ===============================
def compute_rfm(df):
    snapshot_date = df["Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer").agg(
        Recency=pd.NamedAgg(column="Date", aggfunc=lambda x: (snapshot_date - x.max()).days),
        Frequency=pd.NamedAgg(column="Date", aggfunc='count'),
        Monetary=pd.NamedAgg(column="Sales", aggfunc='sum')
    ).reset_index()

    def safe_qcut(series, name):
        unique_vals = series.nunique()
        if unique_vals <= 1:
            return pd.Series([1]*len(series), index=series.index)
        bins = min(4, unique_vals)
        return pd.qcut(series.rank(method='first'), q=bins,
                       labels=range(bins, 0, -1) if name=='R' else range(1, bins+1),
                       duplicates='drop')

    rfm['R_Score'] = safe_qcut(rfm['Recency'], 'R')
    rfm['F_Score'] = safe_qcut(rfm['Frequency'], 'F')
    rfm['M_Score'] = safe_qcut(rfm['Monetary'], 'M')
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    return rfm

# ===============================
# Market Basket Analysis
# ===============================
def market_basket_analysis(df, min_support=0.001, min_confidence=0.1):
    # Group by Transaction
    grouped = df.groupby('TransactionID')['Product'].apply(list)
    grouped = grouped[grouped.apply(len) > 1]  # keep only multi-item baskets

    if grouped.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(grouped).transform(grouped)
    basket_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Frequent itemsets
    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    # Association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]] \
                .sort_values("confidence", ascending=False)

    return frequent_itemsets, rules
