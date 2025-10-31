import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import ssl

# -------------------------------------------
# Tắt verify SSL để tránh lỗi CERTIFICATE_VERIFY_FAILED
# -------------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context

# -------------------------------------------
# Lấy tên công ty từ website HNX (nếu có)
# -------------------------------------------
def get_company_name(stock_code):
    try:
        url = f"https://www.hnx.vn/vi-vn/co-phieu-{stock_code}.html"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10, verify=False)
        if resp.status_code != 200:
            return "Không tìm thấy thông tin công ty"
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.find("title")
        if title and stock_code.upper() in title.text:
            name = title.text.split("|")[0].strip()
            return name
        return "Không tìm thấy tên công ty"
    except Exception as e:
        return f"Không thể lấy tên công ty: {e}"

# -------------------------------------------
# Hàm tải dữ liệu an toàn (có cache)
# -------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(stock_code):
    df = yf.download(f"{stock_code}.VN", period="6mo", progress=False)
    if df.empty:
        df = yf.download(stock_code, period="6mo", progress=False)
    if df is None or df.empty:
        return None

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # tìm cột close
    possible_close_cols = [c for c in df.columns if c.lower() in ("close", "adj close", "adjclose")]
    if not possible_close_cols:
        return None
    close_col = possible_close_cols[0]
    df = df[["Date", close_col]].rename(columns={close_col: "Close"})
    df = df.dropna(subset=["Date", "Close"])
    if df.empty:
        return None
    return df

# -------------------------------------------
# Huấn luyện mô hình Prophet (an toàn)
# -------------------------------------------
@st.cache_resource
def train_prophet_model(df):
    try:
        if "Date" in df.columns and "Close" in df.columns:
            df = df.rename(columns={"Date": "ds", "Close": "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["ds", "y"]).sort_values("ds")
        df = df.drop_duplicates(subset="ds", keep="last").reset_index(drop=True)

        if len(df) < 10:
            return None, "Dữ liệu quá ít để dự đoán"

        model = Prophet(daily_seasonality=True)
        model.fit(df)
        return model, None
    except Exception as e:
        return None, f"Lỗi khi train Prophet: {e}"

# -------------------------------------------
# Giao diện chính
# -------------------------------------------
st.set_page_config(page_title="Dự đoán giá cổ phiếu - Prophet", layout="wide")

st.title("📈 Dự đoán giá cổ phiếu bằng Prophet")

stock_code = st.text_input("Nhập mã cổ phiếu (ví dụ: FPT, VCB, HPG...):").strip().upper()

days_to_predict = st.slider("Số ngày muốn dự đoán", 7, 60, 14)

if stock_code:
    with st.spinner("⏳ Đang tải dữ liệu..."):
        df = load_stock_data(stock_code)

    if df is None or df.empty:
        st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này.")
    else:
        company_name = get_company_name(stock_code)
        st.subheader(f"🏢 {stock_code} - {company_name}")
        st.write("**Dữ liệu gần nhất:**")
        st.dataframe(df.tail(10))

        model, err = train_prophet_model(df)
        if err:
            st.error(err)
        else:
            with st.spinner("🧠 Đang dự đoán..."):
                future = model.make_future_dataframe(periods=days_to_predict)
                forecast = model.predict(future)

            st.subheader("📅 Biểu đồ dự đoán giá")
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Bảng dữ liệu dự đoán")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days_to_predict))

            # Xu hướng dự đoán
            last_price = df["Close"].iloc[-1]
            avg_next = forecast["yhat"].tail(days_to_predict).mean()
            diff_pct = ((avg_next - last_price) / last_price) * 100
            trend = "📈 TĂNG" if diff_pct > 0 else "📉 GIẢM"
            st.markdown(
                f"""
                ### 🔍 Phân tích xu hướng
                - Giá hiện tại: **{last_price:,.2f}**
                - Giá trung bình {days_to_predict} ngày tới: **{avg_next:,.2f}**
                - Chênh lệch: **{diff_pct:+.2f}%**
                - Dự báo xu hướng: **{trend}**
                """
            )
else:
    st.info("💡 Nhập mã cổ phiếu để bắt đầu dự đoán (ví dụ: FPT, VCB, HPG...)")
