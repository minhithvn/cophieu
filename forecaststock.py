# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import certifi

# =====================
# Streamlit page config
# =====================
st.set_page_config(page_title="📈 Dự đoán cổ phiếu Việt Nam (Prophet)", layout="wide")
st.title("📈 Dự đoán xu hướng cổ phiếu Việt Nam bằng Prophet")
st.write("Nhập mã cổ phiếu HNX/HOSE, ví dụ: FPT, SSI, VNM, VIC, HAG ...")

# =====================
# Hàm lấy tên công ty từ HNX.vn
# =====================
def get_company_name(stock_code):
    """
    Lấy tên công ty từ HNX.vn
    fallback: nếu lỗi SSL hoặc không tìm thấy → trả về thông báo
    """
    try:
        url = f"https://www.hnx.vn/vi-vn/co-phieu-{stock_code}.html"
        # Bỏ verify SSL để tránh lỗi trên máy không có chứng chỉ
        res = requests.get(url, timeout=10, verify=False)
        if res.status_code == 200:
            # Dùng parser mặc định html.parser
            soup = BeautifulSoup(res.text, "html.parser")
            h1 = soup.find("h1")
            if h1:
                return h1.text.strip()
            div_title = soup.find("div", class_="company-title")
            if div_title:
                return div_title.text.strip()
        return "Không tìm thấy tên công ty"
    except Exception as e:
        return f"Không thể lấy tên công ty: {e}"
# =====================
# Nhập dữ liệu
# =====================
stock_code = st.text_input("Nhập mã cổ phiếu:", value="FPT").upper().strip()
days_to_predict = st.slider("Số ngày cần dự đoán:", 7, 60, 15)

if st.button("🔍 Phân tích & Dự đoán") and stock_code:
    # --------------------
    # Lấy tên công ty
    # --------------------
    company_name = get_company_name(stock_code)
    st.write(f"**Mã cổ phiếu:** {stock_code}")
    st.write(f"**Tên công ty:** {company_name}")

    try:
        # --------------------
        # Lấy dữ liệu cổ phiếu 1 năm qua
        # --------------------
        df = yf.download(f"{stock_code}.VN", period="1y", progress=False)
        if df.empty:
            df = yf.download(stock_code, period="1y", progress=False)

        if df.empty:
            st.error("❌ Không tìm thấy dữ liệu cho mã này. Hãy thử mã khác.")
        else:
            # --------------------
            # Làm sạch dữ liệu
            # --------------------
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["Date", "Close"]].dropna()
            if df.empty or df["Close"].nunique() < 5:
                st.error("⚠️ Dữ liệu không đủ để huấn luyện mô hình.")
            else:
                df = df.rename(columns={"Date": "ds", "Close": "y"})
                df["ds"] = pd.to_datetime(df["ds"])
                df["y"] = df["y"].astype(float)

                # --------------------
                # Huấn luyện Prophet
                # --------------------
                model = Prophet(daily_seasonality=True)
                model.fit(df)

                # --------------------
                # Dự đoán
                # --------------------
                future = model.make_future_dataframe(periods=days_to_predict)
                forecast = model.predict(future)

                # --------------------
                # Biểu đồ giá dự đoán
                # --------------------
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                with st.expander("🔍 Xem chi tiết thành phần xu hướng"):
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)

                # --------------------
                # Phân tích xu hướng
                # --------------------
                current_price = df["y"].iloc[-1]
                avg_future = forecast["yhat"].iloc[-days_to_predict:].mean()
                change_percent = ((avg_future - current_price) / current_price) * 100
                trend = "📈 **Tăng**" if change_percent > 0 else "📉 **Giảm**"

                st.markdown(f"""
                ## 🔎 Kết quả dự báo
                - **Mã cổ phiếu:** `{stock_code}`
                - **Tên công ty:** {company_name}
                - **Giá hiện tại:** {current_price:,.2f} VND  
                - **Giá trung bình {days_to_predict} ngày tới:** {avg_future:,.2f} VND  
                - **Chênh lệch:** {change_percent:+.2f}%  
                - **Xu hướng dự kiến:** {trend}
                """)

                # --------------------
                # Xuất CSV
                # --------------------
                df_export = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days_to_predict)
                csv = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Tải dữ liệu dự đoán (CSV)",
                    data=csv,
                    file_name=f"{stock_code}_forecast_prophet.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"⚠️ Lỗi khi tải hoặc xử lý dữ liệu: {e}")
