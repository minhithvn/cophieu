import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dự đoán xu hướng cổ phiếu (Prophet)", layout="wide")
st.title("📈 Dự đoán xu hướng cổ phiếu Việt Nam (HNX / HOSE) bằng Prophet")
st.write("Nhập mã cổ phiếu bạn muốn dự đoán (ví dụ: `FPT`, `VNM`, `SSI`, `VIC`, `HAG` ...)")

symbol = st.text_input("Nhập mã cổ phiếu:", value="FPT").upper().strip()
days_to_predict = st.slider("Số ngày cần dự đoán:", 7, 60, 15)

if st.button("🔍 Phân tích & Dự đoán"):
    try:
        # --------------------------
        # 1️⃣ Lấy dữ liệu
        # --------------------------
        df = yf.download(f"{symbol}.VN", period="1y", progress=False)

        # Nếu rỗng thì thử bỏ đuôi .VN
        if df.empty:
            df = yf.download(symbol, period="1y", progress=False)

        if df.empty:
            st.error("❌ Không tìm thấy dữ liệu cho mã này. Hãy thử mã khác (FPT, SSI, VNM, VIC...)")
        else:
            # --------------------------
            # 2️⃣ Làm sạch dữ liệu
            # --------------------------
            df = df.reset_index()

            # Một số mã có MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            # Chỉ giữ các dòng có giá trị hợp lệ
            df = df[["Date", "Close"]].dropna()

            if df.empty or df["Close"].nunique() < 5:
                st.error("⚠️ Dữ liệu không đủ để huấn luyện mô hình.")
            else:
                # --------------------------
                # 3️⃣ Chuẩn hóa dữ liệu Prophet
                # --------------------------
                df = df.rename(columns={"Date": "ds", "Close": "y"})
                df["ds"] = pd.to_datetime(df["ds"])
                df["y"] = df["y"].astype(float)

                # --------------------------
                # 4️⃣ Huấn luyện Prophet
                # --------------------------
                model = Prophet(daily_seasonality=True)
                model.fit(df)

                # --------------------------
                # 5️⃣ Dự đoán
                # --------------------------
                future = model.make_future_dataframe(periods=days_to_predict)
                forecast = model.predict(future)

                # --------------------------
                # 6️⃣ Biểu đồ
                # --------------------------
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                with st.expander("🔍 Xem chi tiết thành phần xu hướng"):
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)

                # --------------------------
                # 7️⃣ Phân tích kết quả
                # --------------------------
                current_price = df["y"].iloc[-1]
                avg_future = forecast["yhat"].iloc[-days_to_predict:].mean()
                change_percent = ((avg_future - current_price) / current_price) * 100
                trend = "📈 **Tăng**" if change_percent > 0 else "📉 **Giảm**"

                st.markdown(f"""
                ## 🔎 Kết quả dự báo
                - **Mã cổ phiếu:** `{symbol}`
                - **Giá hiện tại:** {current_price:,.2f} VND  
                - **Giá trung bình {days_to_predict} ngày tới:** {avg_future:,.2f} VND  
                - **Chênh lệch:** {change_percent:+.2f}%  
                - **Xu hướng dự kiến:** {trend}
                """)

                # --------------------------
                # 8️⃣ Xuất CSV
                # --------------------------
                df_export = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days_to_predict)
                csv = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Tải dữ liệu dự đoán (CSV)",
                    data=csv,
                    file_name=f"{symbol}_forecast_prophet.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"⚠️ Lỗi khi tải hoặc xử lý dữ liệu: {e}")
