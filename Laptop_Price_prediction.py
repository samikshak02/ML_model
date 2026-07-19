import streamlit as st
import pandas as pd
import joblib

# Load model and files
model = joblib.load("LR_laptop_data.pkl")
scaler = joblib.load("scaler (1).pkl")
encoded_columns = joblib.load("columns (1).pkl")

st.set_page_config(
    page_title="Laptop Price Prediction"
    )

st.title("Laptop Price Prediction")
st.write("Enter Laptop Specifications")

company = st.selectbox("Company",
["Dell","Lenovo","HP","Asus","Acer","Apple","MSI","Toshiba","Samsung",
 "Huawei","Microsoft","Xiaomi","Vero","Chuwi","Mediacom","Razer","LG","Google","Fujitsu"])

typename = st.selectbox("Type",
["Notebook","Ultrabook","Gaming","2 in 1 Convertible","Workstation","Netbook"])

inches = st.number_input("Screen Size",
                         min_value=10.0,
                         max_value=20.0,
                         value=15.6)


screen = st.text_input("Screen Resolution","1920x1080")

cpu = st.selectbox(
    "CPU",
    [
        "Intel Core i3",
        "Intel Core i5",
        "Intel Core i7",
        "Intel Core M",
        "Intel Celeron",
        "Intel Pentium",
        "Intel Xeon",
        "AMD A-Series",
        "AMD E-Series",
        "AMD Ryzen",
        "Samsung Cortex",
        "Other"
    ]
)

ram = st.selectbox("RAM",
["2GB","4GB","6GB","8GB","12GB","16GB","24GB","32GB","64GB"])

memory = st.text_input("Memory","256GB SSD")

gpu = st.selectbox(
    "GPU",
    [
        "Intel HD Graphics 620",
        "Intel HD Graphics 520",
        "Intel UHD Graphics 620",
        "Intel Iris Graphics",
        "Intel Iris Plus Graphics 640",
        "Intel Iris Pro Graphics",
        "Nvidia GeForce GTX 1050",
        "Nvidia GeForce GTX 1060",
        "Nvidia GeForce GTX 1070",
        "Nvidia GeForce GTX 1080",
        "Nvidia GeForce 940MX",
        "Nvidia GeForce MX150",
        "AMD Radeon R5",
        "AMD Radeon R7",
        "AMD Radeon RX 560",
        "AMD FirePro",
        "ARM Mali T860 MP4"
    ]
)

opsys = st.selectbox("Operating System",
["Windows 10","Windows 7","Linux","Mac OS X","macOS","No OS","Chrome OS"])

weight = st.selectbox("Weight",
["1.05kg","1.25kg","1.37kg","1.5kg","1.8kg","2.0kg","2.2kg","2.5kg","3.0kg"])

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Unnamed: 0":[0],
        "Company":[company],
        "TypeName":[typename],
        "Inches":[inches],
        "ScreenResolution":[screen],
        "Cpu":[cpu],
        "Ram":[ram],
        "Memory":[memory],
        "Gpu":[gpu],
        "OpSys":[opsys],
        "Weight":[weight]
    })

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=encoded_columns, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"Predicted Laptop Price: ₹ {prediction[0]:,.2f}")