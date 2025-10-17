# Projecting the Human Loss: A Time-Series Forecast of Palestinian Fatalities

This project uses machine learning to create a time-series forecast that projects the potential human loss. By analyzing historical fatality data from 2000 to 2023, the model generates a forecast for the coming years, illustrating a potential future if historical trends of violence persist.

The primary goal is not to predict the future with certainty, but to provide a stark, data-driven visualization of the long-term consequences of the genocide. The dashboard serves as an analytical tool to foster a deeper understanding of the scale of loss and the urgent need for a peaceful resolution.

### Model & Visualization

* **Model**: A SARIMA (Seasonal AutoRegressive Integrated Moving Average) time-series model is trained on monthly fatality counts to learn the underlying trends, seasonality, and patterns in the historical data.
* **Dashboard**: The core of the project is an interactive dashboard built with Streamlit. It displays:
    * A solid line representing the **actual number of Palestinian fatalities** recorded each month since 2000.
    * A dotted line representing the **model's forecast** for the next several years, showing the projected continuation of this tragic trend.

### **Ethical Considerations & Model Limitations**

It is crucial to understand that this is a **mathematical projection, not a prophecy**. This model **cannot predict**:
* New peace treaties or successful de-escalations.
* Sudden, large-scale military operations or escalations.
* Fundamental shifts in the political landscape.

The forecast's value is in its ability to answer the question: "What might the human loss look like if nothing changes?"

### Technology Stack 

* **Forecasting Model**: `statsmodels` (for SARIMA), Pandas
* **Web Dashboard**: Streamlit, Plotly Express
* **Deployment**: Docker, GitHub Actions, AWS EC2

### Local Setup

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Reproduce the Pipeline**:
    This will process the data and train the forecasting model.
    ```bash
    pip install -r requirements.txt
    dvc repro
    ```

3.  **Run the Dashboard**:
    ```bash
    streamlit run app.py
    ```
