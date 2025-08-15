# Demand Forecasting and Inventory Optimization

![Project Banner](https://via.placeholder.com/1200x200.png?text=Demand+Forecasting+and+Inventory+Optimization)  
*Streamlining supply chain operations with advanced forecasting and optimization*

## 📖 Project Overview

This project delivers an end-to-end machine learning pipeline for demand forecasting and inventory optimization using the `demand_inventory.csv` dataset. It integrates Prophet and XGBoost models for accurate demand predictions, calculates optimal inventory metrics (EOQ, Reorder Point, Safety Stock), and deploys a Streamlit app for real-time analysis. Interactive Plotly visualizations provide insights into demand trends and inventory levels, making it ideal for supply chain stakeholders and a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Forecast Demand**: Predict future customer demand using time-series and feature-based models.
- **Optimize Inventory**: Calculate optimal order quantities, reorder points, and safety stock.
- **Visualize Trends**: Provide interactive visualizations for demand and inventory analysis.
- **Deploy Interactive Interface**: Build a Streamlit app for real-time forecasting and optimization.
- **Enable Supply Chain Insights**: Offer actionable recommendations for operational efficiency.

## 📊 Dataset Description

The dataset (`demand_inventory.csv`) contains daily demand and inventory data for product P1 from June to August 2023:

- **Features**:
  - `Date`: Date of observation.
  - `Product_ID`: Product identifier (P1).
  - `Demand`: Daily customer demand.
  - `Inventory`: Daily inventory level.
- **Insights**:
  - Size: 62 records, 4 columns.
  - Context: Tracks demand and inventory depletion, with inventory reaching zero in mid-July.
  - Preprocessing: Extracted features (day, month, weekday, lag, rolling mean) for modeling.

## 🛠 Methodology

The pipeline is implemented in `demand_forecasting.py`:

1. **Data Acquisition**:
   - Loaded `demand_inventory.csv` and converted dates to datetime.

2. **Data Preprocessing**:
   - Extracted features (day, month, weekday, lag, rolling mean) for forecasting.

3. **Demand Forecasting**:
   - Trained Prophet for time-series forecasting and XGBoost for feature-based forecasting.
   - Evaluated models using RMSE and MAE.

4. **Inventory Optimization**:
   - Calculated EOQ, Reorder Point, Safety Stock, and total cost based on user inputs.

5. **Streamlit Deployment**:
   - Built a Streamlit app for real-time forecasting and inventory optimization.
   - Allowed user inputs for forecast horizon and cost parameters.

6. **Visualizations**:
   - Created interactive Plotly charts for historical demand/inventory and forecasts.

7. **Outputs**:
   - Saved processed data, model, and visualizations.

## 📈 Key Results

- **Forecast Performance**:
  - Prophet excels in capturing time-series patterns; XGBoost leverages engineered features.
  - Low RMSE and MAE indicate reliable forecasts.
- **Inventory Optimization**:
  - Optimal Order Quantity (EOQ): ~236 units.
  - Reorder Point: ~235 units.
  - Safety Stock: ~114 units.
  - Total Cost: ~$562, balancing holding and ordering costs.
- **Visualizations**:
  - Interactive line chart of historical demand and inventory.
  - Forecast chart for user-specified horizons.
- **Supply Chain Insights**:
  - Accurate forecasts enable proactive inventory management.
  - Optimized metrics reduce costs and prevent stockouts.
  - Applicable to retail, manufacturing, and distribution.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `prophet`, `xgboost`, `plotly`, `streamlit`, `matplotlib`, `seaborn`
- Dataset: `demand_inventory.csv` (included in the repository)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   cd portfolio/Intermediate-Projects
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` with:
   ```
   pandas==1.5.3
   numpy==1.23.5
   prophet==1.1.4
   xgboost==1.7.5
   plotly==5.15.0
   streamlit==1.22.0
   matplotlib==3.7.1
   seaborn==0.12.2
   ```

3. Ensure `demand_inventory.csv` is in the project directory.

### Running the Project
1. Run the Streamlit app:
   ```bash
   streamlit run demand_forecasting.py
   ```
2. Access the app at `http://localhost:8501` to forecast demand and optimize inventory.
3. Open HTML files (e.g., `demand_forecast.html`) for standalone visualizations.

## 📋 Usage

- **For Supply Chain Stakeholders**: Use the Streamlit app to forecast demand and optimize inventory, presenting visualizations to streamline operations.
- **For Data Scientists**: Extend the pipeline with additional models (e.g., ARIMA) or external factors (e.g., promotions).
- **For Developers**: Deploy the Streamlit app on a cloud platform (e.g., Streamlit Cloud) for broader access.

## 🔮 Future Improvements

- **Advanced Forecasting**: Incorporate external factors like holidays or promotions.
- **Multi-Product Support**: Extend to multiple products in the dataset.
- **Real-Time Data**: Integrate live sales data for dynamic forecasting.
- **Enhanced Visualizations**: Add interactive dashboards for supply chain metrics.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: `demand_inventory.csv` for demand and inventory data.
- **Tools**: Built with `prophet`, `xgboost`, `plotly`, `streamlit`, and other open-source libraries.
- **Inspiration**: Thanks to Aman Kharwal and the data science community for foundational ideas.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in demand forecasting and inventory optimization. Last updated: August 15, 2025.*