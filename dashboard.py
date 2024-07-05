import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import time
from datetime import datetime, timedelta
from EV_class import EV_BATTERY
import plotly.express as px
import plotly.graph_objects as go

# Set the page layout
st.set_page_config(layout="wide")

# Sidebar for input sliders
st.sidebar.title("User Input")

charge_option = st.sidebar.selectbox(
    "Select your charger",
    ("Electra", "Wallbox", "Circontrol"))

charging_type = st.sidebar.multiselect(
    "Charging type",
    ["V2G", "Smart", "Fast", "aFRR"])

n_EV_input = st.sidebar.number_input("Insert a number")
n_EV = int(n_EV_input)
# State of charge data
SOC_ini = st.sidebar.slider(f'Initial State of Charge', 0, 100, 50)
SOC_final = st.sidebar.slider(f'Desired State of Charge', 0, 100, 95)
    

#start_time = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
start_time = datetime(2024, 6, 19, 18, 00)
end_time = start_time + timedelta(days=1, hours=-1)

# Create the slider charging schedule timeline
charging_schedule = st.sidebar.slider(
    "Schedule your charging schedule:",
    min_value=start_time,
    max_value=end_time,
    value=(start_time + timedelta(hours=0, minutes=0), start_time + timedelta(hours=5, minutes=40)),
    step=timedelta(minutes=15),
    format="MM/DD HH:mm"
)

#=====================================================================================
#============= Run simulation ========================================================
#=====================================================================================

Batt = EV_BATTERY(P_charg=46, capacity=62, P_dischar=7, eta = 0.8, n_EV=n_EV)

# Define the start and end datetime for the range
date_range = [pd.to_datetime('06/06/2024 18:00', format='%d/%m/%Y %H:%M'), 
              pd.to_datetime('07/07/2024 18:00', format='%d/%m/%Y %H:%M')]

Batt.read_spot_price('Price_data/bourseEpex_06_06_2024.csv', resample=True, sampling =4)
Batt.read_aFFR_price('Price_data/aFRR_06_06_2024.csv', date_range=date_range)
Batt.reformate_charing_time(charging_schedule)
Batt.definec_charger_charac(charge_type=charge_option)

print(Batt.start_pos)
print(Batt.len_ones)
Batt.generate_contiguous_matrix(SOC_random=False,
                                         SOC_ini= SOC_ini/100*np.ones((n_EV,)),
                                         SOC_final= SOC_final/100*np.ones((n_EV,)))
charge, discharge, SOC = Batt.battery_dispatch('VPP')
# Generate the df only for the VPP
df_VPP = Batt.generate_result_df()

total_price= Batt.obj_value

#Look at the other benchmark
total_price_ls = []
#VPP
charge, discharge, SOC = Batt.battery_dispatch('VPP')
total_price_ls.append(Batt.obj_value)
#V1G
charge_smart, discharge_smart, SOC_smart = Batt.battery_dispatch('smart')
total_price_ls.append(Batt.obj_value)
df_Smart = Batt.generate_result_df()
#Fast
charge_fast, discharge_fast, SOC_fast = Batt.battery_dispatch('fast')
total_price_ls.append(Batt.obj_value)
df_fast = Batt.generate_result_df()
#VPP + aFRR
charge_aFRR, discharge_aFRR, SOC_aFRR = Batt.battery_dispatch('aFRR')
total_price_ls.append(-Batt.obj_value)
df_aFRR = Batt.generate_result_df()

#Compute the electra price
price_electra = 0.49*62*(SOC_final - SOC_ini)/100
total_price_ls.append(price_electra)

dict_df = {'V2G': df_VPP, 'Smart': df_Smart, 
           'Fast': df_fast, 'aFRR': df_aFRR}

# Main dashboard layout
col1, col2 = st.columns(2)

# Top-left: Text and number
col1.subheader("Charging operation for Nissan LEAF")
col1.metric(label="Charging cost [$]", value=np.round(total_price,2))
col1.metric(label="Initial SOC [%]", value=SOC_ini)
col1.metric(label="Desired SOC [%]", value=SOC_final)

# Top-right: Bar plot
col2.subheader("Comparison of the charging method")
#fig, ax = plt.subplots(figsize = (4,3), dpi = 300)
#bars = ax.bar(['VPP', 'Smart', 'Fast'], total_price_ls)
#ax.set_ylim(np.min(total_price_ls)*1.15, np.max(total_price_ls)*1.15)
# Adding annotations
#for bar, value in zip(bars, total_price_ls):
  #  height = np.round(bar.get_height(),2)
  #  print(height)
 #   ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{np.round(value,2)}', ha='center', va='bottom')
#ax.set_ylabel('Price [$]')


data_price = pd.DataFrame({'Price [$]' : total_price_ls, 'Method': ['V2G', 'Smart', 'Fast', 'aFRR','Electra']})
fig = px.bar(data_price, x='Method', y='Price [$]')

#col2.pyplot(fig)
col2.plotly_chart(fig)

# Bottom: Line plot
st.subheader("Charging profile")


# Create figure
fig = go.Figure()

# Add scatter plot for 'Price [$/MWh]'
fig.add_trace(
    go.Scatter(x=df_VPP['Time'], y=df_VPP['Price [$/MWh]'], name="Price [$/MWh]", mode='lines+markers', yaxis="y1")
)


for elem in charging_type:
    df = dict_df[elem]
    # Add scatter plot for 'SOC'
    fig.add_trace(
        go.Scatter(x=df['Time'], y=df['SOC 0'], name=f"SOC {elem}", mode='lines+markers', yaxis="y2")
    )



# Create axis objects
fig.update_layout(
    xaxis=dict(
        title="Time"
    ),
    yaxis=dict(
        title="Price [$/MWh]",
        titlefont=dict(color="#1f77b4"),
        tickfont=dict(color="#1f77b4")
    ),
    yaxis2=dict(
        title="SOC",
        titlefont=dict(color="#ff7f0e"),
        tickfont=dict(color="#ff7f0e"),
        anchor="x",
        overlaying="y",
        side="right"
    )
)

# Update layout
fig.update_layout(
    legend=dict(x=0.85, y=0),
    margin=dict(l=40, r=40, t=40, b=40)
)

event = st.plotly_chart(fig)
