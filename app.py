import streamlit as st
import torch
import matplotlib.pyplot as plt
import pandas as pd

from models.client_model import ClientModel
from models.server_model import ServerModel
from trainers.sflv1_trainer import train_sflv1
from trainers.sflv2_trainer import train_sflv2
from utils.data_loader import load_mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="SplitFed Dashboard", layout="wide")
st.title(" SplitFed Learning Dashboard")

with st.sidebar:
    st.header("⚙️ Configuration")
    clients = st.number_input("Number of Clients", min_value=2, max_value=10, value=2)
    rounds = st.number_input("Global Rounds", min_value=1, value=3)
    local_epochs = st.number_input("Local Epochs per Client", min_value=1, value=3)
    mode = st.selectbox("Training Mode", ["SFLV1", "SFLV2"])
    quick_mode = st.checkbox("Quick Mode (Small Dataset)", value=True)

if st.button(" Start Training"):
    st.subheader(f" Training Mode: {mode}")
    st.write(f"Clients: {clients} | Rounds: {rounds} | Local Epochs: {local_epochs} | Quick Mode: {quick_mode}")
    
    train_set, test_set = load_mnist(batch_size=32, quick_mode=quick_mode)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    client_models = [ClientModel().to(device) for _ in range(clients)]
    server_model = ServerModel().to(device)
    fed_server = None

    if mode == "SFLV1":
        acc_list, emissions_list = train_sflv1(
            client_models, server_model, fed_server,
            (train_loader, test_loader), device,
            num_rounds=rounds, epochs_per_client=local_epochs
        )

        # Accuracy plot
        st.subheader(" Accuracy vs Rounds")
        fig = plt.figure()
        plt.plot(range(1, len(acc_list) + 1), acc_list, marker='o')
        plt.title("Accuracy vs Rounds (SFLV1)")
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        st.pyplot(fig)

        # Round-wise emissions trend
        st.subheader("🌱 Carbon Emissions per Round")
        try:
            df = pd.read_csv("codecarbon/roundwise_emissions.csv")
            fig2 = plt.figure()
            plt.plot(df["Round"], df["CO₂ Emissions (kg)"], marker='o', color='green')
            plt.title("CO₂ Emissions Per Round")
            plt.xlabel("Round")
            plt.ylabel("kgCO₂eq")
            plt.grid(True)
            st.pyplot(fig2)

            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                label=" Download Emissions CSV",
                data=csv,
                file_name='roundwise_emissions.csv',
                mime='text/csv'
            )
        except:
            st.warning(" Emissions CSV not found.")
    else:
        st.warning("SFLV2 emissions tracking not implemented yet.")
