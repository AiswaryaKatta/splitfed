import torch
import matplotlib.pyplot as plt
import pandas as pd
from utils.fedavg import fed_avg
from privacy.differential_privacy import apply_dp_to_gradients
from privacy.pixeldp import apply_pixeldp
from codecarbon import EmissionsTracker

def evaluate(server, clients, test_loader, device):
    server.eval()
    for client in clients:
        client.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = server(clients[0](data))
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def train_sflv1(clients, server, fed_server, data_loaders, device, num_rounds=5, epochs_per_client=3):
    train_loader, test_loader = data_loaders
    round_accuracies = []
    round_emissions = []
    best_acc = 0

    for rnd in range(num_rounds):
        print(f"\n [Round {rnd+1}/{num_rounds}]")

        # Start emissions tracker per round
        tracker = EmissionsTracker(output_dir="codecarbon", save_to_file=False)
        tracker.start()

        client_updates = []
        server_updates = []

        for client_idx, client in enumerate(clients):
            print(f" Client {client_idx+1} training...")
            for epoch in range(epochs_per_client):
                for data, labels in train_loader:
                    data, labels = data.to(device), labels.to(device)
                    smashed_data = client(data)
                    smashed_data = apply_pixeldp(smashed_data)
                    output = server(smashed_data)
                    loss = torch.nn.functional.cross_entropy(output, labels)

                    server.optimizer.zero_grad()
                    client.optimizer.zero_grad()
                    loss.backward()

                    apply_dp_to_gradients(list(client.parameters()))
                    apply_dp_to_gradients(list(server.parameters()))

                    client.optimizer.step()
                    server.optimizer.step()

            client_updates.append(client)
            server_updates.append(server)

        # FedAvg aggregation
        fed_client_model = fed_avg(client_updates)
        fed_server_model = fed_avg(server_updates)
        for client in clients:
            client.load_state_dict(fed_client_model.state_dict())
        server.load_state_dict(fed_server_model.state_dict())

        # Evaluate
        acc = evaluate(server, clients, test_loader, device)
        round_accuracies.append(acc)
        best_acc = max(best_acc, acc)
        print(f" Accuracy after Round {rnd+1}: {acc:.2f}% | Best so far: {best_acc:.2f}%")

        emissions = tracker.stop()
        round_emissions.append(emissions)
        print(f" Emissions for Round {rnd+1}: {emissions:.6f} kgCO₂eq")

    # Save accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_rounds + 1), round_accuracies, marker='o')
    plt.title("Accuracy vs Rounds (SFLV1)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/sflv1_accuracy_plot.png")

    # Save round-wise emissions CSV
    df = pd.DataFrame({
        "Round": list(range(1, num_rounds + 1)),
        "CO₂ Emissions (kg)": round_emissions
    })
    df.to_csv("codecarbon/roundwise_emissions.csv", index=False)
    print(" Saved round-wise emissions to codecarbon/roundwise_emissions.csv")

    return round_accuracies, round_emissions
