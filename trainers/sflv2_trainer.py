import torch
import matplotlib.pyplot as plt
from privacy.differential_privacy import apply_dp_to_gradients
from privacy.pixeldp import apply_pixeldp

def evaluate(server, clients, test_loader, device):
    server.eval()
    for client in clients:
        client.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = server(clients[0](data))  # all clients are similar
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    return acc

def train_sflv2(clients, server, fed_server, data_loaders, device, num_rounds=5, epochs_per_client=3):
    train_loader, test_loader = data_loaders
    round_accuracies = []
    best_acc = 0

    for rnd in range(num_rounds):
        print(f"\n [Round {rnd+1}/{num_rounds}]")

        for client_idx, client in enumerate(clients):
            print(f" Training Client {client_idx+1} sequentially...")

            for epoch in range(epochs_per_client):
                for data, labels in train_loader:
                    data, labels = data.to(device), labels.to(device)

                    smashed_data = client(data)
                    smashed_data = apply_pixeldp(smashed_data)

                    output = server(smashed_data)
                    loss = torch.nn.functional.cross_entropy(output, labels)

                    server.zero_grad()
                    client.zero_grad()
                    loss.backward()
                    apply_dp_to_gradients(list(client.parameters()))
                    apply_dp_to_gradients(list(server.parameters()))
                    server.optimizer.step()
                    client.optimizer.step()

            # Update client model from the current round
            clients[client_idx].load_state_dict(client.state_dict())

        # Sync all clients with latest
        final_weights = clients[-1].state_dict()
        for client in clients:
            client.load_state_dict(final_weights)

        acc = evaluate(server, clients, test_loader, device)
        round_accuracies.append(acc)
        best_acc = max(best_acc, acc)
        print(f" Accuracy after Round {rnd+1}: {acc:.2f}% | Best so far: {best_acc:.2f}%")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_rounds + 1), round_accuracies, marker='o')
    plt.title("Accuracy vs Rounds (SFLV2)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/sflv2_accuracy_plot.png")
    print(" SFLV2 accuracy graph saved as 'static/sflv2_accuracy_plot.png'")

    return round_accuracies
