def log_metrics(epoch, d_loss, g_loss, log_file='logs/training_metrics.log'):
    with open(log_file, 'a') as f:
        f.write(f'Epoch: {epoch}, Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}\n')


def save_loss_plot(losses, title, filename):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(losses, label=title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def generate_noise(batch_size, noise_dim=100, device='cpu'):
    return torch.randn(batch_size, noise_dim, 1, 1, device=device)