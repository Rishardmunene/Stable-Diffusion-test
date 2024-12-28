def visualize_images(generated_images, epoch, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Set up the figure
    fig, axes = plt.subplots(1, len(generated_images), figsize=(15, 5))
    for ax, img in zip(axes, generated_images):
        img = (img.permute(1, 2, 0).cpu().detach().numpy() + 1) / 2  # Rescale to [0, 1]
        ax.imshow(img)
        ax.axis('off')

    plt.suptitle(f'Generated Images at Epoch {epoch}')
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()