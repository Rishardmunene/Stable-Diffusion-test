class DataLoader:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self.fetch_image_paths()

    def fetch_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def load_image(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)