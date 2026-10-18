import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except NameError:
        return False

def install_package(package):
    """Installs a python package using pip."""
    logging.info(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Dependency setup
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    install_package("torch")
    install_package("torchvision")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.transforms as transforms
except ImportError:
    install_package("torchvision")
    import torchvision.transforms as transforms

try:
    import cv2
except ImportError:
    install_package("opencv-python-headless")
    import cv2

try:
    import numpy as np
except ImportError:
    install_package("numpy")
    import numpy as np

try:
    import ezdxf
except ImportError:
    install_package("ezdxf")
    import ezdxf

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -------------------------
# U-Net Model Architecture
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    A basic U-Net architecture for Image-to-Image translation
    (e.g., noisy scanned drawing -> clean raster drawing).
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # In case the spatial dimensions don't perfectly match
            if x.shape != skip_connection.shape:
                import torch.nn.functional as F
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return torch.sigmoid(self.final_conv(x))

# -------------------------
# Dataset handling
# -------------------------
class DrawingDataset(Dataset):
    """
    Dataset to load pairs of scanned (input) and clean (target) drawings.
    Images should be split into smaller patches/tiles (e.g., 256x256 or 512x512) for memory efficiency.
    """
    def __init__(self, scanned_dir, clean_dir, img_size=(256, 256)):
        self.scanned_dir = Path(scanned_dir)
        self.clean_dir = Path(clean_dir)
        self.img_size = img_size

        # Match files by exact name
        self.images = [f.name for f in self.scanned_dir.iterdir() if f.is_file() and f.suffix in ['.png', '.jpg', '.jpeg']]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        scanned_path = self.scanned_dir / img_name
        clean_path = self.clean_dir / img_name

        scan_img = cv2.imread(str(scanned_path), cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)

        if scan_img is None or clean_img is None:
            # Fallback to zeros if file missing or corrupt
            scan_img = np.zeros(self.img_size, dtype=np.uint8)
            clean_img = np.zeros(self.img_size, dtype=np.uint8)

        # Resize for fixed size inputs (ideally data should be pre-tiled)
        scan_img = cv2.resize(scan_img, self.img_size)
        clean_img = cv2.resize(clean_img, self.img_size)

        # Invert to make lines white on black background for easier NN processing
        scan_img = cv2.bitwise_not(scan_img)
        clean_img = cv2.bitwise_not(clean_img)

        scan_tensor = self.transform(scan_img)
        clean_tensor = self.transform(clean_img)

        return scan_tensor, clean_tensor

# -------------------------
# Trainer / Inference Agent
# -------------------------
class MLCADAgent:
    def __init__(self, model_path="cad_unet_model.pth", device=None):
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        self.criterion = nn.BCELoss() # Binary Cross Entropy for line detection
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def load_model(self):
        if os.path.exists(self.model_path):
            logging.info(f"Loading existing model from {self.model_path}...")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            logging.warning(f"No existing model found at {self.model_path}. A new one will be initialized.")

    def save_model(self):
        logging.info(f"Saving model to {self.model_path}...")
        torch.save(self.model.state_dict(), self.model_path)

    def train(self, scanned_dir, clean_dir, epochs=10, batch_size=4, retrain=False):
        if retrain:
            self.load_model()

        logging.info(f"Preparing dataset from {scanned_dir} and {clean_dir}...")
        dataset = DrawingDataset(scanned_dir, clean_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        logging.info("Starting training loop...")
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (scan, clean) in enumerate(loader):
                scan, clean = scan.to(self.device), clean.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(scan)
                loss = self.criterion(predictions, clean)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        self.save_model()
        logging.info("Training complete.")

    def infer(self, input_image_path, output_dxf_path):
        """
        Runs the model on a full image, converts the cleaned raster output to a vector DXF.
        """
        self.load_model()
        self.model.eval()

        logging.info(f"Processing {input_image_path}...")
        img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {input_image_path}")

        # Keep original shape for later
        orig_h, orig_w = img.shape

        # U-Net requires input dimensions to be a multiple of 16 (due to pooling layers)
        # Pad the image rather than resizing it down to 512x512, which destroys CAD detail.
        pad_h = (16 - orig_h % 16) % 16
        pad_w = (16 - orig_w % 16) % 16
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)

        # Invert (lines become white on black bg)
        img_inverted = cv2.bitwise_not(img_padded)

        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(img_inverted).unsqueeze(0).to(self.device)

        # For very large images, this may require a GPU with significant VRAM.
        # A robust solution would tile the image, but padding preserves detail for now.
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Convert output back to numpy image
        output_np = output_tensor.squeeze().cpu().numpy()
        output_np = (output_np * 255).astype(np.uint8)

        # Crop back to original dimensions
        output_np = output_np[:orig_h, :orig_w]

        # Threshold to binary
        _, binary = cv2.threshold(output_np, 127, 255, cv2.THRESH_BINARY)

        self.vectorize_to_dxf(binary, output_dxf_path, orig_h)

    def vectorize_to_dxf(self, binary_image, output_path, img_height):
        """
        A basic vectorization pass using Hough Lines to create DXF segments from the cleaned raster.
        """
        logging.info(f"Vectorizing and exporting to DXF: {output_path}")

        # Thinning or skeletonization is usually required here for accurate single lines.
        # We'll use basic probabilistic hough transform for demonstration.
        lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)

        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Invert Y axis for DXF space
                msp.add_line((x1, img_height - y1), (x2, img_height - y2))

        doc.saveas(str(output_path))
        logging.info("DXF export complete.")

# -------------------------
# CLI Setup
# -------------------------
if __name__ == "__main__":
    if is_notebook():
        logging.info("Running in a notebook environment. Skipping CLI argument parsing.")
        logging.info("Use the MLCADAgent class directly: agent = MLCADAgent(); agent.train(...) or agent.infer(...)")
    else:
        parser = argparse.ArgumentParser(description="End-to-End ML Model for Scanned Electrical Drawing to Clean CAD Conversion.")
        subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

        # Train
        parser_train = subparsers.add_parser("train", help="Train the model from scratch on paired dataset")
        parser_train.add_argument("--scanned_dir", type=str, required=True, help="Directory containing scanned images")
        parser_train.add_argument("--clean_dir", type=str, required=True, help="Directory containing target clean images (same filenames)")
        parser_train.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
        parser_train.add_argument("--model_path", type=str, default="cad_unet_model.pth", help="Path to save the trained model")

        # Retrain
        parser_retrain = subparsers.add_parser("retrain", help="Fine-tune an existing model on new paired data")
        parser_retrain.add_argument("--scanned_dir", type=str, required=True, help="Directory containing new scanned images")
        parser_retrain.add_argument("--clean_dir", type=str, required=True, help="Directory containing new target clean images")
        parser_retrain.add_argument("--epochs", type=int, default=10, help="Number of retraining epochs")
        parser_retrain.add_argument("--model_path", type=str, default="cad_unet_model.pth", help="Path to existing model to update")

        # Infer
        parser_infer = subparsers.add_parser("infer", help="Convert a scanned drawing to DXF using the trained model")
        parser_infer.add_argument("input_image", type=str, help="Path to the input scanned drawing image")
        parser_infer.add_argument("output_dxf", type=str, help="Path to save the output DXF file")
        parser_infer.add_argument("--model_path", type=str, default="cad_unet_model.pth", help="Path to the trained model")

        args = parser.parse_args()

        agent = MLCADAgent(model_path=args.model_path)

        try:
            if args.command == "train":
                agent.train(args.scanned_dir, args.clean_dir, epochs=args.epochs, retrain=False)
            elif args.command == "retrain":
                agent.train(args.scanned_dir, args.clean_dir, epochs=args.epochs, retrain=True)
            elif args.command == "infer":
                agent.infer(args.input_image, args.output_dxf)
        except Exception as e:
            logging.error(f"Error during {args.command}: {e}")
