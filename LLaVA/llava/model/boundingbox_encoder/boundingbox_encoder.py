import re
import torch
import torch.nn as nn

class CLIPBoundingBoxEncoder(nn.Module):
    def __init__(self, input_channels, output_dim, hidden_dim=512, patch_size=24, num_layers=4, num_heads=4, image_size=336, bb_projector_type="linear"):
        """
        Mimics the CLIP encoder architecture for bounding box features, with separate handling for depth info.

        Args:
            input_channels (int): Number of input channels (e.g., 35 = 34 classes for bounding box maps + 1 for depth).
            patch_size (int): Size of each patch (e.g., 16x16).
            output_dim (int): Output dimensionality of the encoder.
            num_layers (int): Number of Transformer encoder layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimensionality of the Transformer.
        """
        super(CLIPBoundingBoxEncoder, self).__init__()
        print("input_channels", input_channels)
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.projector_type = bb_projector_type

        # Separate depth channel processing (depth = 1 channel)
        self.depth_patch_embedding = nn.Conv2d(
            1, hidden_dim, kernel_size=patch_size, stride=patch_size
        )  # Output: (B, hidden_dim, H/patch_size, W/patch_size)

        # Patch embedding for the bounding box (34 channels)
        self.bb_patch_embedding = nn.Conv2d(
            input_channels - 1, hidden_dim, kernel_size=patch_size, stride=patch_size
        )  # Output: (B, hidden_dim, H/patch_size, W/patch_size)

        # Learnable positional embeddings
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, (image_size // patch_size) ** 2, hidden_dim)
        )  # Assuming input size is 336, 336

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4),
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self.apply(self.init_weights)
        self.check_and_replace_nans()

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Use Xavier normal initialization for Linear layers
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # Xavier normal initialization for embedding layers
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm initialization (with optional small epsilon for numerical stability)
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def check_and_replace_nans(self):
        """Check for NaNs in parameters and replace them."""
        for name, param in self.named_parameters():
            if param.device.type == "meta":
                continue
            if torch.isnan(param).any():
                print(f"NaN detected in {name}. Replacing with safe values.")
                # Replace NaNs with random values or zeros
                param.data = torch.randn_like(param) if param.size(0) > 1 else torch.zeros_like(param)
                # Alternatively, you can also replace NaNs with a small constant:
                # param.data = torch.full_like(param, fill_value=0.01)  # Replace NaNs with a small constant value

    def build_boundingbox_projector(self):

        if self.projector_type == 'linear':
            return nn.Linear(self.hidden_dim, self.output_dim)

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.hidden_dim, self.output_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.output_dim, self.output_dim))
            return nn.Sequential(*modules)

        raise ValueError(f'Unknown projector type: {self.projector_type}')

    def forward(self, x):
        """
        Forward pass for the bounding box encoder with separate depth handling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, input_channels, H, W).
                - input_channels: 35 (34 bounding box classes + 1 depth channel)

        Returns:
            torch.Tensor: Encoded features of shape (B, N_patches, output_dim).
        """
        # Separate depth channel (last channel) and process
        depth_channel = x[:, -1:, :, :]  # (B, 1, H, W)
        bbox_channels = x[:, :-1, :, :]  # (B, 34, H, W)

        # for i in range(depth_channel.shape[2]):
        #     for j in range(depth_channel.shape[3]):
        #         # print to decimal point 1
        #         print(f"{depth_channel[0, 0, i, j]:.0f}", end=" ")
        #     print()
                
        # Process depth channel
        depth_patches = self.depth_patch_embedding(depth_channel)  # Shape: (B, hidden_dim, H/patch_size, W/patch_size)

        depth_patches = depth_patches.flatten(2).permute(0, 2, 1)  # Shape: (B, N_depth_patches, hidden_dim)

        # Process bounding box channels
        bbox_patches = self.bb_patch_embedding(bbox_channels)  # Shape: (B, hidden_dim, H/patch_size, W/patch_size)
        bbox_patches = bbox_patches.flatten(2).permute(0, 2, 1)  # Shape: (B, N_bbox_patches, hidden_dim)

        # Combine depth and bbox patches (sum them)
        combined_patches = (depth_patches + bbox_patches) / 2

        # Add positional embeddings
        combined_patches += self.positional_embeddings  # Add positional embeddings

        # Transformer requires input shape (N_patches, B, hidden_dim)
        combined_patches = combined_patches.permute(1, 0, 2)  # Shape: (N_combined_patches, B, hidden_dim)

        # Pass through Transformer
        encoded_features = self.transformer(combined_patches)  # Shape: (N_combined_patches, B, hidden_dim)

        # Revert to (B, N_patches, hidden_dim) and project to output_dim
        encoded_features = encoded_features.permute(1, 0, 2)  # Shape: (B, N_combined_patches, hidden_dim)
        output_features = self.output_projection(encoded_features)  # Shape: (B, N_combined_patches, output_dim)

        
        return output_features


# Example usage:
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bbox_encoder = CLIPBoundingBoxEncoder(input_channels=35, patch_size=16, output_dim=4096).to(dtype=torch.bfloat16, device=DEVICE)
    # random binary 34 classes + 1 depth channel
    bbox = torch.randint(0, 2, (1, 34, 336, 336)).to(dtype=torch.bfloat16, device=DEVICE)
    depth = torch.rand((1, 1, 336, 336)).to(dtype=torch.bfloat16, device=DEVICE)
    x = torch.cat([bbox, depth], dim=1).to(dtype=torch.bfloat16, device=DEVICE)
    bbox_encoder.to(dtype=torch.bfloat16, device=DEVICE)
    for p in bbox_encoder.parameters():
        p.requires_grad = True


    for name, param in bbox_encoder.named_parameters():
        if torch.isnan(param).any():
            print(f"Found nan in {name}")

    encoded_features = bbox_encoder(x)
    print(encoded_features.shape)  # Should be (8, N_patches, 4096)
