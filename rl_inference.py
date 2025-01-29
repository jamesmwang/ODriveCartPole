# rl_inference.py

import yaml
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
import time

class CartPolePolicy:
    def __init__(self, model_path):
        self.model_dict = self.load_model(model_path)
        self.action_min = self.model_dict["action_min"]
        self.action_max = self.model_dict["action_max"]
        self.in_shape = self.model_dict["in_shape"]
        self.model_pytorch = self.build_network(self.model_dict)


    def load_model(self, model_path: str):
        try:
            with open(model_path, "r") as file:
                model_data = yaml.safe_load(file)
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {model_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

        return model_data

    def print_model_attributes(self):
        print(f"Keys: {self.model_dict.keys()}")

        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")
        print(f"Input shape: {self.in_shape}")

    def build_network(self, config: dict) -> nn.Module:
        """
        Given the parsed network configuration, build a PyTorch Sequential model
        and load the weights from the YAML if available.
        """
        layers = []

        # The YAML indicates `in_shape: [null, 6]`, i.e. (batch_size, 6).
        in_features = config["in_shape"][1]

        for layer_cfg in config["layers"]:
            if layer_cfg["type"] == "dense":
                out_features = layer_cfg["shape"][1]

                # Create linear layer
                linear_layer = nn.Linear(in_features, out_features)

                # If 'weights' are provided in the config, load them
                if "weights" in layer_cfg and layer_cfg["weights"] is not None:
                    w = torch.tensor(layer_cfg["weights"][0], dtype=torch.float).T
                    b = torch.tensor(layer_cfg["weights"][1], dtype=torch.float)

                    # Validate shapes
                    if list(w.shape) != [out_features, in_features]:
                        raise ValueError(
                            f"Weight shape {list(w.shape)} doesn't match "
                            f"[{out_features}, {in_features}]"
                        )
                    if list(b.shape) != [out_features]:
                        raise ValueError(f"Bias shape {list(b.shape)} doesn't match [{out_features}]")

                    # Load data into the linear layer
                    with torch.no_grad():
                        linear_layer.weight.copy_(w)
                        linear_layer.bias.copy_(b)

                layers.append(linear_layer)

                # Add activation if present
                activation = layer_cfg.get("activation", None)
                if activation is not None:
                    activation = activation.lower()
                    if activation == "relu":
                        layers.append(nn.ReLU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
                    elif activation == "sigmoid":
                        layers.append(nn.Sigmoid())
                    else:
                        raise ValueError(f"Unsupported activation: {activation}")

                # Update for next layer
                in_features = out_features

            else:
                raise ValueError(f"Unsupported layer type: {layer_cfg['type']}")

        return nn.Sequential(*layers)

    # Expects input in the form (x, theta, x_dot, theta_dot)
    # Theta 0 is when pendulum is hanging down
    def inference(self, input_tensor: torch.Tensor, scaled=True):
        with torch.no_grad():
            tensor_output = self.model_pytorch(input_tensor)
            float_output = tensor_output.item()
            output = float_output
            if scaled:
                if float_output >= 0:
                    output = float_output * np.abs(self.action_max)
                else:
                    output = float_output * np.abs(self.action_min)
            return output

def main():
    model_path = "rl_model/params_012825.yaml"

    print("Loading and converting model...")
    cart_pole_policy = CartPolePolicy(model_path)
    cart_pole_policy.print_model_attributes()

    summary(cart_pole_policy.model_pytorch, input_size=(4,))

    print("Running test inference...")
    timestamp = time.time()
    test_tensor = torch.tensor([0.024, 0.608, 1.552, 6.571], dtype=torch.float32)
    test_inference = cart_pole_policy.inference(test_tensor)
    inference_time = int((time.time() - timestamp) * 1000)
    print(f"Test tensor:")
    print(f"Inference: {test_inference}")
    print(f"Inference complete in {inference_time} ms")

if __name__ == "__main__":
    main()
