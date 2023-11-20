import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        device="cpu",
        dropout: float = 0.0,
        dtype=torch.bfloat16,
    ):
        super(MLP, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(input_size, hidden_size, device=device, dtype=dtype)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


class dense_layer_variants(
    torch.nn.Module,
):
    def __init__(
        self,
        para_dim_in,
        para_dropout_prob,
        para_archi_type,
    ):
        """
        Args.:
          para_dim_in: int, input dimensionality
          para_dropout_prob: float
          para_archi_type: string from the set ['identity', 'plain', 'wide', 'residual', 'residual_with_input']
        """
        super().__init__()
        self.archi_type = para_archi_type

        if self.archi_type == "identity":
            self.identity = torch.nn.Identity()

        else:
            dense1 = nn.ModuleList()
            dense1.append(nn.Dropout(p=para_dropout_prob))

            if self.archi_type == "plain":
                # dimension reduction
                dim1_out = int(0.5 * para_dim_in)

            elif self.archi_type == "wide":
                # keep the dimensionality
                dim1_out = int(para_dim_in)

            elif self.archi_type == "residual":
                # dimension reduction
                dim1_out = int(0.5 * para_dim_in)

            elif self.archi_type == "residual_with_input":
                # dimension reduction
                dim1_out = int(0.5 * para_dim_in)
            elif self.archi_type == "residual_with_input_wide":
                dim1_out = int(para_dim_in * 2)

            dense1.append(nn.Linear(para_dim_in, dim1_out))
            # torch.nn.init.xavier_normal_(dense1[-1].weight)
            dense1.append(nn.ReLU())
            self.dense1 = nn.Sequential(*dense1)

            dense2 = nn.ModuleList()
            dense2.append(nn.Dropout(p=para_dropout_prob))
            # dimension reduction
            dim2_out = int(0.5 * dim1_out)
            dense2.append(nn.Linear(dim1_out, dim2_out))
            # torch.nn.init.xavier_normal_(dense2[-1].weight)
            dense2.append(nn.ReLU())
            self.dense2 = nn.Sequential(*dense2)

        # --

        if self.archi_type == "identity":
            self.dim_out = int(para_dim_in)

        elif self.archi_type == "plain":
            self.dim_out = dim2_out

        elif self.archi_type == "wide":
            self.dim_out = dim2_out

        elif self.archi_type == "residual":
            # to concate the outputs of dense layers
            self.dim_out = dim1_out + dim2_out

        elif self.archi_type == "residual_with_input":
            # to concate the outputs of the dense layers and the input vector (i.e., the original factor vector and the interaction vector)
            self.dim_out = para_dim_in + dim1_out + dim2_out
        elif self.archi_type == "residual_with_input_wide":
            # to concate the outputs of the dense layers and the input vector (i.e., the original factor vector and the interaction vector)
            self.dim_out = para_dim_in + dim1_out + dim2_out

    def get_out_dim(
        self,
    ):
        return self.dim_out

    def forward(
        self,
        x,
    ):
        if self.archi_type == "identity":
            return self.identity(x)

        elif self.archi_type == "plain" or self.archi_type == "wide":
            h0 = x
            h1 = self.dense1(h0)
            h2 = self.dense2(h1)
            # [B h2]
            return h2

        elif self.archi_type == "residual":
            h0 = x
            h1 = self.dense1(h0)
            h2 = self.dense2(h1)
            # [B h1+h2]
            return torch.cat((h1, h2), 1)

        elif (
            self.archi_type == "residual_with_input"
            or self.archi_type == "residual_with_input_wide"
        ):
            h0 = x
            h1 = self.dense1(h0)
            h2 = self.dense2(h1)
            # [B h0+h1+h2]
            return torch.cat((h0, h1, h2), 1)
