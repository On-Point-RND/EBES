import glob
from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import (
    ReconPredictor,
)
from ...types import Batch
from ...model import agg
from ...model import BaseModel
from ..preprocess import Batch2Seq
from ...model.seq2seq import GRU
from copy import deepcopy
from ...model import build_model, FrozenModel


class GenModel(BaseModel):
    def __init__(
        self,
        # Preprocess:
        cat_cardinalities,
        num_features,
        cat_emb_dim=16,
        num_emb_dim=16,
        time_process: Literal["cat", "diff", "none"] = "cat",
        num_norm=True,
        # Encoder:
        enc_hidden_size=128,  # get from contrastive model
        enc_num_layers=1,
        enc_aggregation="TakeLastHidden",
        # Decoder:
        dec_hidden_size=128,
        dec_num_layers=3,
        dec_num_heads=8,
        dec_scale_hidden=2,
        max_len=1000,
        # Loss weights:
        mse_weight=1,
        ce_weight=1,
        l1_weight=0.001,
        reconstruction_weight=1,
        contrastive_weight=10,
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.ce_weight = ce_weight
        self.l1_weight = l1_weight
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        ### PROCESSORS ###
        self.processor = Batch2Seq(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            cat_emb_dim=cat_emb_dim,
            num_emb_dim=num_emb_dim,
            time_process=time_process,
            num_norm=num_norm,
        )
        self.input_size = self.processor.output_dim
        ### NORMS ###
        self.post_encoder_norm = nn.LayerNorm(enc_hidden_size)
        self.decoder_norm = nn.LayerNorm(dec_hidden_size)

        ### ENCODER ###
        self.encoder = GRU(
            input_size=self.processor.output_dim,
            hidden_size=enc_hidden_size,
            num_layers=enc_num_layers,
        )
        self.enc_aggregation = getattr(agg, enc_aggregation)()

        ### HIDDEN TO X0 PROJECTION ###
        self.hidden_to_x0 = nn.Linear(enc_hidden_size, self.input_size)

        ### DECODER ###
        self.dec_pos_encoding = nn.Embedding(max_len + 1, dec_hidden_size)
        self.decoder = TransformerDecoder(
            d_model=dec_hidden_size,
            nhead=dec_num_heads,
            num_layers=dec_num_layers,
            norm=self.decoder_norm,
            dim_feedforward=dec_scale_hidden * dec_hidden_size,
        )
        self.decoder_proj = nn.Linear(self.input_size, dec_hidden_size)

        ### ACTIVATION ###
        self.act = nn.GELU()

        ### LOSS ###
        self.recon_predictor = ReconPredictor(
            dec_hidden_size,
            cat_cardinalities,
            num_features,
        )

    def reconstruction_loss(self, batch: Batch):
        """
        output: Dict that is outputed from forward method
        """
        output = self.reconstruct(batch)
        ce_loss, mse_loss = self.recon_predictor.loss(output["prediction"], batch)
        total_ce_loss = sum([value for _, value in ce_loss.items()])
        total_mse_loss = sum([value for _, value in mse_loss.items()])

        ### SPARCE EMBEDDINGS ###
        sparce_loss = torch.mean(torch.sum(torch.abs(output["latent"]), dim=0))

        ### GENERATED EMBEDDINGS DISTANCE ###

        losses_dict = {
            "total_mse_loss": total_mse_loss,
            "total_CE_loss": total_ce_loss,
            "sparcity_loss": sparce_loss,
        }

        total_loss = (
            self.mse_weight * total_mse_loss
            + self.ce_weight * total_ce_loss
            + self.l1_weight * sparce_loss
        )
        losses_dict["reconstruction_loss"] = total_loss

        return losses_dict, output

    def reconstruct(self, batch: Batch):
        global_hidden = self.encode(batch)
        pred = self.decode(batch, global_hidden)
        res_dict = {
            "prediction": pred,
            "latent": global_hidden,
        }
        return res_dict

    def encode(self, batch: Batch):
        x = self.processor(batch)
        all_hid = self.encoder(x)
        global_hidden = self.post_encoder_norm(self.enc_aggregation(all_hid))
        return global_hidden

    def decode(self, batch: Batch, global_hidden):
        x = self.processor(batch).tokens
        x0 = self.hidden_to_x0(global_hidden)
        x = torch.cat([x0.unsqueeze(0), x], dim=0)

        x_proj = self.decoder_proj(self.act(x))
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x.size(0), device=x.device
        )
        x_proj = x_proj + self.dec_pos_encoding(
            torch.arange(x_proj.size(0), device=x_proj.device)
        ).unsqueeze(1)
        # print(x_proj.size(), global_hidden.size())
        dec_out = self.decoder(
            tgt=x_proj,
            memory=x_proj[0, :, :].unsqueeze(
                0
            ),  # we can not pass global hidden due to dimension mismatch
            tgt_mask=mask,
        )

        out = dec_out[:-1, :, :]

        pred = self.recon_predictor(out)
        return pred


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, norm, dim_feedforward):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=norm,
        )

    def forward(self, tgt, memory, tgt_mask):
        return self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)


class MLEMPretrainer(GenModel):
    def __init__(
        self, contr_model_folder: str, normalize_z: bool = False, *args, **kwargs
    ):
        contr_model_config = OmegaConf.load(Path(contr_model_folder) / "config.yaml")[
            "unsupervised_model"
        ]  # type: ignore
        kwargs["enc_hidden_size"] = contr_model_config["encoder"]["params"][
            "hidden_size"
        ]
        super().__init__(*args, **kwargs)
        self.contrastive_model = build_model(contr_model_config)
        contr_model_checkpoint = Path(contr_model_folder) / "pretrain/ckpt"
        ckpts = list(glob.glob(f"{contr_model_checkpoint}/*.ckpt"))
        if len(ckpts) != 1:
            raise ValueError("Not 1 checkpoint in folder")
        self.contrastive_model.load_state_dict(
            torch.load(ckpts[0], map_location="cpu")["model"]
        )
        self.contrastive_model = FrozenModel(self.contrastive_model)

        self.normalize_z = normalize_z
        init_temp = torch.tensor(10.0)
        init_bias = torch.tensor(-10.0)
        self.sigmoid_temp = nn.Parameter(torch.log(init_temp))
        self.bias = nn.Parameter(init_bias)
        print("Pretrain success")

    @property
    def temp(self):
        return torch.exp(self.sigmoid_temp)

    def sigmoid_loss(self, latent, batch: Batch):
        # https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py
        # https://arxiv.org/abs/2303.15343
        with torch.no_grad():
            contrastive_output = self.contrastive_model(batch).detach()

        if not self.normalize_z:
            z_recon = latent
            z_contrastive = contrastive_output
        else:
            z_recon = F.normalize(latent)
            z_contrastive = F.normalize(contrastive_output)

        logits = (z_recon @ z_contrastive.T) * self.temp + self.bias
        m1_diag1 = -torch.ones_like(logits) + 2 * torch.eye(logits.size(0)).to(
            logits.device
        )
        loglik = F.logsigmoid(m1_diag1 * logits)
        nll = -torch.sum(loglik, dim=-1)
        return nll.mean()

    def forward(self, batch: Batch):
        check_batch = deepcopy(batch)
        losses, output = self.reconstruction_loss(batch)
        assert batch == check_batch
        sigmoid_loss = self.sigmoid_loss(output["latent"], batch)
        losses["loss"] = (
            self.reconstruction_weight * losses["reconstruction_loss"]
            + self.contrastive_weight * sigmoid_loss
        )
        return losses


class MLEMEncoder(MLEMPretrainer):
    @property
    def output_dim(self):
        return self.encoder.output_dim

    def forward(self, batch: Batch):
        return self.encode(batch)
