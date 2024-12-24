from diffusion.models.model_util import *
from diffusion.models.attention import AttentionBlock


class Sequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, qf):
        for layer in self:
            if isinstance(layer, QFBlock):
                x = layer(x, qf)
            else:
                x = layer(x)
        return x


class QFBlock(nn.Module):
    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            dims=2,
            use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.qf_layers = nn.Sequential(
            nn.Linear(1, 2 * self.out_channels),
            nn.SiLU(),
        )

        if self.out_channels % 32 == 0:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                # zero_module(
                #     nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
                # ),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                # zero_module(
                #     nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
                # ),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
        self.control_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

    # def forward(self, x, qf):
    #     """
    #     Apply the block to a Tensor, conditioned on a timestep embedding.
    #     :param x: an [N x C x ...] Tensor of features.
    #     :param qf: an [N x 1] Tensor of QF
    #     :return: an [N x C x ...] Tensor of outputs.
    #     """
    #     return checkpoint(
    #         self._forward, (x, qf), self.parameters(), self.use_checkpoint
    #     )

    def forward(self, x, qf):
        h = self.in_layers(x)

        qf_emb = self.qf_layers(qf).type(h.dtype)
        h = self.control_conv(h)
        gamma, beta = torch.chunk(qf_emb, 2, dim=1)
        while gamma.ndim < h.ndim:
            gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)
        h = h + (1 + gamma) * h + beta
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class QFEncoderUNet(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        in_channels=4,
        model_channels=320,
        num_res_blocks=2,
        attention_resolutions=(4, 2, 1),
        dropout=0,
        channel_mult=(1, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=False,
        use_new_attention_order=False,
        *args,
        **kwargs
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.input_blocks = nn.ModuleList(
            [
                Sequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
            ]
        )

        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    QFBlock(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(Sequential(*layers))
                self._feature_size += ch
            out_ch = ch
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    Sequential(
                        QFBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
            else:
                self.input_blocks.append(
                    Sequential(
                        QFBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                        )
                        if resblock_updown
                        else nn.Conv2d(ch, out_ch, 3, padding=1)
                    )
                )
            ch = out_ch
            input_block_chans.append(ch)
            ds *= 2
            self._feature_size += ch

        self.middle_block = Sequential(
            QFBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            QFBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
            ),
        )
        input_block_chans.append(ch)
        self._feature_size += ch
        self.input_block_chans = input_block_chans

        self.fea_tran = nn.ModuleList()
        for i, mul in enumerate(channel_mult):
            if i == 0:
                for j in range(num_res_blocks + 1):
                    self.fea_tran.append(zero_module(nn.Conv2d(model_channels * mul, model_channels * mul, 3, padding=1)))
            else:
                self.fea_tran.append(zero_module(nn.Conv2d(model_channels * channel_mult[i - 1], model_channels * channel_mult[i - 1], 3, padding=1)))
                for j in range(num_res_blocks):
                    self.fea_tran.append(
                        zero_module(nn.Conv2d(model_channels * mul, model_channels * mul, 3, padding=1)))
        self.fea_tran.append(zero_module(nn.Conv2d(model_channels * channel_mult[-1], model_channels * channel_mult[-1], 3, padding=1)))


    def forward(self, x, qf):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param qf: a 1-D batch of QF.
        :return: an [N x K] Tensor of outputs.
        """
        results = []
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            last_h = h
            h = module(h, qf)
            if i:
                results.append(last_h)
        h = self.middle_block(h, qf)
        results.append(h)

        assert len(results) == len(self.fea_tran)

        for i in range(len(results)):
            results[i] = self.fea_tran[i](results[i])
        return results


if __name__ == '__main__':
    x = torch.rand((1, 32, 64, 96))
    qf = torch.rand((1, 1))
    net = QFEncoderUNet(32, 32, num_res_blocks=2, attention_resolutions=[4, 2, 1], channel_mult=[1, 2, 3, 4])
    # net = QFBlock(32, 0)
    # loss = net(x, qf).mean()
    results = net(x, qf)
    loss = 0
    for res in results:
        loss += res.mean()

    loss.backward()
