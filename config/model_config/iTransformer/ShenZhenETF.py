model_conf = dict(
    model_name="iTransformer",
    dataset_name="ShenZhenETF",
    exp_type="traffic_forecasting",

    output_attention=False,
    d_model=512,
    d_ff=512,
    dropout=0.1,
    factor=3,
    n_heads=8,
    activation='gelu',
    e_layers=2,

    lr=0.0005
)