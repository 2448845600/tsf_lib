model_conf = dict(
    model_name='iTransformer',
    dataset_name='ETTh1',
    exp_type="long_time_series_forecasting",
    
    output_attention=False,
    d_model=512,
    d_ff=512,
    dropout=0.1,
    factor=3,
    n_heads=8,
    activation='gelu',
    e_layers=2,
)