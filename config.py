

@attr.s
class Configuration:
    # model related parameters
    arch_sparse_feature_size = attr.ib(2)
    arch_embedding_size = attr.ib("4-3-2")
    arch_mlp_bot = attr.ib("4-3-2")
    arch_mlp_top = attr.ib("4-2-1")
    arch_interaction_op = attr.ib("dot")
    arch_interaction_itself = attr.ib(False)
    weighted_pooling = attr.ib(None)
    # embedding table options
    md_flag = attr.ib(False)
    md_threshold = attr.ib(200)
    md_temperature = attr.ib(0.3)
    md_round_dims = attr.ib(False)
    qr_flag = attr.ib(False)
    qr_threshold = attr.ib(200)
    qr_operation = attr.ib("mult")
    qr_collisions = attr.ib(4)
    # activations and loss
    activation_function = attr.ib("relu")
    loss_function = attr.ib("mse")  # or bce or wbce
    loss_weights = attr.ib("1.0-1.0")
    # for wbce
    loss_threshold = attr.ib(0.0)  # 1.0e-7
    round_targets = attr.ib(False)
    # data
    data_size = attr.ib(1)
    num_batches = attr.ib(0)
    data_generation = attr.ib("random")
    rand_data_dist = attr.ib("uniform")
    rand_data_min = attr.ib(0)
    rand_data_max = attr.ib(1)
    rand_data_mu = attr.ib(-1)
    rand_data_sigma = attr.ib(1)
    data_trace_file = attr.ib("./input/dist_emb_j.log")
    data_set", type=str,
                        default="kaggle")  # or terabyte
    raw_data_file", type=str, default="")
    processed_data-file", type=str, default="")
    data_randomize", type=str,
                        default="total")  # or day or none
    data_trace_enable_padding",
                        type=bool, default=False)
    max_ind_range", type=int, default=-1)
    data_sub_sample_rate",
                        type=float, default=0.0)  # in [0, 1]
    num_indices_per_lookup", type=int, default=10)
    num_indices_per_lookup_fixed",
                        type=bool, default=False)
    num_workers", type=int, default=0)
    memory_map", action="store_true", default=False)
    # training
    mini_batch_size", type=int, default=1)
    nepochs", type=int, default=1)
    learning_rate", type=float, default=0.01)
    print_precision", type=int, default=5)
    ("--numpy-rand-seed", type=int, default=123)
    ("--sync-dense-params", type=bool, default=True)
    ("--optimizer", type=str, default="sgd")
    (
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    ("--inference-only", action="store_true", default=False)
    # quantize
    ("--quantize-mlp-with-bit", type=int, default=32)
    ("--quantize-emb-with-bit", type=int, default=32)
    # onnx
    ("--save-onnx", action="store_true", default=False)
    # gpu
    ("--use-gpu", action="store_true", default=False)
    # distributed
    ("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    ("--print-freq", type=int, default=1)
    ("--test-freq", type=int, default=-1)
    ("--test-mini-batch-size", type=int, default=-1)
    ("--test-num-workers", type=int, default=-1)
    ("--print-time", action="store_true", default=False)
    ("--print-wall-time",
                        action="store_true", default=False)
    ("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling",
                        action="store_true", default=False)
    parser.add_argument("--plot-compute-graph",
                        action="store_true", default=False)
    parser.add_argument("--tensor-board-filename",
                        type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader",
                        action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle",
                        action="store_true", default=False)
    # mlperf gradient accumulation iterations
    parser.add_argument("--mlperf-grad-accum-iter", type=int, default=1)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
