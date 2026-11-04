import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    # input related
    parser.add_argument("--version", type=str, default="mimic3-50",
                        choices=["mimic2", "mimic3", "mimic3-50"], help="Dataset version.")
    parser.add_argument("--label_truncate_length", type=int, default=30)
    
    parser.add_argument("--n_gpu", type=int, default=1)
    
    # word encoder
    parser.add_argument("--word", action="store_true")
    parser.add_argument("--word_embedding_path", type=str)
    parser.add_argument("--word_dim", type=int, default=100)
    parser.add_argument("--word_dp", type=float, default=0.2)
    parser.add_argument("--word_frz", action="store_true")
    
    # combiner
    parser.add_argument("--combiner", type=str, default='lstm',
                    choices=['naive', 'lstm', 'reformer', 'transformer'])
    
    # lstm encoder
    parser.add_argument("--lstm_dp", type=float, default=0.1)
    parser.add_argument("--rnn_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    
    # reformer
    parser.add_argument("--reformer_path", type=str, default="")
    parser.add_argument("--reformer_head", type=int, default=8)
    parser.add_argument("--n_hashes", type=int, default=8)
    parser.add_argument("--local_attention_head", type=int, default=2)
    parser.add_argument("--pkm_layers", type=str, default="")
    # Upstream comments this out, yet train_utils.py still reads args.transformer_dp,
    # so --combiner reformer raises AttributeError. Restored here.
    parser.add_argument("--transformer_dp", type=float, default=0.1)
    
    # Transformer text encoder: 4 layers x 512, defaults set for this task
    # (the upstream block is commented out in the original).
    parser.add_argument("--num_transformer_layers", type=int, default=4)
    parser.add_argument("--transformer_head", type=int, default=8,
                        help="attention heads (upstream default 1 is meaningless for a Transformer; set to 8)")
    parser.add_argument("--transformer_ff", type=int, default=2048,
                        help="feed-forward width (upstream 1024; set to 2048 = 4*h)")
    parser.add_argument("--transformer_activation", type=str, default="gelu",
                        choices=["gelu", "relu"])
    parser.add_argument("--norm_first", action="store_true",
                        help="use pre-LN (norm_first=True) instead of post-LN")
    parser.add_argument("--no_zero_pad_output", action="store_true",
                        help="Do NOT zero the encoder output at padded positions. Default is to zero them, "
                             "matching BiLSTM pad_packed_sequence semantics "
                             "(LabelEncoder's max-pool does not mask)")
    
    # transformer
    parser.add_argument("--pos_embed", type=str, 
                        choices=["none", "learn", "axial", "tri"], default="none")
    parser.add_argument("--axial", type=str, default='(64,64)')
    parser.add_argument("--layer_norm", action="store_true")
    
    # label related
    parser.add_argument("--label_num_layers", type=int, default=0)
    parser.add_argument("--label_dropout", type=float, default=0.0)
    parser.add_argument("--label_pooling", type=str, default='max',
                        choices=['max', 'mean', 'last'])
    parser.add_argument("--est_cls", type=int, default=0)
    
    parser.add_argument("--term_count", type=int, default=1)
    parser.add_argument("--sort_method", type=str, default='random',
                        choices=['max', 'mean', 'random'])
    
    # decoder related
    #                                                       "double", "mil", 'relulaat'], default="caml")
    parser.add_argument("--decoder", type=str, choices=['MultiLabelMultiHeadLAATV2'])
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--code_embedding_path", type=str)
    parser.add_argument("--rep_dropout", type=float, default=0.2)
    parser.add_argument("--att_dropout", type=float, default=0.0)
    parser.add_argument("--xavier", action="store_true")
    parser.add_argument("--attention_head", type=int, default=4) # For MultiHeadLAAT
    
    parser.add_argument("--head_pooling", type=str, default="max", choices=["max", "concat"])
    parser.add_argument("--act_fn_name", type=str, default="tanh", choices=['tanh', 'relu'])

    # Decoder attention implementation: mathematically equivalent, differs only in speed/memory
    parser.add_argument("--attn_impl", type=str, default="naive",
                        choices=["naive", "mem_efficient", "varlen"],
                        help="naive: materialised-softmax baseline; "
                             "mem_efficient: SDPA efficient backend (padded+mask, fp32 or bf16); "
                             "varlen: concatenated representation + flash-attn variable-length "
                             "kernel (requires --bf16 and switches the whole path to ragged)")
    parser.add_argument("--varlen", action="store_true",
                        help="Represent variable-length sequences as ragged buffers (cu_seqlens) instead "
                             "of padded ones, which is what makes Flash Attention usable. "
                             "Same as --attn_impl varlen; applies to encoder, LabelEncoder "
                             "and decoder. Requires --bf16.")
    parser.add_argument("--label_repr", type=str, default=None,
                        choices=["padded", "varlen", "bucketed"],
                        help="Encoding of the label-description branch (None: derived from --varlen/--attn_impl).\n"
                             "  padded  : pad to the longest (5.34x token redundancy, needs a mask)\n"
                             "  varlen  : ragged + flash-attn (no waste, half precision only)\n"
                             "  bucketed: group by exact length + dense SDPA (no waste, runs in fp32)")
    parser.add_argument("--enc_attn_backend", type=str, default="efficient",
                        choices=["auto", "efficient", "flash", "math"],
                        help="SDPA backend for the dense encoder path.\n"
                             "  'auto' silently selects the flash kernel under bf16 when no mask "
                             "is given, so it is not a non-flash baseline; use 'efficient' for that.\n"
                             "  On the bucketed shape, efficient is also faster (560 vs 677 ms) and "
                             "smaller (2.64 vs 3.30 GB).")
    parser.add_argument("--enc_chunk_tokens", type=int, default=16384,
                        help="Max tokens per encoder chunk (0: no chunking).\n"
                             "Chunking with one checkpoint per chunk drops label-branch "
                             "forward+backward from OOM to ~3.2 GB at ~1.21x the time.")
    parser.add_argument("--dec_chunk_codes", type=int, default=0,
                        help="Max codes per decoder chunk (0: no chunking).\n"
                             "The naive fp32 decoder needs 24.12 GB at B=8; chunking brings it "
                             "to 4.22 GB for +3.6% time.")
    parser.add_argument("--attn_vchunk", type=int, default=256,
                        help="Variant 1c: zero-pad q/k to the head width, split v into att_dim/vchunk blocks")

    # ── Hybrid: the two branches may use different encoders ──────────────────
    # Rationale: flash is 6.28x faster on the text branch (L=4000) but about 2x slower on
    # label descriptions (<=30 words). 'same' reproduces the original behaviour exactly.
    parser.add_argument("--label_combiner", type=str, default="same",
                        choices=["same", "lstm"],
                        help="Encoder used by the label-description branch.\n"
                             "  same: shares the text branch's TextEncoder (default, one copy of "
                             "the weights)\n"
                             "  lstm: an independent BiLSTM for the label branch while the text "
                             "branch keeps --combiner (shared embeddings, different combiner)")
    parser.add_argument("--label_rnn_dim", type=int, default=256,
                        help="Output width of the label-branch BiLSTM when --label_combiner lstm (default 256)")
    parser.add_argument("--label_lstm_layers", type=int, default=2,
                        help="Number of label-branch BiLSTM layers when --label_combiner lstm (default 2)")
    parser.add_argument("--label_lstm_dp", type=float, default=0.1,
                        help="Dropout of the label-branch BiLSTM when --label_combiner lstm (only if layers > 1)")

    # ── Precision ablation: pin one branch to fp32 ──────────────────────────
    # These two switches isolate which side of the model bf16 damages.
    parser.add_argument("--text_fp32", action="store_true",
                        help="Force the text branch to fp32 while the rest follows --bf16; incompatible with --attn_impl varlen")
    parser.add_argument("--label_fp32", action="store_true",
                        help="Force the label branch to fp32 while the rest follows --bf16; incompatible with --label_repr varlen")

    # ── Global step budget (exact truncation for probes) ────────────────────
    # --max_steps compares the within-epoch index, so a value larger than steps/epoch never
    # fires; truncating a probe at k full epochs needs this switch instead.
    parser.add_argument("--total_max_steps", type=int, default=0,
                        help="Global step budget across epochs (0: unlimited). Set it to a multiple of "
                             "steps/epoch to run exactly k full epochs; the current epoch's "
                             "evaluation and logging still complete before stopping.")
    
    # Precision
    parser.add_argument("--bf16", action="store_true",
                        help="Train in bf16 mixed precision. Flash Attention ships only fp16/bf16 kernels, "
                             "so --attn_impl varlen requires --bf16")
    
    # Train setting
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["AdamW", "SGD", "Adam"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_epoch", type=int, default=20)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip the end-of-epoch dev/test evaluation (useful for smoke tests; evaluation is slow)")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save a checkpoint every N epochs (default: every epoch)")
    parser.add_argument("--keep_epochs", type=int, default=10,
                        help="Keep the weights of the last N epochs, plus best_epoch.pth.\n"
                             "One .pth is about 241 MB: keeping 10 is 2.66 GB per cell.")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Print the loss every N steps (0: never)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Cap on steps per epoch (0: unlimited), useful for a quick smoke test")
    parser.add_argument("--early_stop_epoch", type=int, default=-1)
    parser.add_argument("--early_stop_metric", type=str, default="prec_at_8")
    parser.add_argument("--truncate_length", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--output_base_dir", type=str, default="./output/")
    parser.add_argument("--prob_threshold", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=['linear', 'constant', 'cosine'])

    # loss
    parser.add_argument("--loss_name", type=str, default="ce",
                        choices=['ce', 'focal', 'asy', 'kexue', 'ldam', 'maskce', 'hungarian'])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--asy_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asy_gamma_pos", type=float, default=1.0)
    parser.add_argument("--asy_clip", type=float, default=0.05)
    parser.add_argument("--able_torch_grad_focal_loss", action="store_true")
    parser.add_argument("--ldam_c", type=float, default=3.0)
    
    parser.add_argument("--rdrop_alpha", type=float, default=0.0)
   
    parser.add_argument("--main_code_loss_weight", type=float, default=1.0)
    parser.add_argument("--code_loss_weight", type=float, default=1.0)


    
    # knowledge_distill
    parser.add_argument("--knowledge_distill", action="store_true") # not avaliable
    
    # # others
    parser.add_argument("--adv_training", default=None, choices=['fgm', 'pgd'])
    
    parser.add_argument("--tag", type=str, default="")

    return parser
