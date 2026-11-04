import numpy as np
from model.icd_model import IcdModel


def resolve_reprs(args):
    """Resolve the label-branch and text-branch representations, which may differ.

    They must be resolved separately: one TextEncoder serves both branches, but
      * the label branch (35,684 descriptions) is the memory wall, so it uses bucketed or
        varlen (no padding waste);
      * the text branch (B x 4000) must match what the decoder expects: padded for
        attn_impl naive/mem_efficient, ragged for varlen.

    --varlen keeps its old meaning: one switch that makes both branches ragged.
    """
    attn_impl = getattr(args, 'attn_impl', 'naive')
    varlen = bool(getattr(args, 'varlen', False))
    text_repr = 'varlen' if (varlen or attn_impl == 'varlen') else 'padded'
    label_repr = getattr(args, 'label_repr', None) or text_repr
    args.text_repr_final = text_repr
    args.label_repr_final = label_repr
    # Whether the combiner needs ragged support (VarlenEncoderStack rather than nn.TransformerEncoder)
    args.needs_ragged = (text_repr == 'varlen' or label_repr != 'padded')
    return args


def create_all_config(args, train_dataset):
    resolve_reprs(args)
    word_config = create_word_config(args, train_dataset)
    combine_config = create_combine_config(args)
    decoder_config = create_decoder_config(args, train_dataset)
    label_config = create_label_config(args, train_dataset)
    loss_config = create_loss_config(args, train_dataset)
    return word_config, combine_config, decoder_config, label_config, loss_config

def short_name(path):
    return path.split('/')[-1]
    

def short(x):
    if x.startswith("["):
        return x[1:-1]
    return x

def generate_output_folder_name(args):
    word_lst = ['word', args.word_dp]
    if args.word_frz:
        word_lst.append('frz')

    combine_lst = [args.combiner]
    if args.combiner == "lstm":
        combine_lst.extend([args.num_layers, args.rnn_dim, args.lstm_dp])
    if args.combiner == "reformer":
        combine_lst.extend([args.num_layers, args.rnn_dim, args.reformer_head, args.n_hashes, args.local_attention_head, args.transformer_dp])
    if args.combiner == "transformer":
        combine_lst.extend([args.num_transformer_layers, args.rnn_dim,
                            args.transformer_head, args.transformer_ff,
                            args.transformer_activation])
        if args.norm_first:
            combine_lst.append('preln')
        if args.no_zero_pad_output:
            combine_lst.append('nopadzero')
        
    if args.pos_embed != "none" and args.combiner in ["rac", "transformer", "fastformer", "reformer"]:
        combine_lst.append(args.pos_embed)
    if args.pos_embed == "axial" and args.combiner in ["rac", "transformer", "fastformer", "reformer"]:
        combine_lst.append(str(args.axial)[1:-1])

    if args.layer_norm:
        combine_lst.append('ln')
    
    decoder_lst = [args.decoder, args.attention_dim]
    if args.xavier:
        decoder_lst.append('xav')

    if args.decoder in ["MultiLabelMultiHeadLAATV2"]:
        decoder_lst.extend([args.rep_dropout, args.attention_head])
        # Attention implementation and precision must be in the run name, or cells overwrite each other
        if getattr(args, 'varlen', False):
            decoder_lst.append('varlen')
        elif getattr(args, 'attn_impl', 'naive') != "naive":
            decoder_lst.append(args.attn_impl)
        if getattr(args, 'bf16', False):
            decoder_lst.append('bf16')
        if args.att_dropout > 0.0:
            decoder_lst.append(args.att_dropout)
        if args.decoder in ["MultiLabelMultiHeadLAATV2"]:
            if args.head_pooling != "max":
                decoder_lst.append(args.head_pooling)
            if args.decoder in ["MultiLabelMultiHeadLAATV2"]:
                if args.act_fn_name != "tanh":
                    decoder_lst.append(args.act_fn_name)
    
    # Representation and chunking must be in the run name for the same reason
    if getattr(args, 'label_repr_final', 'padded') != 'padded':
        decoder_lst.append('lrep-' + args.label_repr_final)
    if int(getattr(args, 'enc_chunk_tokens', 0) or 0) > 0:
        decoder_lst.append('eck' + str(args.enc_chunk_tokens))
    if getattr(args, 'enc_attn_backend', 'efficient') != 'efficient':
        decoder_lst.append('eab-' + args.enc_attn_backend)
    if int(getattr(args, 'dec_chunk_codes', 0) or 0) > 0:
        decoder_lst.append('dck' + str(args.dec_chunk_codes))
    # The hybrid combiner must be in the run name, or it collides with the shared-encoder cell
    if str(getattr(args, 'label_combiner', 'same') or 'same') != 'same':
        decoder_lst.append('lenc-%s%d-%d' % (args.label_combiner,
                                             args.label_rnn_dim,
                                             args.label_lstm_layers))
    # Per-branch precision flags must be in the run name, or the two probes collide
    if getattr(args, 'text_fp32', False):
        decoder_lst.append('tfp32')
    if getattr(args, 'label_fp32', False):
        decoder_lst.append('lfp32')

    label_lst = [args.label_pooling]
    if args.label_num_layers > 0:
        label_lst.append(args.label_num_layers)
    if args.label_dropout > 0:
        label_lst.append(args.label_dropout)
    label_lst.append(f'est{args.est_cls}')
        
    if args.term_count > 1:
        label_lst.extend([args.term_count, args.sort_method])
        
    loss_lst = [args.loss_name, args.main_code_loss_weight, args.code_loss_weight]

    bsz = args.batch_size * args.gradient_accumulation_steps * args.n_gpu
    train_lst = [f'bsz{bsz}', args.optimizer, args.train_epoch, args.truncate_length, f'warm{args.warmup_ratio}', f'wd{args.weight_decay}']
    if args.optimizer in ["Adam", "SGD", "AdamW"]:
        train_lst.append(args.learning_rate)
    if args.rdrop_alpha > 0.0:
        train_lst.append(f"rdrop{args.rdrop_alpha}")
    if args.scheduler != "linear":
        train_lst.append(args.scheduler)
                  
    all_lst = [[args.version], word_lst, combine_lst, decoder_lst, label_lst, loss_lst, train_lst]
    folder_name = "_".join(["-".join([str(y) for y in x]) for x in all_lst if x])
    if args.debug:
        folder_name = "debug_" + folder_name
    if args.tag:
        folder_name = folder_name + "-" + args.tag
        
    return folder_name
                  
def create_word_config(args, train_dataset):
    word_config = {}
    
    try:
        padding_idx = train_dataset.word2id['**PAD**']
    except BaseException:
        padding_idx = None
    word_config['padding_idx'] = padding_idx
    word_config['count'] = len(train_dataset.word2id)
    word_config['dropout'] = args.word_dp
    word_config['word_embedding_path'] = args.word_embedding_path
    word_config['dim'] = args.word_dim
    word_config['frz'] = args.word_frz
    
    return word_config


def _effective_attn_impl(args):
    """--varlen is the master switch: when set, the decoder must take the varlen path to
    match the encoder's ragged output."""
    impl = getattr(args, 'attn_impl', 'naive')
    if getattr(args, 'varlen', False):
        return 'varlen'
    return impl


def create_combine_config(args):
    combine_config = {}
    combine_config['input_dim'] = args.word_dim
    
    combine_config['model'] = args.combiner
    if args.combiner == "lstm":
        combine_config['lstm_dropout'] = args.lstm_dp
        combine_config['rnn_dim'] = args.rnn_dim
        combine_config['num_layers'] = args.num_layers
        if args.num_layers <= 1:
            combine_config['lstm_dropout'] = 0.0
    if args.combiner == "reformer":
        combine_config['rnn_dim'] = args.rnn_dim
        combine_config['num_layers'] = args.num_layers
        reformer_config = {'reformer_head':args.reformer_head,
                           'n_hashes':args.n_hashes,
                           'local_attention_head':args.local_attention_head,
                           'pkm_layers':()}
        if args.pkm_layers:
            reformer_config['pkm_layers'] = tuple([int(n) for n in args.pkm_layers.split(',')])
        combine_config.update(reformer_config)
        combine_config['pos_embed'] = args.pos_embed
        if args.pos_embed == "axial":
            combine_config['axial'] = args.axial
        combine_config['transformer_dropout'] = args.transformer_dp
    if args.combiner == "transformer":
        # Ragged support is needed whenever either branch is not padded
        combine_config['varlen'] = bool(getattr(args, 'needs_ragged', False))
        combine_config['enc_repr'] = getattr(args, 'label_repr_final', 'padded')
        combine_config['enc_chunk_tokens'] = int(getattr(args, 'enc_chunk_tokens', 0) or 0)
        combine_config['enc_attn_backend'] = getattr(args, 'enc_attn_backend', 'efficient')
        combine_config['rnn_dim'] = args.rnn_dim
        combine_config['num_layers'] = args.num_transformer_layers
        combine_config['transformer_head'] = args.transformer_head
        combine_config['transformer_ff'] = args.transformer_ff
        combine_config['transformer_activation'] = args.transformer_activation
        combine_config['transformer_dropout'] = args.transformer_dp
        combine_config['norm_first'] = args.norm_first
        combine_config['zero_pad_output'] = not args.no_zero_pad_output
        combine_config['pos_embed'] = args.pos_embed
        if args.pos_embed == "axial":
            combine_config['axial'] = args.axial

    combine_config['layer_norm'] = args.layer_norm
    combine_config['dim'] = args.attention_dim

    return combine_config

def create_decoder_config(args, train_dataset):
    decoder_config = {}
    decoder_config['model'] = args.decoder
    decoder_config['input_dim'] = args.attention_dim
    decoder_config['attention_dim'] = args.attention_dim
    decoder_config['label_count'] = train_dataset.code_count
    decoder_config['code_embedding_path'] = args.code_embedding_path
    decoder_config['ind2c'] = train_dataset.ind2c
    decoder_config['ind2mc'] = train_dataset.ind2mc
    decoder_config['xavier'] = args.xavier
    decoder_config['est_cls'] = args.est_cls
    decoder_config['att_dropout'] = args.att_dropout

    if args.decoder in ["MultiLabelMultiHeadLAATV2"]:
        decoder_config['attention_head'] = args.attention_head
        decoder_config['rep_dropout'] = args.rep_dropout
        # Attention implementation: mathematically equivalent, affects speed and memory only
        decoder_config['attn_impl'] = _effective_attn_impl(args)
        decoder_config['attn_vchunk'] = getattr(args, 'attn_vchunk', 256)
        # Chunk along the code axis, one checkpoint per chunk (0: no chunking)
        decoder_config['dec_chunk_codes'] = int(getattr(args, 'dec_chunk_codes', 0) or 0)
        
        if args.decoder in ["MultiLabelMultiHeadLAATV2"]:
            decoder_config['head_pooling'] = args.head_pooling
            if args.decoder in ["MultiLabelMultiHeadLAATV2"]:
                decoder_config['act_fn_name'] = args.act_fn_name
        
    return decoder_config


def create_label_config(args, train_dataset):
    label_config = {}
    # Must be attention_dim, not rnn_dim: the combiner always outputs combine_config['dim']
    # (= args.attention_dim; reduce_linear widens when rnn_dim differs). Writing rnn_dim here
    # was a latent bug that only surfaced when rnn_dim != attention_dim and
    # --label_num_layers > 0, which no cell we ran satisfied. The hybrid combiner makes the
    # label branch's rnn_dim differ from the text branch's, so it has to be right.
    label_config['input_dim'] = args.attention_dim
    label_config['num_layers'] = args.label_num_layers
    label_config['dropout'] = args.label_dropout
    label_config['pooling'] = args.label_pooling
    # The label encoder's input shape must follow what the label branch actually emits, not
    # --label_repr alone: --label_combiner lstm forces that branch to padded (see
    # create_label_combine_config), so a ragged label encoder would be handed a padded
    # tensor. Only the shared branch honours --label_repr.
    _label_combiner = str(getattr(args, 'label_combiner', 'same') or 'same')
    label_config['varlen'] = (_label_combiner != 'lstm'
                              and getattr(args, 'label_repr_final', 'padded') != 'padded')
    return label_config


def create_loss_config(args, train_dataset):
    if args.loss_name == "ce":
        loss_dict = {'name':'ce'}
    if args.loss_name == "focal":
        loss_dict = {'name':'focal', 'gamma':args.focal_gamma, 'alpha':args.focal_alpha}

    if args.loss_name == "asy":
        if args.able_torch_grad_focal_loss:
            disable = False
        else:
            disable = True
        loss_dict = {'name':'asy', 'gamma_neg':args.asy_gamma_neg, 'gamma_pos':args.asy_gamma_pos,
                     'clip':args.asy_clip, 'disable_torch_grad_focal_loss':disable}
    if args.loss_name == "ldam":
        loss_dict = {'name':'ldam', 'ldam_c':args.ldam_c}
        total_label_count = np.array([0] * train_dataset.code_count)
        for i in range(len(train_dataset)):
            label = np.array(train_dataset[i][2])
            total_label_count += label
        loss_dict['label_count'] = total_label_count

    loss_dict['rdrop_alpha'] = args.rdrop_alpha

    loss_dict['main_code_loss_weight'] = args.main_code_loss_weight
    loss_dict['code_loss_weight'] = args.code_loss_weight

    return loss_dict

def generate_model(args, train_dataset):
    word_config, combine_config, decoder_config, label_config, loss_config = \
        create_all_config(args, train_dataset)
    if args.term_count > 1:
        assert args.decoder.startswith("MultiLabelMultiHeadLAAT")
        assert args.term_count == args.attention_head

    model = IcdModel(word_config, combine_config,
                     decoder_config, label_config, loss_config, args) 
    return model
