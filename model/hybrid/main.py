import warnings
import os as _os
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import time
import shutil
import torch
if _os.environ.get("ICD_ANOMALY"):
    torch.autograd.set_detect_anomaly(True)
from torch.nn.parallel import DistributedDataParallel
from data_util import MimicFullDataset, my_collate_fn

# Training statistics shared between train_one_epoch (writes) and the epoch loop (reads).
_TRAIN_STATS = {"n": 0, "ms": 0.0, "bad_streak": 0, "total": 0, "stop": False}
from tqdm import tqdm
import contextlib
import json
import numpy as np
from evaluation import all_metrics, print_metrics
from torch.utils.data import DataLoader
from train_parser import generate_parser
from train_utils import generate_output_folder_name, generate_model
from find_threshold import find_threshold_micro
from accelerate import DistributedDataParallelKwargs, Accelerator


def run(args):
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]

    # --- Precision switches ------------------------------------------------
    # Flash Attention has fp16/bf16 kernels only (forcing the flash backend in fp32 raises
    # "No available kernel"), so these flags must be consistent.
    attn_impl = getattr(args, "attn_impl", "naive")
    use_bf16 = getattr(args, "bf16", False)
    if getattr(args, "varlen", False):
        attn_impl = "varlen"
        if not use_bf16:
            raise SystemExit(
                "[config] --varlen requires --bf16: Flash Attention has fp16/bf16 kernels only.\n"
                "         For a memory-efficient attention path in fp32, use --attn_impl mem_efficient.")
        if args.combiner != "transformer":
            raise SystemExit(
                "[config] --varlen is only supported with --combiner transformer, "
                "because the ragged representation is produced by the encoder.")
    if attn_impl == "varlen" and not use_bf16:
        raise SystemExit(
            "[config] --attn_impl varlen requires --bf16: flash-attn has half-precision kernels only.")
    # Label-branch representation: bucketed uses dense SDPA and does not require bf16.
    lrep = getattr(args, "label_repr", None)
    if lrep in ("varlen", "bucketed"):
        if args.combiner != "transformer":
            raise SystemExit(
                "[config] --label_repr %s is only supported with --combiner transformer "
                "(ragged/bucketed representations come from VarlenEncoderStack)." % lrep)
        if lrep == "varlen" and not use_bf16:
            raise SystemExit(
                "[config] --label_repr varlen requires --bf16: flash-attn has half-precision "
                "kernels only. For a memory-efficient fp32 path, use --label_repr bucketed.")

    # Per-branch precision flags are incompatible with flash, which has no fp32 kernel.
    if getattr(args, "text_fp32", False) and (attn_impl == "varlen" or getattr(args, "varlen", False)):
        raise SystemExit(
            "[config] --text_fp32 is incompatible with --attn_impl varlen: flash-attn has "
            "fp16/bf16 kernels only, so the text branch cannot run in fp32 there.")
    if getattr(args, "label_fp32", False) and getattr(args, "label_repr", None) == "varlen":
        raise SystemExit(
            "[config] --label_fp32 is incompatible with --label_repr varlen: to run the label "
            "branch in fp32, use --label_repr bucketed (or padded).")

    mixed_precision = "bf16" if use_bf16 else "no"
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers,
                              mixed_precision=mixed_precision)
    accelerator.print(f"[precision] mixed_precision={mixed_precision} | attn_impl={attn_impl}")
    
    output_basename = generate_output_folder_name(args)
    accelerator.print(output_basename)
    output_path = os.path.join(args.output_base_dir, output_basename)

    try:
        os.system(f"mkdir -p {output_path}")
    except BaseException:
        pass
   
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    word_embedding_path = args.word_embedding_path         
    accelerator.print(f"Use word embedding from {word_embedding_path}")
    
    train_dataset = MimicFullDataset(args.version, "train", word_embedding_path, args.truncate_length, args.label_truncate_length, args.term_count, args.sort_method)
    dev_dataset = MimicFullDataset(args.version, "dev", word_embedding_path, args.truncate_length)
    test_dataset = MimicFullDataset(args.version, "test", word_embedding_path, args.truncate_length)

    if args.knowledge_distill:
        raise NotImplementedError
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=8, pin_memory=True)
    
    model = generate_model(args, train_dataset).to(accelerator.device)
    accelerator.print(model)
    optimizer, scheduler_step = model.configure_optimizers(train_dataloader)
    optimizer = optimizer[0]
    scheduler_step = scheduler_step[0]
    
    # prepare label input feature
    model.c_input_word = train_dataset.c_input_word.to(accelerator.device)
    model.c_word_mask = train_dataset.c_word_mask.to(accelerator.device)
    model.c_word_sent = train_dataset.c_word_sent.to(accelerator.device)
    
    model, optimizer, train_dataloader = \
        accelerator.prepare(model, optimizer, train_dataloader)

    # adversarial_training
    if args.adv_training:
        raise NotImplementedError
    
    steps = 0
    best_dev_metric = {}
    best_test_metric = {}
    early_stop_count = 0
    best_epoch_idx = 0
   
    if accelerator.is_local_main_process and args.debug:
        dev_metric, _, _ = eval_func(model, dev_dataloader, args.device, args.prob_threshold, True, args)
        print_metrics(dev_metric, 'DEBUG')

    for epoch_idx in range(1, args.train_epoch + 1):
        _epoch_t0 = time.time()
        _TRAIN_STATS["n"] = 0
        _TRAIN_STATS["ms"] = 0.0
        epoch_dev_metric, epoch_test_metric, steps = train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler_step, args, accelerator)
        accelerator.wait_for_everyone()
        if _TRAIN_STATS.get("stop"):
            # --total_max_steps reached: finish this epoch's evaluation and logging, then stop
            accelerator.print("[total_max_steps] global step budget reached, stopping after epoch %d" % epoch_idx)
        if accelerator.is_local_main_process:
            # One .pth is ~241 MB, so keeping every epoch of every cell does not fit.
            # Keep the last _KEEP epoch weights. best_epoch.pth is written separately,
            # the moment dev improves (see the early-stop block below), because pruning
            # here would otherwise delete a best epoch from early in training before
            # the run ends.
            _se = max(1, int(getattr(args, 'save_every', 1) or 1))
            _KEEP = max(1, int(getattr(args, 'keep_epochs', 10) or 10))
            if epoch_idx % _se == 0 or epoch_idx == args.train_epoch:
                accelerator.save(accelerator.unwrap_model(model),
                                 os.path.join(output_path, f"epoch{epoch_idx}.pth"))
                import glob as _glob
                _ckpts = sorted(_glob.glob(os.path.join(output_path, 'epoch*.pth')),
                                key=lambda q: int(''.join(ch for ch in os.path.basename(q)
                                                          if ch.isdigit()) or 0))
                for _old in _ckpts[:-_KEEP]:        # keep only the last _KEEP
                    try:
                        os.remove(_old)
                    except OSError:
                        pass
            # With --skip_eval the metrics dict is empty, so printing is skipped.
            if epoch_dev_metric:
                print_metrics(epoch_dev_metric, 'Dev_Epoch' + str(epoch_idx))
                print_metrics(epoch_dev_metric, 'Dev_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))
                print_metrics(epoch_test_metric, 'Test_Epoch' + str(epoch_idx))
                print_metrics(epoch_test_metric, 'Test_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))

            # ---- Structured record, one JSON line per epoch (machine-readable) ----
            try:
                _rec = {
                    'epoch': epoch_idx,
                    'train_loss': _TRAIN_STATS.get("loss", 0.0) / max(1, _TRAIN_STATS["n"]),
                    'train_kl': _TRAIN_STATS.get("kl", 0.0) / max(1, _TRAIN_STATS["n"]),
                    'train_c_loss': _TRAIN_STATS.get("c_loss", 0.0) / max(1, _TRAIN_STATS["n"]),
                    'steps': _TRAIN_STATS["n"],
                    'step_ms_mean': _TRAIN_STATS["ms"] / max(1, _TRAIN_STATS["n"]),
                    'peak_mem_GB': torch.cuda.max_memory_allocated() / 1024 ** 3,
                    'lr': (scheduler_step.get_last_lr()[0] if hasattr(scheduler_step, 'get_last_lr') else None),
                    'epoch_seconds': time.time() - _epoch_t0,
                    'dev': epoch_dev_metric or {},
                    'test': epoch_test_metric or {},
                }
                with open(os.path.join(output_path, 'epochs.jsonl'), 'a', encoding='utf-8') as _f:
                    _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
            except Exception as _e:
                accelerator.print("[warn] failed to write epochs.jsonl: %s" % _e)
    
        # Early Stop
        _improved = False
        if not best_dev_metric:
            best_dev_metric = epoch_dev_metric
            best_test_metric = epoch_test_metric
            best_epoch_idx = epoch_idx
            _improved = True
        else:
            if args.early_stop_metric in epoch_dev_metric:
                if epoch_dev_metric[args.early_stop_metric] >= best_dev_metric[args.early_stop_metric]:
                    best_dev_metric = epoch_dev_metric
                    best_test_metric = epoch_test_metric
                    best_epoch_idx = epoch_idx
                    early_stop_count = 0
                    _improved = True
                else:
                    early_stop_count += 1

        # Copy the best weights the moment dev improves. Doing it only at the end is
        # not enough: the pruning above drops epoch{N}.pth from earlier rounds, so a
        # best epoch in the first (train_epoch - keep_epochs) rounds is already gone
        # by then and the copy fails silently.
        if _improved and accelerator.is_local_main_process:
            _cur = os.path.join(output_path, "epoch%d.pth" % epoch_idx)
            if os.path.exists(_cur):
                shutil.copy2(_cur, os.path.join(output_path, "best_epoch.pth"))

        if args.early_stop_epoch > 0 and early_stop_count >= args.early_stop_epoch:
            accelerator.print(f"Early Stop at Epoch {epoch_idx}, \
                    metric {args.early_stop_metric} not improve on dev set for {early_stop_count} epoch.")
            break

        # --total_max_steps reached: this epoch has fully finished, so leave the loop
        if _TRAIN_STATS.get("stop"):
            break
        
    if accelerator.is_local_main_process and not getattr(args, 'skip_eval', False):
        best_train_metric, _, _ = eval_func(model, train_dataloader, accelerator.device, args.prob_threshold, True, args)
        print_metrics(best_train_metric, 'Best_Train_Epoch' + str(best_epoch_idx))
        print_metrics(best_train_metric, 'Best_Train_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        print_metrics(best_dev_metric, 'Best_Dev_Epoch' + str(best_epoch_idx))
        print_metrics(best_dev_metric, 'Best_Dev_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        print_metrics(best_test_metric, 'Best_Test_Epoch' + str(best_epoch_idx))
        print_metrics(best_test_metric, 'Best_Test_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        best_path = os.path.join(output_path, "epoch%d.pth" % best_epoch_idx)
        new_path = os.path.join(output_path, "best_epoch.pth")
        if os.path.exists(best_path):
            shutil.copy2(best_path, new_path)
        elif not os.path.exists(new_path):
            accelerator.print(
                "[error] best_epoch.pth is missing: epoch%d.pth was pruned by "
                "--keep_epochs and no earlier copy exists. Predictions will fall "
                "back to the last epoch, which is not the dev-selected one."
                % best_epoch_idx)

    
def train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, accelerator=None):
    model.train()
    epoch_loss = 0.
    epoch_kl_loss = 0.
    epoch_c_loss = 0.
    
        
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", ascii=True, disable=not accelerator.is_local_main_process)
    step_ms = 0.0
    for batch_idx, batch in enumerate(epoch_iterator):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.time()
        batch_gpu = tuple([x.to(accelerator.device) for x in batch])
        ori_loss = model(batch_gpu, rdrop=args.rdrop_alpha > 0.0)
        if isinstance(ori_loss, dict):
            loss = ori_loss['loss']
        else:
            loss = ori_loss
            
        batch_loss = float(loss.item())
        # Abort the cell after 3 consecutive non-finite losses instead of training on.
        if batch_loss != batch_loss or batch_loss in (float('inf'), float('-inf')):
            _TRAIN_STATS["bad_streak"] += 1
            accelerator.print("[FATAL] step=%d non-finite loss (%s), consecutive #%d"
                              % (batch_idx, batch_loss, _TRAIN_STATS["bad_streak"]))
            if _TRAIN_STATS["bad_streak"] >= 3:
                raise SystemExit(3)
        else:
            _TRAIN_STATS["bad_streak"] = 0
        epoch_loss += batch_loss
        
        batch_c_loss = float(ori_loss['c_loss'].item())
        epoch_c_loss += batch_c_loss
        
        if args.rdrop_alpha > 0.0:
            batch_kl_loss = float(ori_loss['kl_loss'].item())
            epoch_kl_loss += batch_kl_loss
                
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        accelerator.backward(loss)

        # ICD_NAN_DEBUG=1 checks every gradient for non-finite values (off by default).
        if os.environ.get("ICD_NAN_DEBUG"):
            _bad, _sq = [], 0.0
            for _n, _p in model.named_parameters():
                if _p.grad is None:
                    continue
                _g = _p.grad.detach().float()
                _sq += float(_g.norm() ** 2)
                if not torch.isfinite(_g).all():
                    _bad.append((_n, float(_g.abs().max())))
            accelerator.print(
                "[nan-debug] step=%d loss=%.4f kl=%.4f c=%.4f | grad_norm=%.4e | "
                "non-finite gradient tensors: %d %s"
                % (batch_idx, float(loss), float(ori_loss.get('kl_loss', 0.0))
                   if isinstance(ori_loss, dict) else 0.0,
                   float(ori_loss['c_loss']) if isinstance(ori_loss, dict) else 0.0,
                   _sq ** 0.5, len(_bad), _bad[:6]))
            

        if args.rdrop_alpha > 0.0:
            epoch_iterator.set_description("Epoch: %0.4f/%0.4f/%0.4f, Batch: %0.4f/%0.4f/%0.4f" % \
                                       (epoch_loss / (batch_idx + 1), epoch_kl_loss/(batch_idx + 1), epoch_c_loss/(batch_idx + 1), \
                                        batch_loss, batch_kl_loss, batch_c_loss)
                                      )
        else:
            epoch_iterator.set_description("Epoch: %0.4f, Batch: %0.4f" % (epoch_loss / (batch_idx + 1), batch_loss))

        if (steps + 1) % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(
                 model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            model.zero_grad()

        steps += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_ms = (time.time() - _t0) * 1e3
        _TRAIN_STATS["n"] += 1
        _TRAIN_STATS["ms"] += step_ms

        # Periodic loss logging (--log_every, default 50).
        _le = int(getattr(args, 'log_every', 50) or 0)
        if _le and (batch_idx + 1) % _le == 0:
            accelerator.print(
                "[loss] step=%d | step %.4f | epoch mean %.4f | kl %.4f | c %.4f | %.0f ms"
                % (batch_idx + 1, batch_loss, epoch_loss / (batch_idx + 1),
                   (epoch_kl_loss / (batch_idx + 1)) if args.rdrop_alpha > 0.0 else 0.0,
                   epoch_c_loss / (batch_idx + 1), step_ms))

        # --max_steps: stop the epoch after N steps (0: unlimited), for smoke tests.
        if getattr(args, 'max_steps', 0) and (batch_idx + 1) >= args.max_steps:
            accelerator.print(
                "[max_steps] reached %d steps, ending this epoch | peak allocated=%.2f GB "
                "reserved=%.2f GB | step %.1f ms"
                % (args.max_steps,
                   torch.cuda.max_memory_allocated() / 1024 ** 3,
                   torch.cuda.max_memory_reserved() / 1024 ** 3,
                   step_ms))
            break

        # --total_max_steps is a budget across epochs (0: unlimited). --max_steps cannot
        # serve this purpose: it compares the within-epoch index, so a value larger than
        # steps/epoch never fires. Truncating a probe at k full epochs needs this switch.
        _TRAIN_STATS["total"] += 1
        _tms = int(getattr(args, 'total_max_steps', 0) or 0)
        if _tms and _TRAIN_STATS["total"] >= _tms:
            accelerator.print(
                "[total_max_steps] global budget of %d steps reached (step %d this epoch), "
                "ending training"
                % (_tms, batch_idx + 1))
            _TRAIN_STATS["stop"] = True
            break

    if getattr(args, 'skip_eval', False):
        accelerator.wait_for_everyone()
        accelerator.print("[skip_eval] dev/test evaluation skipped")
        return {}, {}, steps

    tqdm_bar = False
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        dev_metric, _, threshold = eval_func(model, dev_dataloader, accelerator.device, None, tqdm_bar, args)
        print('Threshold find on dev:', threshold)
        test_metric, _, _ = eval_func(model, test_dataloader, accelerator.device, threshold, tqdm_bar, args)
    else:
        dev_metric = None
        test_metric = None
    _TRAIN_STATS["loss"] = epoch_loss
    _TRAIN_STATS["kl"] = epoch_kl_loss if args.rdrop_alpha > 0.0 else 0.0
    _TRAIN_STATS["c_loss"] = epoch_c_loss
    return dev_metric, test_metric, steps

def predict(model, dataloader, device, threshold=None, tqdm_bar=None, args=None):
    model.eval()
    outputs = []
    device = args.device if args is not None else device
    it = tqdm(dataloader) if tqdm_bar else dataloader
    # accelerate's autocast covers model.forward only; evaluation goes through
    # model.predict(), which is outside it. Without the block below, evaluation would run
    # in fp32 -- the varlen path would crash (no fp32 flash kernel) and the others would
    # disagree with training.
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if (args is not None and getattr(args, "bf16", False) and torch.cuda.is_available())
           else contextlib.nullcontext())
    with torch.no_grad(), amp:
        if isinstance(model, DistributedDataParallel):
            model.module.calculate_label_hidden()
        else:
            model.calculate_label_hidden()
        for batch in it:
            batch_gpu = tuple([x.to(device) for x in batch])
            if isinstance(model, DistributedDataParallel):
                now_res = model.module.predict(batch_gpu, threshold)
            else:
                now_res = model.predict(batch_gpu, threshold)
            # bf16 tensors cannot be passed to numpy; cast to fp32 first.
            outputs.append({key: (value.float() if value.is_floating_point() else value)
                            .cpu().detach() for key, value in now_res.items()})
            
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy()
    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
    return yhat, y, yhat_raw

def eval_func(model, dataloader, device, threshold=None, tqdm_bar=False, args=None):
    yhat, y, yhat_raw = predict(model, dataloader, device, threshold, tqdm_bar, args)
    if threshold is None:
        threshold = find_threshold_micro(yhat_raw, y)
    yhat = np.where(yhat_raw > threshold, 1, 0)
    metric = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)
    return metric, (yhat, y, yhat_raw), threshold

def main():
    parser = generate_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
