import torch
from typing import List, Any
from lsrr.interfaces import DataSample

def process_batch_with_oom_recovery(
    extractor: Any,
    writer: Any,
    samples_chunk: List[DataSample],
    position_rule: str = "last_token"
):
    """Processes a chunk of samples for hidden state extraction.
    If a CUDA OutOfMemoryError occurs, clears GPU cache and recursively splits the chunk into smaller sub-batches.
    If a single sample triggers OOM on GPU, temporarily falls back to CPU for that sample.
    """
    if not samples_chunk:
        return

    try:
        questions = [s.question for s in samples_chunk]
        answers = [s.answer for s in samples_chunk]

        enc_q = extractor.tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        enc_ans = extractor.tokenizer(
            answers,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            H_batch = extractor.extract_hidden_states(
                input_ids=enc_q["input_ids"],
                attention_mask=enc_q["attention_mask"],
                position_rule=position_rule
            )  # [B, L, d]

        for b in range(len(samples_chunk)):
            h_sample = H_batch[b]  # [L, d]
            ans_ids = enc_ans["input_ids"][b]
            ans_mask = enc_ans["attention_mask"][b]
            clean_ans_ids = ans_ids[ans_mask == 1]
            # Record the gold answer alongside the sample meta. Evaluation reads it from
            # here; without it the scorer silently compares against an empty string.
            sample_meta = dict(samples_chunk[b].meta or {})
            sample_meta["answer"] = samples_chunk[b].answer
            writer.add_sample(h_sample, target_ids=clean_ans_ids, meta=sample_meta)

    except Exception as e:
        err_msg = str(e).lower()
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in err_msg
        if is_oom:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(samples_chunk) > 1:
                mid = len(samples_chunk) // 2
                print(f"\n[OOM Recovery] CUDA OOM on batch size {len(samples_chunk)}. Dynamically splitting into sub-batches ({mid} + {len(samples_chunk)-mid})...")
                process_batch_with_oom_recovery(extractor, writer, samples_chunk[:mid], position_rule)
                process_batch_with_oom_recovery(extractor, writer, samples_chunk[mid:], position_rule)
            else:
                print(f"\n[OOM Recovery] Single sample OOM (Question len: {len(samples_chunk[0].question)} chars). Falling back to CPU for this sample...")
                orig_device = extractor.device
                try:
                    extractor.model.to("cpu")
                    extractor.device = torch.device("cpu")
                    process_batch_with_oom_recovery(extractor, writer, samples_chunk, position_rule)
                finally:
                    extractor.model.to(orig_device)
                    extractor.device = orig_device
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            raise e
