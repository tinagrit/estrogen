from __future__ import annotations

from functools import lru_cache
import re

from rdkit import Chem
import selfies as sf
import torch


MODEL_ID = "ncfrey/ChemGPT-4.7M"


@lru_cache(maxsize=1)
def load_generator():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    cls_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        bos_token_id=cls_token_id,
        eos_token_id=sep_token_id,
        pad_token_id=sep_token_id,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        device_map="auto",
    )
    model.generation_config.bos_token_id = cls_token_id
    model.generation_config.eos_token_id = sep_token_id
    model.generation_config.pad_token_id = sep_token_id
    model.eval()
    return tokenizer, model


def _close_smiles_prefix(prefix: str) -> str:
    if Chem.MolFromSmiles(prefix) is not None:
        return prefix

    parentheses = 0
    bracket_open = False
    ring_counts: dict[str, int] = {}
    index = 0
    while index < len(prefix):
        character = prefix[index]
        if character == "[":
            bracket_open = True
        elif character == "]":
            bracket_open = False
        elif not bracket_open:
            if character == "(":
                parentheses += 1
            elif character == ")":
                parentheses -= 1
                if parentheses < 0:
                    break
            elif character == "%" and index + 2 < len(prefix) and prefix[index + 1:index + 3].isdigit():
                ring = prefix[index:index + 3]
                ring_counts[ring] = ring_counts.get(ring, 0) + 1
                index += 2
            elif character.isdigit():
                ring_counts[character] = ring_counts.get(character, 0) + 1
        index += 1
    else:
        bracket_suffix = "]" if bracket_open else ""
        ring_suffix = "".join(ring for ring, count in ring_counts.items() if count % 2)
        close_suffix = ring_suffix + ")" * parentheses
        fillers = ("", "C", "N", "O", "c", "n", "cc", "CC")
        for filler in fillers:
            candidate = prefix + bracket_suffix + filler + close_suffix
            if Chem.MolFromSmiles(candidate) is not None:
                return candidate

    raise ValueError("This SMILES prefix cannot be closed into a valid molecule.")


def _encode_prefix(prefix: str, tokenizer) -> tuple[str, list[int]]:
    completed_smiles = _close_smiles_prefix(prefix)
    selfies = sf.encoder(completed_smiles)
    token_ids = tokenizer(selfies, add_special_tokens=False)["input_ids"]
    if not token_ids or tokenizer.unk_token_id in token_ids:
        raise ValueError("The SMILES prefix contains tokens unsupported by ChemGPT.")
    return selfies, token_ids


def generate_smiles(
    seed: str,
    num_return_sequences: int = 10,
    temperature: float = 0.8,
    top_k: int = 50,
    max_length: int = 80,
) -> list[str]:
    """Complete a user SMILES prefix through ChemGPT's SELFIES tokenizer."""
    tokenizer, model = load_generator()
    prefix_selfies, prefix_token_ids = _encode_prefix(seed, tokenizer)
    cls_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    input_ids = torch.tensor([[cls_token_id, *prefix_token_ids]], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    token_limit = input_ids.shape[1] + max(1, int(max_length))
    try:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=max(float(temperature), 0.05),
            top_k=max(1, int(top_k)),
            max_new_tokens=max(1, int(max_length)),
            min_new_tokens=1,
            num_return_sequences=max(1, min(int(num_return_sequences), 50)),
            bos_token_id=cls_token_id,
            eos_token_id=sep_token_id,
            pad_token_id=tokenizer.convert_tokens_to_ids("[PAD]"),
        )
    except Exception as error:
        raise RuntimeError(f"ChemGPT generation failed: {error}") from error

    results: list[str] = []

    for sequence in output:
        # Let the tokenizer reconstruct WordPiece fragments instead of
        # concatenating raw tokens containing "##".
        generated_selfies = tokenizer.decode(
            sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).replace(" ", "")

        if not generated_selfies:
            continue

        try:
            completed_smiles = sf.decoder(generated_selfies)
            molecule = Chem.MolFromSmiles(completed_smiles)
            if molecule is None:
                continue

            smiles = Chem.MolToSmiles(molecule, canonical=True)
            if smiles not in results:
                results.append(smiles)
        except (ValueError, sf.DecoderError):
            continue

    return results