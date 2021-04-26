from typing import Any, List, Dict, Iterator, Callable, Set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import top_k_top_p_filtering
from .beam_hypotheses import BeamHypotheses

import logging

logger = logging.getLogger(__name__)

# pylint:disable=no-member


def penalize_repetition(next_token_logits, sampled_token_sequences, repetition_penalty):
    """repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)"""
    #TODO: Fix bug in this function
    if repetition_penalty != 1.0:
        for i in range(next_token_logits.shape[0]):
            for previous_token in set(sampled_token_sequences[i]):
                # if score < 0 then repetition penalty has to be multiplied to reduce the previous
                # token probability
                if next_token_logits[i, previous_token] < 0:
                    next_token_logits[i, previous_token] *= repetition_penalty
                else:
                    next_token_logits[i, previous_token] /= repetition_penalty
    return next_token_logits


class TransformerDecoder:
    """
    Modularized Design for Transformer Autoregresive Decoding
    """
    def __init__(self, config):
        self.setup_config(config)

    def setup_config(self, decode_config):
        self.decode_config = decode_config

        decode_config.max_steps = decode_config.max_steps if decode_config.max_steps is not None else 20
        decode_config.do_sample = decode_config.do_sample if decode_config.do_sample is not None else True
        decode_config.num_beams = decode_config.num_beams if decode_config.num_beams is not None else 1
        decode_config.early_stopping = decode_config.early_stopping if decode_config.early_stopping is not None else True
        decode_config.temperature = decode_config.temperature if decode_config.temperature is not None else 1.0
        decode_config.top_k = decode_config.top_k if decode_config.top_k is not None else -1
        decode_config.top_p = decode_config.top_p if decode_config.top_p is not None else 0.9
        decode_config.repetition_penalty = decode_config.repetition_penalty if decode_config.repetition_penalty is not None else 1.0
        decode_config.length_penalty = decode_config.length_penalty if decode_config.length_penalty is not None else 1.0

        decode_config.num_return_sequences = decode_config.num_return_sequences if decode_config.num_return_sequences is not None else 1

        decode_config.bos_token_ids = decode_config.bos_token_ids if decode_config.bos_token_ids is not None else None
        decode_config.eos_token_ids = decode_config.eos_token_ids if decode_config.eos_token_ids is not None else [-1]

        decode_config.output_log_probs = decode_config.output_log_probs if decode_config.output_log_probs is not None else False

        for key, value in decode_config.items():
            setattr(self, key, value)

    def register_generator(self, model):
        self._generator = model

    def register_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def prepare_model_inputs_for_generation(
        self,
        input_ids: torch.Tensor = None,
        model_inputs: Dict[str, torch.Tensor] = None,
        num_return_sequences: int = 1
    ) -> Dict[str, torch.Tensor]:
        """Overrides this function whenever necessary.
           This function should output a key word args as the inputs for the generator
        """
        model_inputs = {}

        if input_ids is None:
            self.bos_token_ids = torch.LongTensor(self.bos_token_ids)
            model_inputs["input_ids"] = self.bos_token_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
            model_inputs["past"] = None
        else:
            model_inputs["input_ids"] = input_ids
            model_inputs["past"] = None
        return model_inputs

    def compute_logits(
        self,
        timestep: int,
        model_inputs: Dict[str, torch.Tensor],
        generated_token_sequences: Dict[int, List[int]] = None,
        generated_log_prob_sequences: Dict[int, List[float]] = None,
    ) -> Dict[str, torch.Tensor]:

        # generate next token
        logits, past = self._generator(**model_inputs)
        logits = logits[:, -1, :]

        if self.output_log_probs:
            raw_log_probs = torch.log_softmax(logits, dim=-1)
        else:
            raw_log_probs = None

        logits = penalize_repetition(logits, generated_token_sequences, self.repetition_penalty)

        # Temperature (higher temperature => more likely to sample low probability tokens)
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Prepare for the next batch
        model_inputs["past"] = past
        model_inputs = self.increment_model_inputs(model_inputs)

        return logits, raw_log_probs

    def sample_next_token(
        self,
        timestep: int,
        logits: torch.Tensor,
        model_inputs: Dict[str, torch.Tensor],
        generated_token_sequences: Dict[int, List[int]] = None,
        generated_log_prob_sequences: Dict[int, List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Sample the next token
        if self.do_sample:
            logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p, min_tokens_to_keep=2)
            # Sample
            # TODO: Test numpy.random.multinomial
            probs = torch.softmax(logits, -1)
            predicted_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            predicted_token = torch.argmax(logits, dim=-1).unsqueeze(1)

        # Autoregressive Generation
        model_inputs["input_ids"] = predicted_token

        return predicted_token

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor = None, model_inputs: Dict[str, torch.Tensor] = None, **kwargs):
        """ Generates sequences for models with a LM head. The method currently supports greedy or penalized greedy decoding, 
            nucleus sampling and beam-search.
        Args:
            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.
            max_steps: (`optional`) int
                The max steps of the sequence to be generated.  Between 1 and infinity. Default to 20.
            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Default to greedy sampling.
            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.
            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictely positive. Default to 1.0.
            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.
            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.
            bos_token_ids: (`optional`) int
                Beginning of sentence token if no prompt is provided. Default to 0.
            eos_token_ids: (`optional`) int or list of int
                End of sequence token or list of tokens to stop the generation. Default to 0.
            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.
            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.
        """
        # Pass temp configs
        for key, value in self.decode_config.items():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, value)


        assert len(input_ids.shape) == 2
        batch_size = input_ids.shape[0]
        
        if self.eos_token_ids is not None:
            self.eos_token_ids = torch.LongTensor(self.eos_token_ids)
        if self.bos_token_ids is not None:
            self.bos_token_ids = torch.LongTensor(self.bos_token_ids)

        # current position and vocab size
        seq_len = input_ids.shape[1]

        # Initialize history state if it is not initialized
        model_inputs = self.prepare_model_inputs_for_generation(input_ids, model_inputs, self.num_return_sequences)

        # Effective batch size
        # We don't handle the num_return_sequences here
        # It should be done in prepare_model_inputs_for_generation
        if self.num_return_sequences != 1 and self.num_beams <=1:
            #TODO: currently we don't support `num_return_sequences` for beam search
            self.batch_size = batch_size * self.num_return_sequences
        else:
            self.batch_size = batch_size

        if self.num_beams <= 1:
            results = self._generate_no_beam_search(model_inputs)
        else:
            results = self._generate_beam_search(model_inputs)
        return results

    def _generate_no_beam_search(self, model_inputs: Dict[str, torch.Tensor]):
        """
        High Performance Generation via Dynamic Batching 
        """
        # record the index of each sequence for pop out
        current_batch_indices = torch.arange(self.batch_size)
        generated_token_sequences = torch.ones((self.batch_size, self.max_steps), dtype=torch.long) * -1
        generated_log_prob_sequences = torch.zeros((self.batch_size, self.max_steps), dtype=torch.float)

        if self.bos_token_ids is not None:
            start_position = len(self.bos_token_ids)
            generated_token_sequences[:, :start_position] = self.bos_token_ids
        else:
            start_position = 0

        # Main Decoding loop
        for timestep in range(start_position, self.max_steps):
            logits, raw_log_probs = self.compute_logits(
                timestep=timestep,
                model_inputs=model_inputs,
                generated_token_sequences=generated_token_sequences,
                generated_log_prob_sequences=generated_log_prob_sequences,
            )

            predicted_tokens = self.sample_next_token(
                timestep=timestep,
                logits=logits,
                model_inputs=model_inputs,
                generated_token_sequences=generated_token_sequences[current_batch_indices],
                generated_log_prob_sequences=generated_log_prob_sequences[current_batch_indices],
            )

            # Collect the predicted token
            generated_token_sequences[current_batch_indices, timestep] = predicted_tokens.squeeze(1).cpu()

            # Collect Log Probs
            if self.output_log_probs:
                raw_log_probs = raw_log_probs.gather(-1, predicted_tokens)
                generated_log_prob_sequences[current_batch_indices, timestep] = raw_log_probs.squeeze(1).cpu()

            # Pop finished sequences
            sequence_poped, keeped_batch_indices, current_batch_indices = self.pop_finished_sequences(
                timestep, current_batch_indices, generated_token_sequences
            )

            # Filter poped sequences from history state
            if sequence_poped:
                model_inputs = self.filter_finished_model_inputs(model_inputs, keeped_batch_indices)

            # Check if every sequence is done
            if len(current_batch_indices) == 0:
                break

        final_generated_sequences = []
        for batch_idx in range(0, generated_token_sequences.shape[0], self.num_return_sequences):
            batch_tokens = []
            for return_idx in range(self.num_return_sequences):
                token_sequence = generated_token_sequences[batch_idx + return_idx]
                token_sequence = token_sequence[token_sequence != -1]
                batch_tokens.append(token_sequence)
            final_generated_sequences.append(batch_tokens)

        results = {"tokens": final_generated_sequences}

        if self.output_log_probs:
            results["log_probs"] = []
            for batch_idx in range(0, generated_token_sequences.shape[0], self.num_return_sequences):
                batch_log_probs = []
                for return_idx in range(self.num_return_sequences):
                    log_prob_sequence = generated_log_prob_sequences[batch_idx + return_idx]
                    log_prob_sequence = log_prob_sequence[:results["tokens"][batch_idx][return_idx].shape[0]]
                    batch_log_probs.append(log_prob_sequence)
                results["log_probs"].append(batch_log_probs)

        return results

    def _generate_beam_search(self, model_inputs: Dict[str, torch.Tensor]):
        """
        High Performance Generation via Dynamic Batching 
        """
        # Where to store all the beams
        generated_hyps = [
            BeamHypotheses(self.num_beams, self.max_steps, self.length_penalty, early_stopping=self.early_stopping)
            for _ in range(self.batch_size)
        ]

        # Buffer for unfinished beams
        beam_token_sequences_buffer = torch.ones((self.batch_size, self.num_beams, self.max_steps), dtype=int) * -1
        beam_log_prob_sequences_buffer = torch.zeros((self.batch_size, self.num_beams, self.max_steps))

        # Buffer to track scores for unfinished beams
        beam_log_prob_scores_1d = torch.zeros((self.batch_size * self.num_beams, 1))
        done_sequences = [False for _ in range(self.batch_size)]
        eos_token_ids = torch.LongTensor(self.eos_token_ids)
        eos_token_len = len(self.eos_token_ids)

        if self.bos_token_ids is not None:
            start_position = len(self.bos_token_ids)
            beam_token_sequences_buffer[:, :, :start_position] = self.bos_token_ids
        else:
            start_position = 0

        logits, raw_log_probs = self.compute_logits(
            timestep=0,
            model_inputs=model_inputs,
            generated_token_sequences=beam_token_sequences_buffer,
            generated_log_prob_sequences=beam_log_prob_sequences_buffer,
        )

        vocab_size = logits.shape[-1]

        # Sample the next token
        if self.do_sample:
            # Top-p/top-k filtering, the scores here might be different from log_probs
            logits = top_k_top_p_filtering(
                logits, top_k=self.top_k, top_p=self.top_p, min_tokens_to_keep=self.num_beams
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            # Sample
            # TODO: Test numpy.random.multinomial
            predicted_tokens = torch.multinomial(probs, num_samples=self.num_beams, replacement=False)
            # gather predicted tokens' log_probs
            log_probs = torch.gather(log_probs, -1, predicted_tokens)  # (batch_size, num_beams * 2)
            # sort the sampled vector to make sure that the first num_beams samples are the best
            log_probs, log_probs_indices = torch.sort(log_probs, descending=True, dim=1)
            predicted_tokens = torch.gather(predicted_tokens, -1, log_probs_indices)
        else:
            # Greedy Beam Search
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs, predicted_tokens = torch.topk(log_probs, self.num_beams, dim=-1)

        # assign predicted token
        beam_token_sequences_buffer[:, :, start_position] = predicted_tokens.cpu()
        # assign predicted log probs
        if self.output_log_probs:
            beam_log_prob_sequences_buffer[:, :, start_position] = torch.gather(raw_log_probs, -1,
                                                                                predicted_tokens).cpu()

        # set log_prob scores before beam search
        beam_log_prob_scores_1d = log_probs.reshape(self.batch_size * self.num_beams, 1)

        # expand model_inputs
        # so that every shape becomes (batch_size * num_beams)
        model_inputs = self.expand_model_inputs(model_inputs, predicted_tokens, self.num_beams)

        start_position += 1

        for timestep in range(start_position, self.max_steps + 1):
            # Check EOS condition
            if timestep >= eos_token_len:
                check_if_eos = torch.all(
                    beam_token_sequences_buffer[:, :, timestep - eos_token_len:timestep] == eos_token_ids, dim=2
                )

                # for all beams that end with eos
                for batch_idx, beam_idx in (check_if_eos).nonzero():

                    # convert 2d index to 1d
                    beam_idx_1d = batch_idx * self.num_beams + beam_idx
                    # get from buffer
                    beam_token_sequence = beam_token_sequences_buffer[batch_idx, beam_idx]
                    beam_log_prob_sequence = beam_log_prob_sequences_buffer[batch_idx, beam_idx]
                    # clean the sequence
                    beam_log_prob_sequence = beam_log_prob_sequence[beam_token_sequence != -1]
                    beam_token_sequence = beam_token_sequence[beam_token_sequence != -1]
                    # get the cumulative score
                    beam_score = beam_log_prob_scores_1d[beam_idx_1d].item()
                    # add the generated sequence to its corresponding prompt

                    generated_hyps[batch_idx].add(
                        {
                            "tokens": beam_token_sequence.clone(),
                            "log_probs": beam_log_prob_sequence.clone().detach() if self.output_log_probs else None
                        }, beam_score
                    )

                    # avoid sampling from already ended beam any more
                    beam_log_prob_scores_1d[beam_idx_1d] = -1e4

                    # check if batch is finished
                    done_sequences[batch_idx] = done_sequences[batch_idx] or generated_hyps[batch_idx].is_done(
                        beam_log_prob_scores_1d[batch_idx].max().item(), cur_len=timestep
                    )

                # Cannot find enough sequences ended with eos under max_steps
                # So we add all remaining beams to hypothesises
                if timestep == self.max_steps:
                    # shape: (batch_size, num_beams)
                    beam_log_prob_scores = beam_log_prob_scores_1d.reshape(self.batch_size, self.num_beams)
                    # shape: (batch_size, num_beams)
                    beam_indices = torch.argsort(beam_log_prob_scores.cpu(), dim=1, descending=True)

                    # add the unfinished sequences to the generated hypothesis
                    for batch_idx in range(beam_indices.shape[0]):
                        # compute max score in a batch
                        max_score = beam_log_prob_scores_1d[batch_idx].max().item()

                        for beam_idx in range(beam_indices.shape[1]):
                            # convert 2d index to 1d
                            beam_idx_1d = batch_idx * self.num_beams + beam_idx
                            # get from buffer
                            beam_token_sequence = beam_token_sequences_buffer[batch_idx, beam_idx]
                            beam_log_prob_sequence = beam_log_prob_sequences_buffer[batch_idx, beam_idx]
                            # clean the sequence
                            beam_log_prob_sequence = beam_log_prob_sequence[beam_token_sequence != -1]
                            beam_token_sequence = beam_token_sequence[beam_token_sequence != -1]
                            # get the cumulative score
                            beam_score = beam_log_prob_scores_1d[beam_idx_1d].item()
                            # add the generated sequence to its corresponding prompt
                            generated_hyps[batch_idx].add(
                                {
                                    "tokens":
                                        beam_token_sequence.clone(),
                                    "log_probs":
                                        beam_log_prob_sequence.clone().detach() if self.output_log_probs else None
                                }, beam_score
                            )

                            done_sequences[batch_idx] = done_sequences[batch_idx] or generated_hyps[batch_idx].is_done(
                                max_score, cur_len=timestep
                            )
                    # Stop the generation
                    break

                # if all sequences are finished with eos
                if all(done_sequences):
                    break

            logits, raw_log_probs = self.compute_logits(
                timestep=timestep,
                model_inputs=model_inputs,
                generated_token_sequences=beam_token_sequences_buffer,
                generated_log_prob_sequences=beam_log_prob_sequences_buffer,
            )

            # Sample the next token
            if self.do_sample:
                # Top-p/top-k filtering, the scores here might be different from log_probs
                # Must make sure that there are at least num_beams each node to sample from
                # shape: (batch_size * num_beams, vocab_size)
                logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p, min_tokens_to_keep=self.num_beams)

                # shape: (batch_size * num_beams, vocab_size)
                log_probs = torch.log_softmax(logits, dim=-1)

                # compute the current cumulative log probs
                cur_log_prob_scores = (
                    (beam_log_prob_scores_1d + log_probs).reshape(self.batch_size, self.num_beams * vocab_size)
                )

                # shape: (batch_size, num_beams * vocab_size)
                prob_scores = F.softmax(cur_log_prob_scores, dim=-1)

                # Sample
                # TODO: Test numpy.random.multinomial
                predicted_tokens = torch.multinomial(prob_scores, num_samples=self.num_beams, replacement=False)

                # gather predicted tokens' log_probs
                # restore the actual probabilities
                cur_log_prob_scores = torch.gather(cur_log_prob_scores, -1, predicted_tokens)

                # sort the sampled vector to make sure that the first num_beams samples are the best
                cur_log_prob_scores, sorted_indices = torch.sort(cur_log_prob_scores, descending=True, dim=1)
                predicted_tokens = torch.gather(predicted_tokens, -1, sorted_indices)
            else:
                # Greedy Beam Search
                log_probs = torch.log_softmax(logits, dim=-1)

                cur_log_prob_scores = (
                    (beam_log_prob_scores_1d + log_probs).reshape(self.batch_size, self.num_beams * vocab_size)
                )

                cur_log_prob_scores, predicted_tokens = torch.topk(
                    cur_log_prob_scores, self.num_beams, dim=-1, largest=True, sorted=True
                )

            # shape (batch_size, num_beams)
            beam_indices = torch.floor_divide(predicted_tokens, vocab_size).cpu()
            beam_indices_1d = (
                beam_indices + (torch.arange(self.batch_size) * self.num_beams).unsqueeze(1).expand(-1, self.num_beams)
            )
            # shape (batch_size * num_beams)
            beam_indices_1d = beam_indices_1d.reshape(-1)

            # shape (batch_size, num_beams)
            predicted_tokens = torch.fmod(predicted_tokens, vocab_size)

            expanded_beam_indices = beam_indices.unsqueeze(2).expand(-1, -1, self.max_steps)

            # replace old beam_token_sequences_buffer
            beam_token_sequences_buffer = torch.gather(beam_token_sequences_buffer, dim=1, index=expanded_beam_indices)

            # replace old beam_log_prob_sequences_buffer
            beam_log_prob_sequences_buffer = torch.gather(
                beam_log_prob_sequences_buffer, dim=1, index=expanded_beam_indices
            )

            # replace the old beam_log_prob_scores
            beam_log_prob_scores_1d = cur_log_prob_scores.reshape(self.batch_size * self.num_beams, 1)

            # fill in the new predicted token
            beam_token_sequences_buffer[:, :, timestep] = predicted_tokens.cpu()
            # fill in the log_probs
            if self.output_log_probs:
                # be careful about the beam_indices here
                # shape (batch_size * num_beams, 1)
                # torch.gather(raw_log_probs.reshape(1, -1), -1, predicted_tokens)
                raw_log_probs = torch.gather(raw_log_probs[beam_indices_1d], -1, predicted_tokens.reshape(-1, 1))
                # shape (batch_size, num_beams)
                raw_log_probs = raw_log_probs.reshape(self.batch_size, self.num_beams)

                beam_log_prob_sequences_buffer[:, :, timestep] = raw_log_probs.cpu()

            # Reorder `model_inputs`!
            model_inputs = self.reorder_model_inputs(model_inputs, predicted_tokens, beam_indices_1d)

        generated_hyps = [sorted(item.beams, key=lambda beam: beam[0], reverse=True) for item in generated_hyps]
        results = {}
        results["tokens"] = []
        results["beam_scores"] = []

        if self.output_log_probs:
            results["log_probs"] = []

        for batch in generated_hyps:
            batch_tokens = []
            beam_scores = []
            if self.output_log_probs:
                batch_log_probs = []
            for beam in batch:
                batch_tokens.append(beam[1]["tokens"])
                beam_scores.append(beam[0])

                if self.output_log_probs:
                    batch_log_probs.append(beam[1]["log_probs"])

            results["tokens"].append(batch_tokens)
            results["beam_scores"].append(beam_scores)

            if self.output_log_probs:
                results["log_probs"].append(batch_log_probs)

        return results

    def expand_model_inputs(
        self, model_inputs: Dict[str, torch.Tensor], predicted_tokens: torch.Tensor, num_beams: int
    ) -> Dict[str, torch.Tensor]:
        """Expand `model_inputs` before beam search generation
            and prepare for the next token generation
            return tensors should have shape: (batch_size*num_beams, *other_dims)
           Overrides this function when necessary
        """
        # Expand input_ids
        for key, value in model_inputs.items():
            if key == "past":
                new_past = []
                for item in model_inputs["past"]:
                    item = item.unsqueeze(2).expand(-1, -1, num_beams, -1, -1, -1)
                    item = item.reshape(item.shape[0], -1, item.shape[3], item.shape[4], item.shape[5])
                    new_past.append(item)
                model_inputs["past"] = new_past
            elif key == "input_ids":
                model_inputs["input_ids"] = predicted_tokens.reshape(self.batch_size * num_beams, 1)
            else:
                # assumes to have the shape (batch_size, other_dims)
                value_shape = value.shape
                model_inputs[key] = value.unsqueeze(1).expand(-1, num_beams,
                                                              *value_shape[1:]).reshape(-1, *value_shape[1:])
        return model_inputs

    def reorder_model_inputs(
        self, model_inputs: Dict[str, torch.Tensor], predicted_tokens: torch.Tensor, beam_indices_1d: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Reorder `model_inputs` during beam search generation
            and prepare for the next token generation
           Overrides this function when necessary
        """
        for key, value in model_inputs.items():
            if key == "past":
                model_inputs["past"] = [item[:, beam_indices_1d] for item in model_inputs["past"]]
            elif key == "input_ids":
                model_inputs["input_ids"] = predicted_tokens.reshape(-1, 1)
            else:
                model_inputs[key] = value[beam_indices_1d]
        return model_inputs

    def filter_finished_model_inputs(self, model_inputs, keeped_indices: List[int]):
        "Overrides this function whenever necessary"
        for key, value in model_inputs.items():
            if key == "past":
                model_inputs["past"] = [item[:, keeped_indices] for item in value]
            else:
                model_inputs[key] = value[keeped_indices]
        return model_inputs

    def increment_model_inputs(self, model_inputs):
        "Overrides this function whenever necessary"
        if "attention_mask" in model_inputs:
            model_inputs["attention_mask"] = F.pad(model_inputs["attention_mask"], (0, 1), 'constant', True)

        if "position_ids" in model_inputs:
            model_inputs["position_ids"] = model_inputs["position_ids"] + 1
        return model_inputs

    def pop_finished_sequences(self, timestep, current_sequence_indices, generated_token_sequences):
        """Used for sampling generation 
        """
        sequence_poped = False
        eos_token_len = len(self.eos_token_ids)
        keeped_sequence_indices = torch.arange(len(current_sequence_indices))

        if timestep >= eos_token_len:
            # shape: (batch_size, 1)
            check_if_eos = torch.all(
                generated_token_sequences[current_sequence_indices, timestep - eos_token_len + 1:timestep +
                                          1] == self.eos_token_ids,
                dim=1
            )
            sequence_poped = torch.any(check_if_eos)
            keeped_sequence_indices = (~check_if_eos).nonzero(as_tuple=False).squeeze(1)
            if sequence_poped:
                current_sequence_indices = current_sequence_indices[keeped_sequence_indices]
        return sequence_poped, keeped_sequence_indices, current_sequence_indices