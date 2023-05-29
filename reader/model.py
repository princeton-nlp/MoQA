from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

@dataclass
class ReaderModelOutput(QuestionAnsweringModelOutput):
    rank_logits: torch.FloatTensor = None

class BertReader(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels # this should be 2

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        offset_mapping: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size, passages per question, num_answers)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size, passages per question, num_answers)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        N, M, L = input_ids.size()

        outputs = self.bert(
            input_ids.view(N*M, L) if input_ids is not None else None,
            attention_mask=attention_mask.view(N*M, L) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(N*M, L) if token_type_ids is not None else None,
            position_ids=position_ids.view(N*M, L) if position_ids is not None else None,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds.view(N*M, L) if inputs_embeds is not None else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # these are the last hidden states

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        rank_logits = self.qa_classifier(sequence_output[:, 0]).view(N, M)

        # compute rank loss
        cel = nn.CrossEntropyLoss()
        rank_labels = torch.zeros(N, dtype=torch.long, device=input_ids.device)
        rank_loss = cel(rank_logits, rank_labels)
        total_loss = rank_loss

        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.view(N * M, -1).long()
            end_positions = end_positions.view(N * M, -1).long()

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignored_index)

            start_losses = [loss_fct(start_logits, start_pos) for start_pos in torch.unbind(start_positions, dim=1)]
            end_losses = [loss_fct(start_logits, end_pos) for end_pos in torch.unbind(end_positions, dim=1)]

            total_losses = torch.stack(start_losses, dim=1) + torch.stack(end_losses, dim=1)

            def _calc_mml(loss_tensor):
                # from DPR code
                marginal_likelihood = torch.sum(torch.exp(-loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)
                log_marginal_likelihood = torch.log(marginal_likelihood + (marginal_likelihood == 0).float())
                return -torch.sum(log_marginal_likelihood)

            #total_losses = total_losses.view(N, M, -1).max(dim=1)[0] # used in DPR, max over passage dim
            #total_losses = total_losses.view(N, M, -1).max(dim=2)[0] # max over all answers in a passage
            #span_loss = _calc_mml(total_losses)
            span_loss = total_losses.sum() # sum all span losses
            total_loss += span_loss

        if not return_dict:
            output = (start_logits, end_logits, rank_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ReaderModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            rank_logits=rank_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaReader(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        answer_mask: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        offset_mapping: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        N, M, L = input_ids.size()

        outputs = self.roberta(
            input_ids.view(N*M, L) if input_ids is not None else None,
            attention_mask=attention_mask.view(N*M, L) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(N*M, L) if token_type_ids is not None else None,
            position_ids=position_ids.view(N*M, L) if position_ids is not None else None,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds.view(N*M, L) if inputs_embeds is not None else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        #start_logits = logits[:,:,0].view(N*M, L)
        #end_logits = logits[:,:,1].view(N*M, L)

        rank_logits = self.qa_classifier(sequence_output[:, 0]).view(N, M)

        # compute rank loss
        cel = nn.CrossEntropyLoss()
        rank_labels = torch.zeros(N, dtype=torch.long, device=input_ids.device)
        rank_loss = cel(rank_logits, rank_labels)
        total_loss = rank_loss

        if start_positions is not None and end_positions is not None and answer_mask is not None:
            start_positions = start_positions.view(N * M, -1).long()
            end_positions = end_positions.view(N * M, -1).long()
            answer_mask = answer_mask.view(N * M, -1).long()

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignored_index)

            start_losses = [
                loss_fct(start_logits, start_pos) * span_mask
                for start_pos, span_mask in zip(
                    torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1)
                )
            ]
            end_losses = [
                loss_fct(end_logits, end_pos) * span_mask
                for end_pos, span_mask in zip(
                    torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1)
                )
            ]

            total_losses = torch.stack(start_losses, dim=1) + torch.stack(end_losses, dim=1)

            def _calc_mml(loss_tensor):
                # from DPR code
                marginal_likelihood = torch.sum(torch.exp(-loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)
                log_marginal_likelihood = torch.log(marginal_likelihood + (marginal_likelihood == 0).float())
                return -torch.sum(log_marginal_likelihood)

            total_losses = total_losses.view(N, M, -1).max(dim=1)[0] # used in DPR, max over passage dim
            #total_losses = total_losses.view(N, M, -1).max(dim=2)[0] # max over all answers in a passage
            span_loss = _calc_mml(total_losses)
            #span_loss = total_losses.sum() # sum all span losses
            total_loss += span_loss

        start_logits = start_logits.view(N, M, L)
        end_logits = end_logits.view(N, M, L)
        if not return_dict:
            output = (start_logits, end_logits, rank_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ReaderModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            rank_logits=rank_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

