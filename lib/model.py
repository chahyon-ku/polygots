import typing
import itertools
import torch
import numpy
import allennlp.modules
import transformers
import lib.metric
import lib.reader_utils


class NERModel(torch.nn.Module):
    def __init__(self, encoder_model, cache_dir, tag_to_id, dropout_rate=0.1, pad_token_id=1):
        super(NERModel, self).__init__()
        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id

        self.target_size = len(self.id_to_tag)

        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = transformers.AutoModel.from_pretrained(encoder_model, cache_dir=cache_dir, return_dict=True)

        self.feedforward = torch.nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.target_size)

        self.crf_layer = allennlp.modules.\
            ConditionalRandomField(num_tags=self.target_size,
                                   constraints=allennlp.modules.conditional_random_field.
                                   allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.span_f1 = lib.metric.SpanF1()

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

    def forward(self, tokens, mask):
        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(torch.nn.functional.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)
        token_scores = torch.nn.functional.log_softmax(token_scores, dim=-1)

        return token_scores

        # compute the log-likelihood loss and compute the best NER annotation sequence
        # output = self._compute_token_tags(token_scores=token_scores, mask=mask, tags=tags, metadata=metadata, batch_size=batch_size, mode=mode)
        # return output

    def compute_results(self, token_scores, mask, tags, metadata, mode=''):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf_layer(token_scores, tags, mask) / float(token_scores.shape[0])
        best_path = self.crf_layer.viterbi_tags(token_scores, mask)

        pred_results, pred_tags = [], []
        for i in range(token_scores.shape[0]):
            tag_seq, _ = best_path[i]
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])
            pred_results.append(lib.reader_utils.extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}

        if mode == 'predict':
            output['token_tags'] = pred_tags
        return output

    def reset_metrics(self):
        self.span_f1.reset()


def get_mlm(encoder_model, cache_dir, device):
    encoder = transformers.XLMRobertaForMaskedLM.from_pretrained(encoder_model, cache_dir=cache_dir, return_dict=True)
    encoder = encoder.to(device)
    encoder.resize_token_embeddings(250100)
    return encoder


def get_clm(encoder_model, cache_dir, device):
    encoder = transformers.XLMRobertaForCausalLM.from_pretrained(encoder_model, cache_dir=cache_dir, return_dict=True)
    encoder = encoder.to(device)
    encoder.resize_token_embeddings(250100)
    return encoder
