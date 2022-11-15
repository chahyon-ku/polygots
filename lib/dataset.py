import collections

import numpy as np
import torch
import transformers
import lib.log
import lib.reader_utils


# From MultiCoNER baseline


tags_v1 = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
tags_v2 = {'O': 0, 'I-Artist': 1, 'B-Artist': 2, 'I-OtherPER': 3, 'I-VisualWork': 4, 'B-HumanSettlement': 5, 'I-MusicalWork': 6, 'I-Athlete': 7, 'B-OtherPER': 8, 'I-WrittenWork': 9, 'I-ORG': 10, 'B-Athlete': 11, 'I-Politician': 12, 'I-SportsGRP': 13, 'B-ORG': 14, 'I-MusicalGRP': 15, 'B-VisualWork': 16, 'B-MusicalWork': 17, 'B-Politician': 18, 'I-Facility': 19, 'B-MusicalGRP': 20, 'B-WrittenWork': 21, 'B-SportsGRP': 22, 'B-Facility': 23, 'B-OtherPROD': 24, 'I-HumanSettlement': 25, 'B-Software': 26, 'I-Scientist': 27, 'I-OtherLOC': 28, 'B-PublicCorp': 29, 'I-ArtWork': 30, 'I-PublicCorp': 31, 'I-OtherPROD': 32, 'I-SportsManager': 33, 'I-Cleric': 34, 'I-AerospaceManufacturer': 35, 'B-Disease': 36, 'B-Medication/Vaccine': 37, 'B-Scientist': 38, 'I-Software': 39, 'B-Food': 40, 'I-Station': 41, 'B-Vehicle': 42, 'B-SportsManager': 43, 'B-CarManufacturer': 44, 'B-Cleric': 45, 'B-AnatomicalStructure': 46, 'B-Drink': 47, 'B-Station': 48, 'I-CarManufacturer': 49, 'B-AerospaceManufacturer': 50, 'B-OtherLOC': 51, 'B-MedicalProcedure': 52, 'I-Vehicle': 53, 'B-Symptom': 54, 'B-Clothing': 55, 'B-ArtWork': 56, 'I-Symptom': 57, 'I-Disease': 58, 'I-PrivateCorp': 59, 'I-Drink': 60, 'B-PrivateCorp': 61, 'I-AnatomicalStructure': 62, 'I-Food': 63, 'I-MedicalProcedure': 64, 'I-Medication/Vaccine': 65, 'I-Clothing': 66}


class CoNLLDataset(torch.utils.data.Dataset):
    def __init__(self, data, target_vocab, encoder_model, cache_dir, max_instances=-1, max_length=50):
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_model, cache_dir=cache_dir)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.tag_to_id = {} if target_vocab is None else target_vocab
        self.instances = []
        self.read_data(data)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        lib.log.logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in lib.reader_utils.get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            if len(fields) == 0:
                continue
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask = self.parse_line_for_ner(fields=fields)

            tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype=torch.long).unsqueeze(0)
            token_masks_rep = torch.tensor(token_masks_rep)
            mask_rep = torch.tensor(mask)

            self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor))
            instance_idx += 1
        lib.log.logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = lib.reader_utils.extract_spans(ner_tags_rep)
        coded_ner_ = [self.tag_to_id[tag] if tag in self.tag_to_id else self.tag_to_id['O'] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        token_masks_rep = [False]
        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))
            rep_ = self.tokenizer(token.lower())['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            tags, masks = lib.reader_utils.assign_ner_tags(ner_tag, rep_)

            ner_tags_rep.extend(tags)
            token_masks_rep.extend(masks)

        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep.append(False)
        mask = [True] * len(tokens_sub_rep)
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask

    def collate_batch(self, batch):
        batch_ = list(zip(*batch))
        tokens, masks, token_masks, gold_spans, tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.tag_to_id['O'])
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)

        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]
            mask_tensor[i, :seq_len] = masks[i]
            token_masks_tensor[i, :seq_len] = token_masks[i]

        return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, data, target_vocab, encoder_model, cache_dir, mlm_prob, max_instances=-1, max_length=50):
        super(MLMDataset, self).__init__()
        self._max_instances = max_instances
        self._max_length = max_length

        additional_special_tokens = list(target_vocab.keys())[1:]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_model, cache_dir=cache_dir,
                                                                    additional_special_tokens=additional_special_tokens)

        self.mlm_prob = mlm_prob

        self.instances = []
        self.read_data(data)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        lib.log.logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in lib.reader_utils.get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            if len(fields) == 0:
                continue

            words, _, _, tags = fields
            sentence_list = []
            for i, (word, tag) in enumerate(zip(words, tags)):
                if tag != 'O':
                    sentence_list.append(tag)
                sentence_list.append(word)

            results = self.tokenizer(sentence_list, return_attention_mask=True, is_split_into_words=True,
                                     return_tensors='pt', return_special_tokens_mask=True)
            input_ids = results['input_ids']
            attention_mask = results['attention_mask']
            special_tokens_mask = results['special_tokens_mask']

            self.instances.append((input_ids.squeeze(), attention_mask.squeeze(), special_tokens_mask.squeeze()))
            instance_idx += 1
        lib.log.logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def collate_batch(self, batch):
        batch = list(zip(*batch))
        batch_input_ids, batch_attention_mask, batch_special_tokens_mask = batch

        batch_size = len(batch_input_ids)
        max_len = max([len(input_ids) for input_ids in batch_input_ids])
        batch_input_ids_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        batch_attention_masks_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.bool).fill_(False)
        batch_labels_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.long).fill_(self.tokenizer.pad_token_id)

        for i_sample, (input_ids, attention_masks, special_tokens_mask) in \
                enumerate(zip(batch_input_ids, batch_attention_mask, batch_special_tokens_mask)):
            masked_input_ids = input_ids.clone()
            seq_len = len(input_ids)

            mapping = []
            for i_id, id in enumerate(input_ids):
                id = id.item()
                if special_tokens_mask[i_id]:
                    continue
                if id in self.tokenizer.additional_special_tokens_ids:
                    continue
                token = self.tokenizer.convert_ids_to_tokens(id)
                # start of a word
                if token.startswith('▁'):
                    mapping.append([])
                if len(mapping) == 0:
                    print(self.tokenizer.convert_ids_to_tokens(input_ids))
                mapping[-1].append(i_id)
            for indices in mapping:
                if np.random.random(1) < self.mlm_prob:
                    masked_input_ids[indices] = self.tokenizer.mask_token_id

            batch_input_ids_tensor[i_sample, :seq_len] = masked_input_ids
            batch_attention_masks_tensor[i_sample, :seq_len] = attention_masks
            batch_labels_tensor[i_sample, :seq_len] = input_ids

        return batch_input_ids_tensor, batch_attention_masks_tensor, batch_labels_tensor


class MLMDatasetv2(torch.utils.data.Dataset):
    def __init__(self, data, target_vocab, encoder_model, cache_dir, mlm_prob, mask_all_entities, max_instances=-1, max_length=50):
        super(MLMDatasetv2, self).__init__()
        self._max_instances = max_instances
        self._max_length = max_length

        additional_special_tokens = list(target_vocab.keys())[1:]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_model, cache_dir=cache_dir,
                                                                    additional_special_tokens=additional_special_tokens)

        self.mlm_prob = mlm_prob
        self.mask_all_entities = mask_all_entities
        self.instances = []
        self.read_data(data)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        lib.log.logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in lib.reader_utils.get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            if len(fields) == 0:
                continue

            words, _, _, tags = fields
            sentence_list = []
            for i, (word, tag) in enumerate(zip(words, tags)):
                if tag != 'O':
                    sentence_list.append(tag)
                sentence_list.append(word)

            results = self.tokenizer(sentence_list, return_attention_mask=True, is_split_into_words=True,
                                     return_tensors='pt', return_special_tokens_mask=True)
            input_ids = results['input_ids']
            attention_mask = results['attention_mask']
            special_tokens_mask = results['special_tokens_mask']

            self.instances.append((input_ids.squeeze(), attention_mask.squeeze(), special_tokens_mask.squeeze()))
            instance_idx += 1
        lib.log.logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def collate_batch(self, batch):
        batch = list(zip(*batch))
        batch_input_ids, batch_attention_mask, batch_special_tokens_mask = batch

        batch_size = len(batch_input_ids)
        max_len = max([len(input_ids) for input_ids in batch_input_ids])
        batch_input_ids_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        batch_attention_masks_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.bool).fill_(False)
        batch_labels_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.long).fill_(self.tokenizer.pad_token_id)

        for i_sample, (input_ids, attention_masks, special_tokens_mask) in \
                enumerate(zip(batch_input_ids, batch_attention_mask, batch_special_tokens_mask)):
            masked_input_ids = input_ids.clone()
            seq_len = len(input_ids)

            is_ne = []
            mapping = []
            is_prev_ner = False
            for i_id, id in enumerate(input_ids):
                id = id.item()
                if special_tokens_mask[i_id]:
                    continue
                if id in self.tokenizer.additional_special_tokens_ids:
                    is_prev_ner = True
                    continue
                token = self.tokenizer.convert_ids_to_tokens(id)
                # start of a word
                if token.startswith('▁'):
                    is_ne.append(is_prev_ner)
                    mapping.append([])
                mapping[-1].append(i_id)
                is_prev_ner = False

            if self.mask_all_entities:
                mapping = [mapping[i_mapping] for i_mapping in range(len(mapping)) if is_ne[i_mapping]]
            for indices in mapping:
                if self.mask_all_entities or np.random.random(1) < self.mlm_prob:
                    masked_input_ids[indices] = self.tokenizer.mask_token_id

            batch_input_ids_tensor[i_sample, :seq_len] = masked_input_ids
            batch_attention_masks_tensor[i_sample, :seq_len] = attention_masks
            batch_labels_tensor[i_sample, :seq_len] = input_ids

        return batch_input_ids_tensor, batch_attention_masks_tensor, batch_labels_tensor


class CLMDataset(torch.utils.data.Dataset):
    def __init__(self, data, target_vocab, encoder_model, cache_dir, lang_code, max_instances=-1, max_length=50):
        super(CLMDataset, self).__init__()
        self._max_instances = max_instances
        self._max_length = max_length

        additional_special_tokens = list(target_vocab.keys())[1:]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_model, cache_dir=cache_dir,
                                                                    additional_special_tokens=additional_special_tokens)

        self.instances = []
        self.read_data(data)
        self.lang_id = self.tokenizer.lang2id[lang_code]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        lib.log.logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in lib.reader_utils.get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            if len(fields) == 0:
                continue

            words, _, _, tags = fields
            sentence_list = []
            for i, (word, tag) in enumerate(zip(words, tags)):
                if tag != 'O':
                    sentence_list.append(tag)
                sentence_list.append(word)

            results = self.tokenizer(sentence_list, return_attention_mask=True, is_split_into_words=True,
                                     return_tensors='pt', return_special_tokens_mask=True)
            input_ids = results['input_ids']
            attention_mask = results['attention_mask']
            special_tokens_mask = results['special_tokens_mask']

            self.instances.append((input_ids.squeeze(), attention_mask.squeeze(), special_tokens_mask.squeeze()))
            instance_idx += 1
        lib.log.logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def collate_batch(self, batch):
        batch = list(zip(*batch))
        batch_input_ids, batch_attention_mask, batch_special_tokens_mask = batch

        batch_size = len(batch_input_ids)
        max_len = max([len(input_ids) for input_ids in batch_input_ids])
        batch_input_ids_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        batch_attention_masks_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.bool).fill_(False)
        batch_lang_ids_tensor = torch.empty(size=(batch_size, max_len), dtype=torch.long).fill_(self.lang_id)

        for i_sample, (input_ids, attention_masks, special_tokens_mask) in \
                enumerate(zip(batch_input_ids, batch_attention_mask, batch_special_tokens_mask)):
            seq_len = len(input_ids)

            batch_input_ids_tensor[i_sample, :seq_len] = input_ids
            batch_attention_masks_tensor[i_sample, :seq_len] = attention_masks

        return batch_input_ids_tensor, batch_attention_masks_tensor, batch_lang_ids_tensor
