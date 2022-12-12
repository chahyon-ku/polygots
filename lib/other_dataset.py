import torch
import transformers
import lib.reader_utils
import lib.log
    


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
        self.tokenizer = None

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.encoder_model, cache_dir=self.cache_dir,
                                                                        additional_special_tokens=self.additional_special_tokens)
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

        self.encoder_model = encoder_model
        self.cache_dir = cache_dir
        self.additional_special_tokens = list(target_vocab.keys())[1:]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_model, cache_dir=cache_dir,
                                                                    additional_special_tokens=self.additional_special_tokens)

        self.mlm_prob = mlm_prob
        self.mask_all_entities = mask_all_entities
        self.instances = []
        self.read_data(data)
        self.tokenizer = None

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.encoder_model, cache_dir=self.cache_dir,
                                                                        additional_special_tokens=self.additional_special_tokens)
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
