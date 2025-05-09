from datasets import load_dataset
from torch.utils.data import Dataset
from typing import List, Dict, Any, Set
from global_user_intents import USER_INTENTS, AtomicIntent
import random


def load_data(dataset: str, num_ex: int = -1, split: str = 'train', num_users: int = 5):
    """
    return an OurInputDataset object with the specified number of examples
    see the choices for split on each dataset below
    
    Args:
        dataset: name of the dataset to load
        num_ex: number of examples to load (-1 for all)
        split: dataset split to use
        num_users: number of users to distribute examples among
    """

    if dataset == 'cnn_dailymail':
        # possible splits: train, validation, test
        data = load_dataset(dataset, '3.0.0', split=split)
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='id', article_key='article',
                                      doc_type=dataset, num_users=num_users)
    elif dataset == 'xsum':
        # possible splits: train, validation, test
        data = load_dataset(dataset)[split]
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='id', article_key='document',
                                      doc_type=dataset, num_users=num_users)
    elif dataset == 'slf5k':
        # possible splits: train, development, validation, test
        data = load_dataset("JeremyAlain/SLF5K")[split]
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='id', article_key='post',
                                      doc_type=dataset, num_users=num_users) 
    elif dataset == 'wikipedia':
        # possible splits: train 
        data = load_dataset(dataset, '20220301.simple')[split]
        filtered_data = []
        for ex in data:
            length = len(ex['text'].split()) 
            if length > 500 and length < 700: # 4809
                filtered_data.append(ex)
        parsed_data = OurInputDataset(data=filtered_data, num_ex=num_ex, id_key='id', article_key='text',
                                      doc_type=dataset, num_users=num_users)
    elif dataset == 'CShorten/ML-ArXiv-Papers':
        # possible splits: train
        data = load_dataset(dataset)[split]
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='Unnamed: 0.1', article_key='abstract',
                                      doc_type=dataset, num_users=num_users)
    elif dataset == 'imdb':
        # possible splits: train, test, unsupervised
        data = load_dataset(dataset)[split]
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='label', article_key='text',
                                      doc_type=dataset, num_users=num_users)
    elif dataset == 'ccby':
        data = load_dataset('orieg/elsevier-oa-cc-by')['train']
        filtered_data = []
        for ex in data:
            if ex['author_highlights'] != []:
                filtered_data.append({'id': len(filtered_data), 
                                      'text': ' My title: ' + ex['title'] + 
                                              '. My abstract: ' + ex['abstract'] +
                                              ' My highlights: ' + '. '.join(ex['author_highlights'])})
        parsed_data = OurInputDataset(data=filtered_data, num_ex=num_ex, id_key='id', article_key='text',
                                      doc_type=dataset, num_users=num_users) 
    elif dataset == 'ampere':
        data = load_dataset('launch/ampere', split='train')
        data = [{'id': ex['doc_id'], 'text': ' '.join(ex['text'])} for ex in data]
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='id', article_key='text',
                                      doc_type=dataset, num_users=num_users) 
    elif dataset == 'paper_tweet':
        data = load_dataset('nitsanb/paper_tweet', split='train')
        filtered_data = []
        for ex in data:
            ex = ex['text']
            if '[' in ex:
                continue
            try:
                tweet = ex[(ex.index('"') + 1):(ex.rfind('"') - 2)]
            except Exception:
                continue 
            filtered_data.append({'id': len(filtered_data), 'text': tweet})
        parsed_data = OurInputDataset(data=filtered_data, num_ex=num_ex, id_key='id', article_key='text',
                                      doc_type=dataset, num_users=num_users) 
    else:
        raise NotImplementedError

    return parsed_data





class OurInputExample:
    """
    Structure for one input example with id, article, and metadata
    """
    def __init__(self, 
                 id: str, 
                 article: str, 
                 doc_type: str,
                 user_pref: Set[AtomicIntent] = None,
                 model_summary: str = None, 
                 user_edits: str = None, 
                 model_refinement: str = None,
                 eval_yesno: str = None, 
                 eval_rationale: str = None):
        """
        Creates one InputExample with the given id, article and metadata
        """
        self.id = id
        self.article = article
        self.doc_type = doc_type
        self.user_pref = user_pref
        self.model_summary = model_summary
        self.user_edits = user_edits
        self.model_refinement = model_refinement
        self.eval_yesno = eval_yesno
        self.eval_rationale = eval_rationale

    def __str__(self):
        article_preview = self.article[:100] + "..." if len(self.article) > 100 else self.article
        return f"<InputExample> article ID {self.id}: {article_preview}"


class OurInputDataset(Dataset): 
    """
    Structure for a dataset with example ids and articles
    We create this object to unify different datasets on hugging face which have different formats
    """
    def __init__(self, data, num_ex: int, id_key: str, article_key: str, doc_type: str = '', num_users: int = 5):
        from itertools import islice
        if num_ex > 0:
            data = islice(data, num_ex)

        # Get list of user IDs
        user_ids = list(USER_INTENTS.keys())
        if num_users > len(user_ids):
            num_users = len(user_ids)
        
        # Create dataset with one user per example
        self.dataset = []
        for i, d in enumerate(data):
            # Assign each example to one user in a round-robin fashion
            user_id = user_ids[i % num_users]
            user_pref = USER_INTENTS[user_id].intents
            self.dataset.append(OurInputExample(
                id=f"{d[id_key]}_{user_id}",
                article=d[article_key],
                doc_type=doc_type,
                user_pref=user_pref
            ))

    def __getitem__(self, item):        
        return self.dataset[item]
    
    def __len__(self):
        return len(self.dataset)
        
    def get_unique_users(self) -> List[str]:
        """
        Returns a list of unique user IDs in the dataset.
        Each user ID is extracted from the example IDs which are in the format 'original_id_userid'
        """
        unique_users = set()
        for example in self.dataset:
            # Extract user ID from the combined ID (format: original_id_userid)
            user_id = example.id.split('_')[-1]
            unique_users.add(user_id)
        return sorted(list(unique_users))
        
    def get_examples_by_user(self, user_id: str) -> List[OurInputExample]:
        """
        Returns all examples assigned to a specific user.
        
        Args:
            user_id: The ID of the user to get examples for
            
        Returns:
            A list of OurInputExample objects assigned to the specified user
        """
        return [example for example in self.dataset if example.id.split('_')[-1] == user_id]
        
