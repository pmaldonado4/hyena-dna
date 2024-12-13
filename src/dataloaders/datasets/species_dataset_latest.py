from pathlib import Path
from pyfaidx import Fasta
import torch
from random import randrange, random
import numpy as np
import random as rnd
from torch.utils.data import DataLoader
# Same chromosome splits for each species
SPECIES_CHROMOSOME_SPLITS = {
    'human': {'train': [f'chr{i}' for i in range(4, 23)] + ['chrX'], 'valid': ['chr1', 'chr2'], 'test': ['chr3', 'chr4']},
    'mouse': {'train': [f'chr{i}' for i in range(4, 20)] + ['chrX', 'chrY'], 'valid': ['chr1', 'chr2'], 'test': ['chr3', 'chr4']},
    'lemur': {'train': [f'scaffold{i}' for i in range(1, 51)], 'valid': [f'scaffold{i}' for i in range(51, 61)], 'test': [f'scaffold{i}' for i in range(61, 74)]},
    'squirrel': {'train': [f'scaffold{i}' for i in range(1, 12)], 'valid': [f'scaffold{i}' for i in range(12, 15)], 'test': [f'scaffold{i}' for i in range(15, 18)]}
}

# Helper functions
def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# Reverse complement function
string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        else:
            rev_comp += base
    return rev_comp

class FastaInterval:
    def __init__(self, fasta_file, rc_aug=False, pad_interval=False):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'Path to fasta file must exist'
        self.seqs = Fasta(str(fasta_file))
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval

        # Calculate lengths of each chromosome/scaffold
        self.chr_lens = {chr_name: len(self.seqs[chr_name]) for chr_name in self.seqs.keys()}

    def __call__(self, chr_name, start, end, max_length):
        """Retrieve the sequence for a given chromosome/scaffold and interval."""
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]

        left_padding = right_padding = 0

        # Check if there's enough sequence to fill the interval
        if (end - start) < max_length:
            extra_seq = max_length - (end - start)
            start -= extra_seq // 2
            end += extra_seq // 2

        if start < 0:
            left_padding = -start
            start = 0
        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        seq = str(chromosome[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq

class SequenceDataset:
    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

@SequenceDataset.register('multispecies')
class SpeciesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        species: list,
        species_dir: str,
        split: str,
        max_length,
        tokenizer,
        tokenizer_name=None,
        add_eos=False,
        rc_aug=False,
        pad_interval=False,
        replace_N_token=False,
        batch_size=32,
        num_workers=4,
        total_size=100000  # New parameter
    ):
        self.species = species
        self.species_dir = species_dir
        self.split = split
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval
        self.replace_N_token = replace_N_token
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_size = total_size  # Store total_size
        self.__train_len = self.dataset_size
    
        # Store FASTA files and chromosome splits for each species
        self.fastas = {
            spec: FastaInterval(
                f"{self.species_dir}/{spec}/{spec}_genome.fna",
                rc_aug=self.rc_aug,
                pad_interval=self.pad_interval
            )
            for spec in self.species
        }

        self.chromosomes = {
            spec: SPECIES_CHROMOSOME_SPLITS[spec][self.split]
            for spec in self.species
        }

        # Calculate dataset length
        self.dataset_size = self.total_size  # Use total_size from config

        print(f"Dataset size: {self.dataset_size}")
        
    def __len__(self):
        return self.dataset_size

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        # Select species and chromosome randomly
        spec = rnd.choice(self.species)
        chromosome = rnd.choice(self.chromosomes[spec])
        
        # Select random start position within the chromosome
        chromosome_length = len(self.fastas[spec].seqs[chromosome])
        start = rnd.randint(0, chromosome_length - self.max_length)
        end = start + self.max_length
        if start < 0 or end > chromosome_length:
            print(f"Sequence exceeds bounds for {chromosome} in {spec}. Start: {start}, End: {end}, Chromosome length: {chromosome_length}")
            
        # Get the sequence
        seq = self.fastas[spec](chromosome, start, end, max_length=self.max_length)
        
        # Tokenize the sequence
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq, add_special_tokens=True if self.add_eos else False, padding="max_length", max_length=self.max_length, truncation=True)
            seq = seq["input_ids"]

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, padding="max_length", max_length=self.max_length, truncation=True)
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos token, keep eos
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens

        # Replace 'N' tokens if necessary
        if self.replace_N_token and self.tokenizer_name == 'char':
     # 'N' is likely mapped to a specific ID in the char-level tokenizer
            n_token_id = self.tokenizer.convert_tokens_to_ids('N')  # or manually assign if you know the id
            pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0  # assuming 0 is the pad token id
            seq = self.replace_value(torch.LongTensor(seq), n_token_id, pad_token_id)
        # Prepare data and target
        data = torch.LongTensor(seq[:-1])  # Input sequence
        target = torch.LongTensor(seq[1:])  # Target sequence

        return data, target
    
    def train_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    # Optionally, include val_dataloader and test_dataloader methods
    def val_dataloader(self):
        # Return None or implement validation dataloader
        return None

    def test_dataloader(self):
        # Return None or implement test dataloader
        return None