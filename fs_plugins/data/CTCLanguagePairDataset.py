from fairseq.data import LanguagePairDataset
from fairseq.data import FairseqDataset, data_utils


class CTCLanguagePairDataset(LanguagePairDataset):
    def __init__(self, trg_length_multiplier=4, *args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.trg_length_multiplier = trg_length_multiplier

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        print('before length filtering:', len(indices))
        indices = indices[self.src_sizes[indices] * self.trg_length_multiplier > self.tgt_sizes[indices] + 3]
        print('after length filtering:', len(indices))
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )