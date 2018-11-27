"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np
import pdb
import scipy.sparse as sp


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.

    Parameters
    ----------

    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """

    def __init__(self, file_path,
                 user_map=None,
                 item_map=None,
                 rate_map=None):

        if not user_map and not item_map and not rate_map:
            user_map = dict()
            item_map = dict()
            rate_map = dict()

            num_user = 0
            num_item = 0
            num_rate = 0
            
            user_ids = list()
            item_ids = list()
            rate_ids = list()
            # read users and items from file
            with open(file_path, 'r') as fin:
                for line in fin:
                    u, i, r = line.strip().split(',')
                    user_ids.append(u)
                    item_ids.append(i)
                    rate_ids.append(r)
            #pdb.set_trace()

            # update user and item mapping
            for u in user_ids:
                if u not in user_map:
                    user_map[u] = num_user
                    num_user += 1
            for i in item_ids:
                if i not in item_map:
                    item_map[i] = num_item
                    num_item += 1

            for r in rate_ids:
                if r not in rate_map:
                    rate_map[r] = num_rate
                    num_rate += 1

        else:
            num_user = len(user_map)
            num_item = len(item_map)
            num_rate = len(rate_map)

            user_ids = list()
            item_ids = list()
            rate_ids = list()
            
            user_keys = user_map.keys()
            item_keys = item_map.keys()
            rate_keys = rate_map.keys()
            
            # read users and items from file
            with open(file_path, 'r') as fin:
                for line in fin:
                    u, i, r = line.strip().split(',')
                    if u in user_keys and i in item_keys:
                        user_ids.append(u)
                        item_ids.append(i)
                        rate_ids.append(r)

        user_ids = np.array([user_map[u] for u in user_ids])
        item_ids = np.array([item_map[i] for i in item_ids])
        rate_ids = np.array([rate_map[i] for i in rate_ids])
        #pdb.set_trace()

        print("READ INPUT FILE ",file_path,"...")
        print("user_id ",min(user_ids),"~",max(user_ids)," [num: ",len(user_ids), " (unique: ", num_user,")]")
        print("item_id ",min(item_ids),"~",max(item_ids)," [num: ",len(item_ids), " (unique: ", num_item,")]")
        print("rate_id ",min(rate_ids),"~",max(rate_ids)," [num: ",len(rate_ids), " (unique: ", num_rate,")]")


        self.num_users = num_user
        self.num_items = num_item
        self.num_rates = num_rate

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.rate_ids = rate_ids

        self.user_map = user_map
        self.item_map = item_map
        self.rate_map = rate_map

        self.sequences = None
        self.test_sequences = None

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        #data = self.rate_ids
        data = np.ones(len(self))
        
        #print("shape", self.num_users+1, self.num_items+1)
        #print(max(row), max(col))
        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users+1, self.num_items+1))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        # change the item index start from 1 as 0 is used for padding in sequences
        for k, v in self.item_map.items():
            self.item_map[k] = v + 1
        self.item_ids = self.item_ids + 1
        self.num_items = self.num_items +1

        print("SET ITEM PADDING...")
        print("user_id ",min(self.user_ids),"~",max(self.user_ids)," [num: ",len(self.user_ids), " (unique: ", self.num_users,")]")
        print("item_id ",min(self.item_ids),"~",max(self.item_ids)," [num: ",len(self.item_ids), " (unique: ", self.num_items,")]")

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]
        rate_ids = self.rate_ids[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)
        # print(indices.size)
        # print(len(item_ids))
        indices = np.append(indices, len(item_ids)) ### append하고 다시 넣어줘야함.
        # print(indices.size)
        # print("counts check", counts[len(counts)-1], len(item_ids[indices[len(counts)-1]:indices[len(counts)]]))

        # num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])
        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 0 for c in counts])

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)
        sequence_rates = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)
        test_rates = np.empty(self.num_users,
                              dtype=np.int64)
        _uid = None
        for i, (uid, rid,
                item_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           rate_ids,
                                                           indices,
                                                           max_sequence_length)):

            if uid != _uid:
                _uid = uid
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_users[uid] = uid
                test_rates[uid] = rid

            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid
            sequence_rates[i] = rid
        # print("i", i)
        # print("num_subsequences", num_subsequences)
        # print("item이 충분한 유저 갯수", len(np.unique(sequence_users)))
        # print("max uid", max(sequence_users))
        # print(self.num_users)
        #print("before SeqInteractions", type(sequence_rates), len(sequence_rates))
        self.sequences = SequenceInteractions(sequence_users, sequences, sequence_rates, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences, test_rates)


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences,
                 rate_ids,
                 targets=None):
        #print("in SequenceInteractions", type(user_ids), len(user_ids), type(rate_ids), len(rate_ids))
        self.user_ids = user_ids
        self.rate_ids = rate_ids
        self.sequences = sequences
        self.targets = targets
        #print(type(self.user_ids), len(self.user_ids), type(rate_ids), len(rate_ids), type(self.rate_ids), len(self.rate_ids))
        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, window_size, step_size=1):
    # if len(tensor) - window_size >= 0:
    for i in range(len(tensor), 0, -step_size):
        if i - window_size >= 0:
            yield tensor[i - window_size:i]
        else:
            break
    # else:
    #     yield None
    #     # print("break!")
    #     # yield tensor #### 여기서 그냥 반환하면 위에서 접근할 때 길이가 안맞아서 오류가 나지 않나?


def _generate_sequences(user_ids, item_ids, rate_ids,
                        indices,
                        max_sequence_length):
    # sum = 0
    for i in range(len(indices)-1):
        start_idx = indices[i]
        stop_idx = indices[i + 1]
        items = item_ids[start_idx:stop_idx]
        if len(items) < max_sequence_length:
            continue
            # print("not" , user_ids[i])
        else:
            # sum += (len(items)-max_sequence_length+1)
            # print(sum)
            # print("yes" , user_ids[i])
            for seq in _sliding_window(items,max_sequence_length):
                yield (user_ids[i], rate_ids[i], seq)
