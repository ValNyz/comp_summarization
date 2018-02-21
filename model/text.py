import collections

s_id = 0


class document(collections.MutableSequence):
    def __init__(self, *args):
        self.list = list()
        self.sents_id = {}
        self.id_sents = {}
        global s_id
        # import pdb
        # pdb.set_trace()
        for i, sent in enumerate(list(args)):
            self.append(sent)
            self.sents_id[i] = s_id
            self.id_sents[s_id] = i
            s_id += 1

    def __getitem__(self, i):
        return self.list[i]

    def __setitem__(self, i, v):
        self.list[i] = v

    def get_dict_id_sents(self):
        return self.id_sents

    def get_dict_sents_id(self):
        return self.sents_id

    def getById(self, id):
        return self.list[self.sents_id[id]]

    def __delitem__(self, i):
        self.remove(i)
        del self.id_sents[self.sents_id[i]]
        del self.sents_id[i]

    def insert(self, i, v):
        global s_id
        self.sents_id[s_id] = i
        self.id_sents[i] = s_id
        self.list.insert(i, v)
        s_id += 1

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return str(self.id_sents) # self.list)
