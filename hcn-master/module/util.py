#util包里面放的一般是公用的方法，多功能的包
#read_content把用户说的话用空格分隔开来(str类型)：good morning i'd like to book a table with italian food <SILENCE> in paris for six people please in a cheap price range please <SILENCE> actually i would prefer for two...
def read_content():
    return ' '.join(get_utterances())
#返回所有的对话，如果with_indices=True,则也返回索引表，start:,end:,基本返回如下：[['good morning', 'hello what can i help you with today'], ["i'd like to book a table with italian food", "i'm on it"], ['<SILENCE>', 'where should it be'], ...
def read_dialogs(with_indices=False):

    def rm_index(row):
        return [' '.join(row[0].split(' ')[1:])] + row[1:]

    def filter_(dialogs):
        filtered_ = []
        for row in dialogs:
            if row[0][:6] != 'resto_':
                filtered_.append(row)
        return filtered_

    with open('data/dialog-babi-task5-full-dialogs-trn.txt') as f:
        dialogs = filter_([ rm_index(row.split('\t')) for row in  f.read().split('\n') ])
        # organize dialogs -> dialog_indices
        prev_idx = -1
        n = 1
        dialog_indices = []
        updated_dialogs = []
        for i, dialog in enumerate(dialogs):
            if not dialogs[i][0]:
                dialog_indices.append({
                    'start' : prev_idx + 1,
                    'end' : i - n + 1
                })
                prev_idx = i-n
                n += 1
            else:
                updated_dialogs.append(dialog)        

        if with_indices:
            return updated_dialogs, dialog_indices[:-1]

        return updated_dialogs

#将用户说的话形成一个列表返回：['good morning', "i'd like to book a table with italian food", '<SILENCE>', 'in paris', 'for six people please', 'in a cheap price range please', ...
def get_utterances(dialogs=[]):
    dialogs = dialogs if len(dialogs) else read_dialogs()
    return [ row[0] for row in dialogs ]
#获得机器的回应，返回一个list:['hello what can i help you with today', "i'm on it", 'where should it be', 'how many people would be in your party', 'which price range are looking for', 'ok let me look into some...
def get_responses(dialogs=[]):
    dialogs = dialogs if len(dialogs) else read_dialogs()
    return [ row[1] for row in dialogs ] 


def get_entities():

    def filter_(items):
        return sorted(list(set([ item for item in items if item and '_' not in item ])))

    with open('data/dialog-babi-kb-all.txt') as f:
        return filter_([item.split('\t')[-1] for item in f.read().split('\n') ])
