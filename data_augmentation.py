import pandas as pd
import numpy as np

CHANGES = dict(zip([" game ", " set "], [" lame ", " bet "]))
def augment_dataset_dict(dataset_dict, changes = None):

    changes=CHANGES
    original_df = pd.DataFrame({x: dataset_dict[x] for x in dataset_dict if x not in ['label']})
    df= original_df.copy()
    #TODO : Correct for start end shifts, do not modify answers
    #df['start_char'] = df.answer.apply(lambda x: x['answer_start'][0])
    #df['end_char'] = df['start_char'] + df.answer.apply(lambda x: len(x['text'][0]))
    #df['final_answer'] = [A[B:C] for A, B, C in zip(df.context, df['start_char'], df['end_char'])]
    df['context'] = df.context.str.strip().replace(changes, regex=True)
    ##df['new_context'] = df.context.str.strip().replace(changes,regex=True)
    ##df['new_answer'] = [A[B:C] for A, B, C in zip(df['new_context'], df['start_char'],df['end_char'])]
    _ = pd.concat([original_df , df[[i for i in dataset_dict.keys() if i != 'label']]])
    new_dataset_dict = pd.concat([original_df , df[[i for i in dataset_dict.keys() if i != 'label']]]).to_dict()
    new_dataset_dict['label'] = dataset_dict['label']
    return new_dataset_dict



