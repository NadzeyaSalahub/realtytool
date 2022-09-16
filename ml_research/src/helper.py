import pandas as pd
from IPython.display import HTML
pd.set_option('display.max_columns', None)


LABELS_COLOR_MAP = {
    'additional_costs': 'rgb(102,194,165)',
    'bedrooms': 'rgb(252,141,98)',
    'city': 'rgb(141,160,203)',
    'district': 'rgb(231,138,195)',
    'estate_type': 'rgb(166,216,84)',
    'price': 'rgb(255,217,47)',
    'price_max': 'rgb(229,196,148)',
    'price_min': 'rgb(179,179,179)',
    }

def fillna_with_iterable(df, iterable_value, cols=None):
    if cols is None:
        cols = df.columns
    df_copy = df.copy()
    df_copy.loc[:, cols] = df_copy[cols].applymap(lambda x: x if hasattr(x, '__iter__') or pd.notnull(x) else iterable_value)
    return df_copy
pd.DataFrame.fillna_with_iterable = fillna_with_iterable


def explore(df):
    print(df.columns)
    print('Rows x columns:', df.shape)
    display(df.head())
pd.DataFrame.explore = explore


def find_token_positions(text, tokens):
    pos = 0
    positions = []
    for tok in tokens:
        start = text.find(tok, pos)
        end = start + len(tok)
        positions.append({'start': start, 'end': end, 'token': tok})
        pos = end
    return positions


def parse_text_column(text_val):
    if isinstance(text_val, str):
        return text_val
    elif isinstance(text_val, dict):
        return text_val['text']
    return ' '.join(parse_text_column(text_val_it) for text_val_it in text_val)


def glue_entities(tags, starts, ends):
    entities = {}
    cur_tag = None
    cur_start, cur_end = None, None
    try:
        for tag, start, end in zip(tags, starts, ends):
            if tag.startswith('B-'):
                cur_tag = tag[2:]
                cur_start, cur_end = [start, end]
            if tag.startswith('I-'):
                cur_end = end
            if tag == 'O':
                if cur_tag is not None:
                    entities.setdefault(cur_tag, []).append([cur_start, cur_end])
                cur_span = [0, 0]
                cur_tag = None
            
    except:
        entities = {}
    return entities


def transform_aws_job_output_to_entities(df_ner, ner_column='rent-ner'):
    return  (df_ner[['source']]
             .join(
                df_ner[ner_column]
                .apply(lambda x: x['annotations']['entities'])
                .explode()
                .dropna()
                .apply(pd.Series)
             )
             .fillna({'startOffset': 0, 'endOffset': 0})
             .astype({'startOffset': 'int', 'endOffset': 'int'})
             .assign(ent_text=lambda df: 
                     df.apply(lambda row:
                              row['source'][row['startOffset']: row['endOffset']], axis=1)
                    )
             .drop(columns=['source'])
            )


def highlight_entities_in_text(text, entities, colors=LABELS_COLOR_MAP, title=None):
    """
    entities: {"start": 0, "end": 4, "label": 'eco'}
    """
    from spacy.displacy.render import EntityRenderer
    entities = sorted(entities, key=lambda x: x['start'])
    renderer = EntityRenderer({'colors': colors})
    res = renderer.render_ents(text=text, spans=entities, title=title)
    return HTML(res)


import plotly.express as px
from sklearn.metrics import classification_report, precision_recall_curve, precision_recall_fscore_support, confusion_matrix
def plot_cm(y_true, y_pred):
    labels = sorted(pd.unique(y_true))
    df = pd.DataFrame(confusion_matrix(y_true, y_pred), index=labels, columns=labels)
    return px.imshow(df, labels={'y': 'actual', 'x': 'pred'}, text_auto=True)


# def transform_token_df_to_entity_df(df_tok_pred, bio_tag_col='tag', start_col='start', end_col='end', sent_id_col='id'):
#     return (df_tok_pred
#      .query(f'{bio_tag_col} != "O"')
#      .assign(
#         label=lambda df: df[bio_tag_col].str.slice(2, None),
#         is_begin=lambda df: df[bio_tag_col].str.slice(0, 1) == 'B',
#         start=lambda df: df[start_col][df['is_begin']]
#      )
#      .ffill()
#      .groupby([sent_id_col, 'label', start_col])[end_col].max()
#      .reset_index()
#      .astype({'start': 'int'})
# )

def transform_token_df_to_entity_df(df_tok_pred, bio_tag_col='tag', start_col='start', end_col='end', sent_id_col='id',
                                   is_bio=True):
    return (df_tok_pred
     .assign(
        label=lambda df: df[bio_tag_col].str.slice(2, None) if is_bio else df[bio_tag_col],
        is_begin=lambda df: df[bio_tag_col].str.slice(0, 1) == 'B' if is_bio else (df[bio_tag_col] != 'O') & (df[bio_tag_col].shift() != df[bio_tag_col]),
        start=lambda df: df[start_col][df['is_begin']]
     )
     .query(f'{bio_tag_col} != "O"')
     .ffill()
     .groupby([sent_id_col, bio_tag_col, start_col])[end_col].max()
     .reset_index()
     .astype({'start': 'int'})
)