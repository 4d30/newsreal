#!/bin/env /bin/python
""" Load and model news with BERT """
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

AUTOTUNE = tf.data.AUTOTUNE
BATCHSIZE = 32
SEED = 42
VALSPLIT = 0.15

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    './input/train/',
	labels = 'inferred',
	label_mode = 'binary',
	batch_size = BATCHSIZE,
	seed = SEED,
	validation_split = VALSPLIT,
	subset = 'training')

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size = AUTOTUNE)

val_ds= tf.keras.preprocessing.text_dataset_from_directory(
    './input/train/',
	labels = 'inferred',
	label_mode = 'binary',
	batch_size = BATCHSIZE,
	seed = SEED,
	validation_split = VALSPLIT,
	subset = 'training')

val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    './input/test',
    batch_size = BATCHSIZE)

test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)

for text_batch, label_batch in train_ds.take(1):
    for i in range(5):
        print(f'Review: {text_batch.numpy()[i]}')
        label = label_batch.numpy()[i]
        print(f'Label: {label} {class_names[int(label)]}\n\n ')

TFHUB_HANDLE_ENCODER= 'https://tfhub.dev.tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1'
TFHUB_HANDLE_PREPROCESS = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

bert_preprocess_model = hub.KerasLayer(TFHUB_HANDLE_PREPROCESS)

TEST_TEXT= ['is this why we worked so hard to get a gop majority in the house and senate? spineless weasels republican house members blasted their gop leadership wednesday for caving to democratic leaders  demand that they abandon the house s bipartisan bill to tighten screening of refugee applicants from syria and iraq. the bill to bolster the syrian and iraqi [refugee] vetting process passed with a veto-proof majority and is supported by a vast majority of americans,  said rep. richard hudson (r-nc)57% , who is one of the two co-authors of the original house bill that gop leaders are now abandoning. it would be a shame to bow to the wishes of sen. harry reid (d-nv) and the white house, leaving our country vulnerable to foreign terrorism,  he told breitbart news exclusively.after the paris attacks that killed over 120 people, the house passed a bill   with a huge bipartisan majority   that would implement tough screening requirements for refugees. the bill was criticized by democrats and president obama, who vowed to veto the legislation and suggested that republicans work with them to tighten oversight of visas given to legal visitors.now, house republican leaders are reportedly planning to abandon their original idea to tighten screening of refugees and are instead caving to democrats who want to only fix security gaps in the legal-visitor visa program.republican house leadership is backing away from the refugee problem because they fear their fix will be rejected by democrats and lead to a partial government shut-down.rep. mo brooks (r-al) also told breitbart news exclusively that he isn t  surprised  that rep. paul ryan (r-wi) is following in the steps of former house speaker rep. john boehner (r-oh)32% by bowing to democrats, rather than putting up a fight.he called the original vote  a show vote  that was arranged by the leaders, and told breitbart news that the entire refugee program needs to be stripped of its funding.rep. brian babin (r-tx) told breitbart news exclusively that gop house leadership should fix both problems.  this should not be an either or question,  babin said.  both should be addressed in a timely manner. we should concern ourselves first and foremost with the safety and security of the american people. babin recently released a letter, signed by more than 70 of his colleagues, calling on house leadership to include language in the must-pass omnibus spending bill that would defund the refugee resettlement program. more than 70 of my colleagues joined me in calling for an immediate suspension in the refugee program to give us time to put in place security measures that the administration has failed to implement  babin told breitbart news. we must restore congressional authority and ensure that congress has a final sign-off on any administration proposed security measures. congress has until december 11th to pass the omnibus bill.via: breitbart news']

text_preprocessed = bert_preprocess_model(TEST_TEXT)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')



#def build_classifier_model():
#    """ input -> preproc -> enc -> dropout -> dense """
#    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
#    preprocessing_layer = hub.KerasLayer(
#        TFHUB_HANDLE_PREPROCESS,
#        trainable=True,
#        name='preprocessing')
#    encoder_inputs = preprocessing_layer(text_input)
#    encoder = hub.KerasLayer(TFHUB_HANDLE_ENCODER, trainable=True, name='BERT_encoder')
#    outputs = encoder(encoder_inputs)
#    net = outputs['pooled_outputs']
#    net = tf.keras.layers.Dropout(0.1)(net)
#    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
#    return tf.keras.Model(text_input, net)
#
#classifier_model = build_classifier_model()
