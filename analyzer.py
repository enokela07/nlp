import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import streamlit as st
import torch
import math


@st.cache_resource
def load_models():

    #load spacy
    nlp = spacy.load("en_core_web_sm")

    #load summarizer
    sum_id = "/home/isaac/Desktop/nlp_project/models/distilbart-cnn-6-6"
    sum_tok = AutoTokenizer.from_pretrained(sum_id, local_files_only=True)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_id, local_files_only=True)

    #load sentiment score guy
    sent_id = "/home/isaac/Desktop/nlp_project/models/distilbert-sst2"
    sent_tok = AutoTokenizer.from_pretrained(sent_id, local_files_only=True, use_fast=False)
    sent_model = AutoModelForSequenceClassification.from_pretrained(sent_id, local_files_only=True)

    return nlp, sum_tok, sum_model, sent_tok, sent_model

def get_reading_stats(doc):
    words = [token for token in doc if not token.is_punct and not token.is_space]
    sentence_count = len(list(doc.sents))
    word_count = len(words)
    reading_time = max(1, math.ceil(word_count/200)) if word_count else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "reading_time": reading_time
    }

def get_sentiment_label(sentiment_score):
    confidence = max(sentiment_score, 1-sentiment_score)

    if 0.4 <= sentiment_score <= 0.6:
        return {
            "label": "neutral",
            "confidence": 1 - abs(sentiment_score-0.5)*2,
            "color": "neutral"
        }

    if sentiment_score > 0.6:
        return {
            "label": "positive",
            "confidence": confidence,
            "color": "positive"
        }

    return {
        "label": "negative",
        "confidence": confidence,
        "color": "negative"
    }

#analyze docs
def analyze_document(text_input, nlp_model, trans_tokenizer, trans_model, sent_mod, sent_tok):

    #spacy summary
    doc = nlp_model(text_input)

    allowed_pos = ["NOUN", "PROPN", "ADJ"]
    word_freq_dist = {}
    for token in doc:
        if token.pos_ in allowed_pos:
            cleaned_word = token.lemma_.lower()

            if cleaned_word not in word_freq_dist.keys():
                word_freq_dist[cleaned_word] = 1
            else:
                word_freq_dist[cleaned_word]+=1

    #spacy summary sentence scoring
    sentence_scores = {}
    for sent in doc.sents:
        score = 0

        for word in sent:
            word_cleaned = word.lemma_.lower()
            if word_cleaned in word_freq_dist:
                score+= word_freq_dist[word_cleaned]

        word_count = len([t for t in sent if not t.is_punct and not t.is_space])

        if word_count>0:
            sentence_scores[sent] = score/word_count

    sorted_sents = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)

    summary_sent_count = min(5, max(2, math.ceil(len(sentence_scores)*0.3)))
    important_sentences = [sent for sent, score in sorted_sents[:summary_sent_count]]
    important_sentences = sorted(important_sentences, key=lambda sent: sent.start_char)

    if len(important_sentences) >= 1:
        spacy_summary = " ".join([sent.text for sent in important_sentences])
    elif len(list(doc.sents)) == 1:
        spacy_summary = list(doc.sents)[0].text
    else:
        spacy_summary = ""

    #sentiment analysis
    sentence_count = 0
    total_positive_score = 0
    for sent in doc.sents:
        sent_input = sent_tok(sent.text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            sent_outputs = sent_mod(**sent_input)

        probabilities = torch.nn.functional.softmax(sent_outputs.logits, dim=-1)

        #extract pos score
        pos_score = probabilities[0][1].item()

        sentence_count += 1
        total_positive_score += pos_score

    sentiment_score = total_positive_score/sentence_count if sentence_count>0 else 0.5
    sentiment = get_sentiment_label(sentiment_score)

    #transformer summary
    model_limit = min(
        getattr(trans_tokenizer, "model_max_length", 1024),
        getattr(trans_model.config, "max_position_embeddings", 1024)
    )
    token_count = len(trans_tokenizer(text_input, truncation=False).input_ids)
    length_warning = None

    if token_count > model_limit:
        transformer_summary = ""
        length_warning = f"This document is {token_count} tokens, but the transformer summary model can safely handle {model_limit}. Shorten the document to generate a transformer summary."
    else:
        inputs = trans_tokenizer(text_input, return_tensors="pt", truncation=False).input_ids
        input_length = inputs.shape[1]
        max_summary_length = min(300, max(80, int(input_length*0.6)))
        min_summary_length = min(120, max(30, int(input_length*0.25)))
        outputs = trans_model.generate(
            inputs,
            max_new_tokens=max_summary_length,
            min_new_tokens=min_summary_length,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        transformer_summary = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "spacy_summary": spacy_summary,
        "transformer_summary": transformer_summary,
        "top_words": sorted(word_freq_dist.items(), key=lambda x: x[1], reverse=True)[:5],
        "sentiment_score": sentiment_score,
        "sentiment": sentiment,
        "important_sentences": [
            {
                "text": sent.text,
                "start": sent.start_char,
                "end": sent.end_char
            }
            for sent in important_sentences
        ],
        "stats": get_reading_stats(doc),
        "token_count": token_count,
        "token_limit": model_limit,
        "length_warning": length_warning
    }



