import numpy as np
from argparse import Namespace
import allosaurus
from allosaurus.app import read_recognizer
from allosaurus.audio import read_audio
from allosaurus.pm.factory import read_pm
from allosaurus.model import resolve_model_name, get_all_models, get_model_path
import re

from pydub import AudioSegment
import random

MODEL = read_recognizer('latest')
PHONE = ['I', 'a', 'aː', 'ã', 'ă', 'b', 'bʲ', 'bʲj', 'bʷ', 'bʼ', 'bː', 'b̞', 'b̤', 'b̥', 'c', 'd', 'dʒ', 'dʲ', 'dː', 'd̚', 'd̥', 'd̪', 'd̯', 'd͡z', 'd͡ʑ', 'd͡ʒ', 'd͡ʒː', 'd͡ʒ̤', 'e', 'eː', 'e̞', 'f', 'fʲ', 'fʷ', 'fː', 'g', 'gʲ', 'gʲj', 'gʷ', 'gː', 'h', 'hʷ', 'i', 'ij', 'iː', 'i̞', 'i̥', 'i̯', 'j', 'k', 'kx', 'kʰ', 'kʲ', 'kʲj', 'kʷ', 'kʷʼ', 'kʼ', 'kː', 'k̟ʲ', 'k̟̚', 'k͡p̚', 'l', 'lʲ', 'lː', 'l̪', 'm', 'mʲ', 'mʲj', 'mʷ', 'mː', 'n', 'nj', 'nʲ', 'nː', 'n̪', 'n̺', 'o', 'oː', 'o̞', 'o̥', 'p', 'pf', 'pʰ', 'pʲ', 'pʲj', 'pʷ', 'pʷʼ', 'pʼ', 'pː', 'p̚', 'q', 'r', 'rː', 's', 'sʲ', 'sʼ', 'sː', 's̪', 't', 'ts', 'tsʰ', 'tɕ', 'tɕʰ', 'tʂ', 'tʂʰ', 'tʃ', 'tʰ', 'tʲ', 'tʷʼ', 'tʼ', 'tː', 't̚', 't̪', 't̪ʰ', 't̪̚', 't͡s', 't͡sʼ', 't͡ɕ', 't͡ɬ', 't͡ʃ', 't͡ʃʲ', 't͡ʃʼ', 't͡ʃː', 'u', 'uə', 'uː', 'u͡w', 'v', 'vʲ', 'vʷ', 'vː', 'v̞', 'v̞ʲ', 'w', 'x', 'x̟ʲ', 'y', 'z', 'zj', 'zʲ', 'z̪', 'ä', 'æ', 'ç', 'çj', 'ð', 'ø', 'ŋ', 'ŋ̟', 'ŋ͡m', 'œ', 'œ̃', 'ɐ', 'ɐ̞', 'ɑ', 'ɑ̱', 'ɒ', 'ɓ', 'ɔ', 'ɔ̃', 'ɕ', 'ɕː', 'ɖ̤', 'ɗ', 'ə', 'ɛ', 'ɛ̃', 'ɟ', 'ɡ', 'ɡʲ', 'ɡ̤', 'ɡ̥', 'ɣ', 'ɣj', 'ɤ', 'ɤɐ̞', 'ɤ̆', 'ɥ', 'ɦ', 'ɨ', 'ɪ', 'ɫ', 'ɯ', 'ɯ̟', 'ɯ̥', 'ɰ', 'ɱ', 'ɲ', 'ɳ', 'ɴ', 'ɵ', 'ɸ', 'ɹ', 'ɹ̩', 'ɻ', 'ɻ̩', 'ɽ', 'ɾ', 'ɾj', 'ɾʲ', 'ɾ̠', 'ʀ', 'ʁ', 'ʁ̝', 'ʂ', 'ʃ', 'ʃʲː', 'ʃ͡ɣ', 'ʈ', 'ʉ̞', 'ʊ', 'ʋ', 'ʋʲ', 'ʌ', 'ʎ', 'ʏ', 'ʐ', 'ʑ', 'ʒ', 'ʒ͡ɣ', 'ʔ', 'ʝ', 'ː', 'β', 'β̞', 'θ', 'χ', 'ә', 'ḁ']

def aug_transcripts(path):
	v = AudioSegment.from_wav(path)
	speed = random.random() > 0.5
	vol = random.random() > 0.5
	if speed:
		auged = v.speedup(1.6, 150,25)
		if vol: 
			auged = auged + 5
		auged.export('sped_up.wav',format='wav')
		path = 'sped_up.wav'
    trans = MODEL.recognize(path)
    enc = np.array([PHONE.index(x) for x in trans.split(" ")])
    return enc

def get_transcript(path):
    trans = MODEL.recognize(path)
    enc = np.array([PHONE.index(x) for x in trans.split(" ")])
    return enc
def encode(transcript):
    '''
    accepts a string transcript and returns a one hot encoded matrix
    for each phone of the transcript
    '''
    phone_list = transcript.split(" ")
    one_hot = np.zeros((len(phone_list), len(PHONE)))
    
    for idx, phone in enumerate(phone_list):
        phone_loc = PHONE.index(phone)
        one_hot[idx][phone_loc] = 1
    
    return one_hot

def recognize(audio_path, feats=False):
    '''
    accepts audio file path and returns the transcript
    '''
    if feats == False:
        return MODEL.recognize(audio_path)
    else:
        model_path = get_model_path("latest")
        model_name = resolve_model_name("latest", None)
        inference_config = Namespace(model=model_name, device_id=-1, lang='ipa', approximate=False, prior=None)
        pm = read_pm(model_path, inference_config)
        audio = read_audio(audio_path)
        return pm.compute(audio)


def list_phones():
    return PHONE
def get_less_probable(path, how_close=2):
    output = MODEL.recognize(path, topk=how_close)
    pattern = r"\([^()]*\)"
    output = re.sub(pattern, ",", output).split(" , | ")
    new_out = []
    for strin in output:
        phone = strin.split(" , ")[-1]
        if phone in PHONE:
            new_out.append(PHONE.index(phone))
        else:
            new_out.append(PHONE.index(strin.split(" , ")[0]))
    return np.array(new_out)
