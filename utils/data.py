import numpy as np
from argparse import Namespace
import allosaurus
from allosaurus.app import read_recognizer
from allosaurus.audio import read_audio
from allosaurus.pm.factory import read_pm
from allosaurus.model import resolve_model_name, get_all_models, get_model_path
import re
import os

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
    os.remove('sped_up.wav')
    return enc

def get_transcript(path, similar, num_similar):
    trans = MODEL.recognize(path)
    enc = np.array([PHONE.index(x) for x in trans.split(" ")])
    if similar:
        ixs = np.random.randint(low=1, high=len(enc), size=1)
        for ix in ixs:
            enc[ix] = SIMILAR[enc[ix]]
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



SIMILAR = {1: 92, 2: 197, 3: 137, 4: 23,
 5: 219,
 6: 204,
 7: 205,
 8: 47,
 9: 138,
 10: 73,
 11: 209,
 12: 59,
 13: 167,
 14: 197,
 15: 205,
 16: 187,
 17: 137,
 18: 211,
 19: 94,
 20: 221,
 21: 68,
 22: 40,
 23: 65,
 24: 137,
 25: 213,
 26: 115,
 27: 209,
 28: 4,
 29: 164,
 30: 133,
 31: 133,
 32: 173,
 33: 202,
 34: 37,
 35: 40,
 36: 100,
 37: 213,
 38: 29,
 39: 188,
 40: 106,
 41: 157,
 42: 167,
 43: 164,
 44: 214,
 45: 209,
 46: 167,
 47: 8,
 48: 223,
 49: 142,
 50: 32,
 51: 212,
 52: 200,
 53: 23,
 54: 137,
 55: 226,
 56: 137,
 57: 214,
 58: 109,
 59: 12,
 60: 215,
 61: 37,
 62: 9,
 63: 137,
 64: 184,
 65: 23,
 66: 185,
 67: 220,
 68: 21,
 69: 211,
 70: 2,
 71: 8,
 72: 205,
 73: 12,
 74: 96,
 75: 69,
 76: 133,
 77: 167,
 78: 213,
 79: 226,
 80: 167,
 81: 76,
 82: 112,
 83: 114,
 84: 137,
 85: 214,
 86: 211,
 87: 35,
 88: 205,
 89: 18,
 90: 207,
 91: 124,
 92: 2,
 93: 4,
 94: 197,
 95: 154,
 96: 184,
 97: 150,
 98: 46,
 99: 215,
 100: 202,
 101: 102,
 102: 167,
 103: 201,
 104: 2,
 105: 39,
 106: 40,
 107: 13,
 108: 164,
 109: 121,
 110: 138,
 111: 221,
 112: 120,
 113: 137,
 114: 184,
 115: 26,
 116: 40,
 117: 137,
 118: 219,
 119: 137,
 120: 112,
 121: 109,
 122: 199,
 123: 139,
 124: 75,
 125: 42,
 126: 111,
 127: 70,
 128: 164,
 129: 211,
 130: 197,
 131: 95,
 132: 213,
 133: 137,
 134: 15,
 135: 97,
 136: 142,
 137: 54,
 138: 9,
 139: 40,
 140: 138,
 141: 202,
 142: 200,
 143: 162,
 144: 197,
 145: 124,
 146: 116,
 147: 221,
 148: 197,
 149: 219,
 150: 97,
 151: 12,
 152: 193,
 153: 211,
 154: 95,
 155: 162,
 156: 202,
 157: 41,
 158: 147,
 159: 164,
 160: 42,
 161: 124,
 162: 69,
 163: 213,
 164: 187,
 165: 110,
 166: 13,
 167: 13,
 168: 137,
 169: 13,
 170: 164,
 171: 153,
 172: 99,
 173: 32,
 174: 27,
 175: 8,
 176: 164,
 177: 53,
 178: 47,
 179: 133,
 180: 17,
 181: 223,
 182: 221,
 183: 21,
 184: 64,
 185: 197,
 186: 197,
 187: 164,
 188: 39,
 189: 124,
 190: 63,
 191: 27,
 192: 148,
 193: 152,
 194: 215,
 195: 54,
 196: 193,
 197: 167,
 198: 129,
 199: 122,
 200: 142,
 201: 103,
 202: 156,
 203: 39,
 204: 137,
 205: 75,
 206: 13,
 207: 190,
 208: 194,
 209: 27,
 210: 142,
 211: 69,
 212: 23,
 213: 163,
 214: 44,
 215: 69,
 216: 158,
 217: 205,
 218: 167,
 219: 197,
 220: 67,
 221: 147,
 222: 197,
 223: 197,
 224: 88,
 225: 164,
 226: 213,
 227: 70,
 228: 2}