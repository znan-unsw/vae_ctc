#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:06:30 2022

@author: nanzheng
"""

SOS = "h#"
EOS = "<eos>"
PAD = "[PAD]"

SAMPLE_RATE = 16000

from_60_to_39_phn = {}
from_60_to_39_phn["sil"] = "sil"
from_60_to_39_phn["aa"] = "aa"
from_60_to_39_phn["ae"] = "ae"
from_60_to_39_phn["ah"] = "ah"
from_60_to_39_phn["ao"] = "aa"
from_60_to_39_phn["aw"] = "aw"
from_60_to_39_phn["ax"] = "ah"
from_60_to_39_phn["ax-h"] = "ah"
from_60_to_39_phn["axr"] = "er"
from_60_to_39_phn["ay"] = "ay"
from_60_to_39_phn["b"] = "b"
from_60_to_39_phn["bcl"] = "sil"
from_60_to_39_phn["ch"] = "ch"
from_60_to_39_phn["d"] = "d"
from_60_to_39_phn["dcl"] = "sil"
from_60_to_39_phn["dh"] = "dh"
from_60_to_39_phn["dx"] = "dx"
from_60_to_39_phn["eh"] = "eh"
from_60_to_39_phn["el"] = "l"
from_60_to_39_phn["em"] = "m"
from_60_to_39_phn["en"] = "n"
from_60_to_39_phn["eng"] = "ng"
from_60_to_39_phn["epi"] = "sil"
from_60_to_39_phn["er"] = "er"
from_60_to_39_phn["ey"] = "ey"
from_60_to_39_phn["f"] = "f"
from_60_to_39_phn["g"] = "g"
from_60_to_39_phn["gcl"] = "sil"
from_60_to_39_phn["h#"] = "sil"
from_60_to_39_phn["hh"] = "hh"
from_60_to_39_phn["hv"] = "hh"
from_60_to_39_phn["ih"] = "ih"
from_60_to_39_phn["ix"] = "ih"
from_60_to_39_phn["iy"] = "iy"
from_60_to_39_phn["jh"] = "jh"
from_60_to_39_phn["k"] = "k"
from_60_to_39_phn["kcl"] = "sil"
from_60_to_39_phn["l"] = "l"
from_60_to_39_phn["m"] = "m"
from_60_to_39_phn["ng"] = "ng"
from_60_to_39_phn["n"] = "n"
from_60_to_39_phn["nx"] = "n"
from_60_to_39_phn["ow"] = "ow"
from_60_to_39_phn["oy"] = "oy"
from_60_to_39_phn["p"] = "p"
from_60_to_39_phn["pau"] = "sil"
from_60_to_39_phn["pcl"] = "sil"
from_60_to_39_phn["q"] = ""
from_60_to_39_phn["r"] = "r"
from_60_to_39_phn["s"] = "s"
from_60_to_39_phn["sh"] = "sh"
from_60_to_39_phn["t"] = "t"
from_60_to_39_phn["tcl"] = "sil"
from_60_to_39_phn["th"] = "th"
from_60_to_39_phn["uh"] = "uh"
from_60_to_39_phn["uw"] = "uw"
from_60_to_39_phn["ux"] = "uw"
from_60_to_39_phn["v"] = "v"
from_60_to_39_phn["w"] = "w"
from_60_to_39_phn["y"] = "y"
from_60_to_39_phn["z"] = "z"
from_60_to_39_phn["zh"] = "sh"

from_60_to_39_phn[EOS] = "sil"

list_phoneme = [
    'aa',
    'ae',
    'ah',
    'ao',
    'aw',
    'ax',
    'ax-h',
    'axr',
    'ay',
    'b',
    'bcl',
    'ch',
    'd',
    'dcl',
    'dh',
    'dx',
    'eh',
    'el',
    'em',
    'en',
    'eng',
    'epi',
    'er',
    'ey',
    'f',
    'g',
    'gcl',
    'h#',
    'hh',
    'hv',
    'ih',
    'ix',
    'iy',
    'jh',
    'k',
    'kcl',
    'l',
    'm',
    'n',
    'ng',
    'nx',
    'ow',
    'oy',
    'p',
    'pau',
    'pcl',
    'q',
    'r',
    's',
    'sh',
    't',
    'tcl',
    'th',
    'uh',
    'uw',
    'ux',
    'v',
    'w',
    'y',
    'z',
    'zh'
]
dict_phoneme = dict(zip(list_phoneme, [i for i in range(len(list_phoneme))]))

