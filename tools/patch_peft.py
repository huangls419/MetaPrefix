# tools/patch_peft.py
import peft
import peft.tuners.prefix_tuning as prefix_tuning

from peft_patch.prefix_tuning import PrefixEncoder, PrefixTuningConfig

prefix_tuning.PrefixEncoder = PrefixEncoder
prefix_tuning.PrefixTuningConfig = PrefixTuningConfig

peft.PrefixEncoder = PrefixEncoder
peft.PrefixTuningConfig = PrefixTuningConfig