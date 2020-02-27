from easydict import EasyDict as edict

config = edict()

# configurations for LDA.
config.N_TOPICS=14

config.DATA_DIR = './data/toy_data'

config.TOKENS_FILE= 'toy.tokens'

config.DATA_FILE='toy.ldac'

config.N_ITER=2000

config.ALPHA=0.01

config.BETA=0.1
